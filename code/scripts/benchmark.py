from __future__ import annotations
import sys
import argparse
from pathlib import Path
import pandas as pd
from biorxngen.paper_benchmark import (
    replace_reranker_scores,
    summarize_heldout_policies,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "benchmark_results"


def internal(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Recompute the held-out Fig. 2 metrics from released scored rows."
    )
    parser.add_argument(
        "--scored",
        type=Path,
        default=BENCHMARK / "internal_test_scored_candidate_rows.parquet",
    )
    parser.add_argument(
        "--reranker-scores",
        type=Path,
        help="Fresh reranker scores, retaining candidate order",
    )
    parser.add_argument(
        "--split", type=Path, default=BENCHMARK / "internal_split_manifest.csv"
    )
    parser.add_argument(
        "--reference", type=Path, default=BENCHMARK / "internal_topk_metric_summary.csv"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional TSV for the recomputed metrics"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if a reported metric does not match the source summary",
    )
    args = parser.parse_args(argv)
    scored = pd.read_parquet(args.scored)
    if args.reranker_scores:
        scored = replace_reranker_scores(scored, pd.read_parquet(args.reranker_scores))
    summary = summarize_heldout_policies(scored, pd.read_csv(args.split))
    if args.check:
        reference = pd.read_csv(args.reference)
        public_policy_names = {
            "candidate_set": "candidate_set",
            "template_rank": "template_rule",
            "enzymemap_reranker": "EnzymeMap_derived",
            "agora2_strict_reranker": "BioRxnGen_rerank",
        }
        reference = reference.loc[
            reference["dataset_scope"].eq("test")
            & reference["label_col"].eq("exact_agora_norm_no_proton_water")
            & reference["policy"].isin(public_policy_names.values())
        ]
        for row in summary.itertuples(index=False):
            expected = reference.loc[
                reference["policy"].eq(public_policy_names[row.policy])
                & reference["topk"].astype(str).eq(str(row.topk))
            ]
            if len(expected) != 1:
                raise RuntimeError(
                    f"Missing or duplicate reference row: {row.policy}, {row.topk}"
                )
            source = expected.iloc[0]
            for field in ("exact_rate", "mean_max_tanimoto", "median_max_tanimoto"):
                if abs(float(getattr(row, field)) - float(source[field])) > 1e-10:
                    raise RuntimeError(
                        f"Mismatch in {row.policy} top-{row.topk} {field}"
                    )
        print("All released held-out metrics match the reference summary.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, sep="\t", index=False)
    print(
        summary[
            ["policy", "topk", "exact_hits", "n_reactions", "exact_rate"]
        ].to_string(index=False)
    )


import json
import numpy as np
from biorxngen.external_workflow import (
    generate_external_candidates,
    make_prediction_rows,
    select_benchmark_units,
)
from biorxngen.production_rules import rule_modules, rule_paths
from biorxngen.reranker import load_model, score_candidates

SELECTION = {
    "max_benchmark_reactions": 1000,
    "max_per_ec_combo": 30,
    "min_external_support_models": 1,
    "require_micromap": True,
}
GENERATION = {
    "original": {
        "rule_topk_per_ec": 20,
        "max_total_rules": 40,
        "max_products_per_rule": 24,
        "max_products_per_reaction": 256,
        "max_reactant_permutations": 96,
    },
    "residual": {
        "rule_topk_per_ec": 80,
        "max_total_rules": 160,
        "max_products_per_rule": 64,
        "max_products_per_reaction": 1024,
        "max_reactant_permutations": 256,
    },
}


def external(argv=None):
    parser = argparse.ArgumentParser(
        description="Rebuild external benchmark units, generate candidates or run frozen-checkpoint inference"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--operation", choices=["selection", "generation", "inference"], required=True
    )
    parser.add_argument("--profile", choices=list(GENERATION), default="original")
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override the separate external benchmark checkpoint",
    )
    args = parser.parse_args(argv)
    root, output = (args.data_root.resolve(), args.output.resolve())
    if args.max_units < 0:
        parser.error("--max-units must be nonnegative")
    if output == root or root in output.parents:
        parser.error("Write outputs outside the immutable data deposit")
    output.mkdir(parents=True, exist_ok=True)
    folder = (
        root
        / "external_benchmark"
        / (
            "candidate_recovery"
            if args.profile == "original"
            else "relaxed_generation/apollo_residual_relaxed_generation"
        )
    )
    units = pd.read_parquet(
        folder
        / (
            "benchmark_units.parquet"
            if args.profile == "original"
            else "residual_benchmark_units.parquet"
        )
    )
    if args.max_units:
        units = units.head(args.max_units)
    report = {"operation": args.operation, "profile": args.profile, "units": len(units)}
    if args.operation == "selection":
        if args.profile != "original" or args.max_units:
            parser.error("Selection checks require the full original cohort")
        selection_dir = root / "external_benchmark/input_selection"
        selected, selection_report = select_benchmark_units(
            pd,
            rule_modules(),
            selection_dir / "complete_ready_strict_all_side_worklist.tsv",
            selection_dir / "apollo_unique_reaction_summary.tsv",
            SELECTION,
        )
        if args.check:
            pd.testing.assert_frame_equal(
                selected[units.columns], units.reset_index(drop=True), check_dtype=False
            )
        selected.to_parquet(output / "benchmark_units.parquet", index=False)
        report.update(selection_report)
        report.pop("worklist_path", None)
    else:
        reference = pd.read_parquet(folder / "generated_candidate_rows.parquet")
        reference = reference.loc[
            reference.benchmark_id.isin(units.benchmark_id)
        ].reset_index(drop=True)
        if args.operation == "generation":
            candidates, attempts, generation_report = generate_external_candidates(
                pd,
                rule_modules(),
                units,
                rule_paths(root, output),
                GENERATION[args.profile],
            )
            candidates["split"] = "apollo_external"
            candidates["benchmark_id"] = candidates.get(
                "reaction_id", pd.Series(dtype=str)
            )
            candidates.to_parquet(
                output / "generated_candidate_rows.parquet", index=False
            )
            attempts.to_parquet(output / "generation_attempts.parquet", index=False)
            report.update(generation_report)
            (output / "generation_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            if args.check:
                compare = [
                    column
                    for column in candidates
                    if column in reference and column != "neural_score"
                ]
                pd.testing.assert_frame_equal(
                    candidates[compare].reset_index(drop=True),
                    reference[compare],
                    check_dtype=False,
                )
        else:
            checkpoint = (
                args.checkpoint
                or Path(__file__).resolve().parents[1]
                / "models/external_product_reranker.pt"
            )
            model, config = load_model(checkpoint, args.device)
            candidates = reference.copy()
            scores = score_candidates(candidates, model, config, device=args.device)
            report["maximum_score_difference"] = float(
                np.max(np.abs(scores - candidates.neural_score))
            )
            candidates["neural_score"] = scores
            predictions = pd.DataFrame(make_prediction_rows(candidates))
            if args.check:
                archived = pd.read_csv(folder / "candidate_predictions.tsv", sep="\t")
                archived = archived.loc[archived.benchmark_id.isin(units.benchmark_id)]
                key = ["benchmark_id", "policy", "rank", "predicted_product_smiles"]
                left = predictions[key].sort_values(key).reset_index(drop=True)
                right = archived[key].sort_values(key).reset_index(drop=True)
                pd.testing.assert_frame_equal(left, right, check_dtype=False)
            predictions.to_csv(
                output / "candidate_predictions.tsv", sep="\t", index=False
            )
            report["prediction_rows"] = len(predictions)
    report["reference_check"] = "passed" if args.check else "not_requested"
    (output / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


from biorxngen.external_benchmark import (
    RepresentationKeyer,
    add_metric_keys,
    build_benchmark_metadata,
    build_topk_summary,
    compute_rank_maps,
    load_pandas_and_rdkit,
)


def recovery(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Recompute external recovery under the archived molecular identity policies."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.data_root / "external_benchmark"
    predictions = pd.read_csv(
        root / "candidate_recovery/candidate_predictions.tsv",
        sep="\t",
        low_memory=False,
    )
    recovery = pd.read_csv(
        root / "candidate_recovery/reaction_level_recovery.tsv",
        sep="\t",
        low_memory=False,
    )
    units = pd.read_parquet(root / "candidate_recovery/benchmark_units.parquet")
    if len(units) != 1000 or units["benchmark_id"].duplicated().any():
        raise ValueError("Expected 1,000 unique external reaction units")
    if not set(predictions["benchmark_id"]).issubset(set(units["benchmark_id"])):
        raise ValueError("Prediction IDs are outside the benchmark")
    _, chem, descriptors, uncharger = load_pandas_and_rdkit()
    keys = RepresentationKeyer(chem, descriptors, uncharger)
    scored = add_metric_keys(predictions, keys)
    summary = pd.DataFrame(
        build_topk_summary(
            compute_rank_maps(scored), build_benchmark_metadata(recovery)
        )
    )
    if args.check:
        reference = pd.read_csv(
            root / "identity_diagnostics/representation_topk_summary.tsv", sep="\t"
        )
        join = ["policy", "metric", "denominator", "topk"]
        for frame in (summary, reference):
            frame["topk"] = frame["topk"].astype(str)
        compared = summary.merge(
            reference,
            on=join,
            validate="one_to_one",
            suffixes=("", "_reference"),
            how="outer",
            indicator=True,
        )
        if not compared["_merge"].eq("both").all():
            raise ValueError("Metric rows differ from the archived reference")
        for column in ("hits", "n_eval"):
            if not compared[column].eq(compared[column + "_reference"]).all():
                raise ValueError(f"External metric mismatch: {column}")
        print("All external recovery counts match the archived reference.")
    summary["scope"] = "original_candidate_set"
    residual_root = root / "relaxed_generation/apollo_residual_relaxed_generation"
    residual_predictions = pd.read_csv(
        residual_root / "candidate_predictions.tsv", sep="\t", low_memory=False
    )
    residual_recovery = pd.read_csv(
        residual_root / "reaction_level_recovery.tsv", sep="\t", low_memory=False
    )
    residual_units = pd.read_parquet(residual_root / "residual_benchmark_units.parquet")
    original_ranks = compute_rank_maps(scored)
    residual_ids = set(residual_units["benchmark_id"])
    if len(residual_ids) != 160 or not residual_ids.issubset(
        set(units["benchmark_id"])
    ):
        raise ValueError(
            "Expected 160 unique residual units within the external benchmark"
        )
    if residual_ids.intersection(
        original_ranks["hybrid_reranker", "neutral_formula_no_hw"]
    ):
        raise ValueError(
            "Residual units must lack formula-equivalent products in the original candidate set"
        )
    residual_summary = pd.DataFrame(
        build_topk_summary(
            compute_rank_maps(add_metric_keys(residual_predictions, keys)),
            build_benchmark_metadata(residual_recovery),
        )
    )
    residual_summary["scope"] = "targeted_relaxed_generation"
    if args.check:
        reference = pd.read_csv(
            residual_root / "residual_representation_topk_summary.tsv", sep="\t"
        )
        for frame in (residual_summary, reference):
            frame["topk"] = frame["topk"].astype(str)
        compared = residual_summary.merge(
            reference,
            on=join,
            validate="one_to_one",
            suffixes=("", "_reference"),
            how="outer",
            indicator=True,
        )
        if (
            not compared["_merge"].eq("both").all()
            or not compared["hits"].eq(compared["hits_reference"]).all()
        ):
            raise ValueError("Residual recovery differs from the archived reference")
    combined = []
    for metric in (
        "string_full",
        "string_no_proton_water",
        "neutral_component_no_hw",
        "neutral_formula_no_hw",
    ):
        for topk in ("20", "candidate_oracle"):

            def count(frame: pd.DataFrame) -> int:
                selected = frame.loc[
                    frame["policy"].eq("hybrid_reranker")
                    & frame["metric"].eq(metric)
                    & frame["denominator"].eq("all_benchmark_units")
                    & frame["topk"].astype(str).eq(topk)
                ]
                if len(selected) != 1:
                    raise ValueError("Missing or duplicated integrated recovery metric")
                return int(selected.iloc[0]["hits"])

            hits = count(summary) + count(residual_summary)
            combined.append(
                {
                    "scope": "original_plus_targeted_relaxed_generation",
                    "policy": "hybrid_reranker",
                    "metric": metric,
                    "denominator": "all_benchmark_units",
                    "topk": topk,
                    "n_eval": len(units),
                    "hits": hits,
                    "rate": hits / len(units),
                }
            )
    combined = pd.DataFrame(combined)
    if args.check:
        formula = combined.loc[
            combined["metric"].eq("neutral_formula_no_hw")
        ].set_index("topk")["hits"]
        if int(formula["20"]) != 710 or int(formula["candidate_oracle"]) != 743:
            raise ValueError(
                "Integrated formula-equivalent counts differ from the manuscript"
            )
        print(
            "Integrated formula-equivalent recovery: 710/1,000 top-20; 743/1,000 candidate-set coverage."
        )
    summary = pd.concat([summary, residual_summary, combined], ignore_index=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, sep="\t", index=False)
    display = summary.loc[
        summary["policy"].eq("hybrid_reranker")
        & summary["denominator"].eq("all_benchmark_units")
        & summary["topk"].astype(str).eq("20")
    ]
    print(display[["scope", "metric", "hits", "n_eval", "rate"]].to_string(index=False))


COMMANDS = {"internal": internal, "external": external, "recovery": recovery}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Evaluate internal and external reaction benchmarks"
    )
    parser.add_argument("command", choices=COMMANDS)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.parse_args(arguments)
    command = parser.parse_args(arguments[:1]).command
    COMMANDS[command](arguments[1:])


if __name__ == "__main__":
    main()
