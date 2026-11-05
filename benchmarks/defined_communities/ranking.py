"""Apply the frozen Fig. 2 reranker to the defined-community candidate pools."""

import argparse
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from biorxngen.reranker import load_model, score_candidates

from benchmark import CODE, GROUPS, RESULTS, check_archive, read_table
from structures import side_keys

CHECKPOINT = CODE / "models/product_reranker.pt"
TOP_K = (1, 5, 10, 20)
IDENTITY = [
    "model_id", "candidate_edge_id", "reactant_context_name", "reactant_side_smiles",
    "generated_product_side_smiles", "candidate_rule_directional_id",
]
GENERATION_FIELDS = [
    *IDENTITY, "ec_number", "substrate_smiles", "generated_product_no_proton_water_smiles",
    "product_full_nonisomeric", "n_product_fragments", "rule_rank", "rule_source_key",
    "rule_source_depth", "rule_train_count", "rule_smarts_count", "product_rank_within_rule",
    "required_reactants", "available_reactants", "permutations_tried", "product_sets_returned",
    "template_product_cap_hit",
]
TIE_COLUMNS = [
    "reactant_side_smiles", "reactant_context_name", "candidate_rule_directional_id",
    "product_full_nonisomeric", "generated_product_side_smiles",
]


def infer(rows, model, config):
    features = rows.copy()
    levels = features.ec_number.str.split(".")
    for level in range(1, 5):
        features[f"ec{level}"] = levels.str[:level].str.join(".")
    features["rule_directional_id"] = features.candidate_rule_directional_id
    features["rule_family_id"] = features.rule_directional_id.str.split("__", n=1).str[0]
    features["product_smiles_len"] = features.product_full_nonisomeric.str.len()
    features["product_no_pw_smiles_len"] = features.generated_product_no_proton_water_smiles.str.len()
    features["true_rule_hit_topk"] = 0
    if features[GENERATION_FIELDS].isna().any().any() or features.product_full_nonisomeric.eq("").any():
        raise ValueError("Missing reranker inputs")
    return score_candidates(features, model, config)


def assign_ranks(rows):
    for rank, keys in {
        "candidate_rank": ["model_id", "candidate_edge_id"],
        "reactant_context_rank": ["model_id", "candidate_edge_id", "reactant_side_smiles"],
    }.items():
        columns = [*keys, "reranker_score", "rule_rank", "product_rank_within_rule"]
        columns += [c for c in TIE_COLUMNS if c not in keys]
        ascending = [name != "reranker_score" for name in columns]
        rows[rank] = rows.sort_values(columns, ascending=ascending).groupby(keys, sort=False).cumcount() + 1
    return rows


def assay_ranks(rows, assays):
    ranks = ("candidate_rank", "reactant_context_rank")
    index = {rank: defaultdict(dict) for rank in ranks}
    fields = ["model_id", "label", "generated_product_side_smiles", "reactant_side_smiles", *ranks]
    for model, label, products, reactants, *values in rows[fields].itertuples(index=False, name=None):
        for key in side_keys(products, label) - side_keys(reactants, label):
            for name, value in zip(ranks, values):
                model_ranks = index[name][model]
                model_ranks[key] = min(value, model_ranks.get(key, value))
    out = []
    for group in GROUPS:
        cohort = group["cohort"]
        for row in assays[assays.cohort.eq(cohort)].itertuples(index=False):
            items = row.structure_key.split("|")
            best = {}
            for name in ranks:
                best[name] = min(
                    (index[name][model][key] for model in group["models"]
                     for key in items if key in index[name][model]),
                    default=None,
                )
            if (best["candidate_rank"] is not None) != row.bio_ec:
                raise ValueError(f"Candidate ceiling mismatch for {cohort}/{row.analyte}")
            models = sorted(
                model for model in group["models"]
                if any(key in index["candidate_rank"][model] and
                       index["candidate_rank"][model][key] == best["candidate_rank"] for key in items)
            )
            out.append({
                "study": row.study, "cohort": cohort, "analyte": row.analyte,
                "structure_key": row.structure_key, "valid": row.valid, "enriched": row.enriched,
                "best_rank": best["candidate_rank"], "best_reactant_context_rank": best["reactant_context_rank"],
                "best_rank_models": "|".join(models), "native_gem": row.gem_community,
                "native_reaction_products": row.native_graph, "native_exchange_mapped": row.native_exchange_mapped,
            })
    frame = pd.DataFrame(out)
    return frame.astype({"best_rank": "Int64", "best_reactant_context_rank": "Int64"})


def coverage(analytes, candidates):
    records = []
    positives = analytes[analytes.valid & analytes.enriched]
    for panel, panel_rows in {
        "enriched": positives,
        "enriched_exchange_mapped": positives[positives.native_exchange_mapped],
    }.items():
        for group_type, groups in [
            ("combined", [("all_three", panel_rows)]),
            ("study", panel_rows.groupby("study", sort=True)),
            ("community", panel_rows.groupby("cohort", sort=True)),
        ]:
            for group, subset in groups:
                for unit, ranks in (("generation_input", subset.best_rank),
                                    ("reactant_context", subset.best_reactant_context_rank)):
                    for k in (*TOP_K, "all"):
                        hits = ranks.notna() if k == "all" else ranks.le(k).fillna(False)
                        record = {
                            "panel": panel, "group_type": group_type, "group": group, "ranking_unit": unit,
                            "top_k": str(k), "observations": len(subset), "biorxngen_hits": int(hits.sum()),
                            "native_gem_hits": int(subset.native_gem.sum()),
                            "native_reaction_product_hits": int(subset.native_reaction_products.sum()),
                            "native_plus_biorxngen_hits": int((hits | subset.native_reaction_products).sum()),
                        }
                        records.append(record)
    ranks = candidates.candidate_rank
    budgets = pd.DataFrame([
        {"top_k": str(k), "candidate_rows": len(ranks) if k == "all" else int(ranks.le(k).sum())}
        for k in (*TOP_K, "all")
    ])
    return pd.DataFrame(records), budgets


def load_ranked(path):
    with ZipFile(path) as archive:
        check_archive(archive)
        rows = read_table(archive, "scored_candidates.parquet")
    return rows


def summarize(rows):
    assays = pd.read_csv(RESULTS / "per_analyte.tsv", sep="\t")
    ranked = assay_ranks(rows, assays)
    return ranked, *coverage(ranked, rows)


def compare_candidates(rows):
    with ZipFile(RESULTS / "generation_records.zip") as archive:
        names = [n for n in archive.namelist() if n.startswith("ec/") and "_products_" in n]
        fields = [c for c in GENERATION_FIELDS if c not in {"product_full_nonisomeric", "n_product_fragments"}]
        archived = pd.concat([read_table(archive, n)[fields] for n in names], ignore_index=True)
    for frame in (archived, rows):
        if frame.duplicated(IDENTITY).any():
            raise ValueError("Candidate rows lack unique generation keys")
    def ordered(frame):
        return frame[fields].sort_values(IDENTITY).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered(rows), ordered(archived))


def score(generation, output, model, config):
    tables = []
    for file in sorted((generation / "ec").glob("*/*_products_*.parquet")):
        frame = pd.read_parquet(file, columns=GENERATION_FIELDS)
        frame["label"] = file.parent.name
        frame["reranker_score"] = infer(frame, model, config)
        tables.append(frame)
    if not tables:
        raise FileNotFoundError("Run the benchmark generation command first")
    rows = pd.concat(tables, ignore_index=True)
    compare_candidates(rows)
    assign_ranks(rows)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / "scored_candidates.parquet", index=False)
    for name, table in zip(("ranked_analytes", "ranked_coverage", "ranking_budgets"), summarize(rows)):
        (output / f"{name}.tsv").write_bytes(table.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8"))
    print(f"Scored {len(rows):,} candidate records")


def verify(model, config):
    rows = load_ranked(RESULTS / "ranked_candidates.zip")
    compare_candidates(rows)
    for _, frame in rows.groupby("model_id", sort=False):
        np.testing.assert_allclose(infer(frame, model, config), frame.reranker_score, atol=1e-6, rtol=0)
    recalculated = assign_ranks(rows.copy())
    for field in ("candidate_rank", "reactant_context_rank"):
        pd.testing.assert_series_equal(recalculated[field], rows[field])
    for name, table in zip(("ranked_analytes", "ranked_coverage", "ranking_budgets"), summarize(rows)):
        with (RESULTS / f"{name}.tsv").open(encoding="utf-8") as file:
            if table.to_csv(sep="\t", index=False, lineterminator="\n") != file.read():
                raise ValueError(f"Archived summary differs for {name}")
    print(f"Verified frozen scores, ranks and coverage for {len(rows):,} candidate records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("score")
    run.add_argument("--generation", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    commands.add_parser("verify")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, config = load_model(CHECKPOINT)
    if args.command == "verify":
        verify(model, config)
    else:
        score(args.generation, args.output, model, config)


if __name__ == "__main__":
    main()
