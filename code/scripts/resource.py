from __future__ import annotations
import sys
import argparse
import json
from pathlib import Path
from biorxngen.graph_audit import audit_map


def classify(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Recompute standard-map signature classes and full-union coverage statistics"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report = audit_map(args.data_root, args.output, args.check)
    (args.output / "graph_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


from collections import Counter, defaultdict
import hashlib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def summarize(root: Path) -> tuple[pd.DataFrame, int]:
    manifest = pd.read_csv(root / "metadata/candidate_product_manifest.tsv", sep="\t")
    manifest = manifest.loc[manifest.scope.eq("regenerated")]
    columns = ["ec_number", "reaction_smiles"]
    if manifest.empty or manifest.batch_id.duplicated().any():
        raise ValueError("Regenerated batch selection is empty or duplicated")
    counts = Counter()
    signatures = defaultdict(set)
    for index, row in enumerate(manifest.itertuples(index=False), 1):
        path = (root / row.file).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("A partition is outside the data root")
        partition = pq.ParquetFile(path)
        if partition.metadata.num_rows != int(row.rows):
            raise ValueError(f"Partition count differs: {row.file}")
        for batch in partition.iter_batches(batch_size=131072, columns=columns):
            grouped = (
                pa.Table.from_batches([batch])
                .group_by(columns)
                .aggregate([("ec_number", "count")])
            )
            for record in grouped.to_pylist():
                ec = str(record["ec_number"]).strip()
                ec_class = "EC" + ec.split(".")[0]
                counts[ec_class] += int(record["ec_number_count"])
                basis = ec + "||" + str(record["reaction_smiles"]).strip()
                signature = (
                    "signature:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:24]
                )
                signatures[ec_class].add(signature)
        if index % 50 == 0:
            print(f"Checked {index}/{len(manifest)} batches", flush=True)
    summary = pd.DataFrame(
        [
            {
                "ec_class": ec,
                "rows": counts[ec],
                "unique_reaction_signatures": len(signatures[ec]),
            }
            for ec in sorted(counts)
        ]
    )
    return (summary, len(manifest))


def regenerated(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Recompute EC-class counts from regenerated candidate products."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    summary, batches = summarize(args.data_root)
    total = int(summary.rows.sum())
    if batches != 460 or total != 220051473:
        raise ValueError("Regenerated subset count differs from the manuscript")
    reference_path = (
        args.reference
        or Path(__file__).resolve().parents[2]
        / "data/map_results/regenerated_ec_class_summary.tsv"
    )
    reference = pd.read_csv(reference_path, sep="\t")
    for record in summary.itertuples(index=False):
        expected = reference.loc[reference.ec_class.astype(str).eq(record.ec_class)]
        if len(expected) != 1:
            raise ValueError(
                f"Missing or duplicated reference class: {record.ec_class}"
            )
        for field in ("rows", "unique_reaction_signatures"):
            if int(getattr(record, field)) != int(expected.iloc[0][field]):
                raise ValueError(f"{record.ec_class} mismatch: {field}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, sep="\t", index=False)
    print(summary.to_string(index=False))
    print(json.dumps({"batches": batches, "candidate_product_rows": total}))


from collections import Counter
from biorxngen.graph_audit import product_inputs


def coverage(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Recompute strain-level native-substrate candidate placement counts"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    total, nonself, native = (Counter(), Counter(), Counter())
    for index, (batch, products) in enumerate(product_inputs(args.data_root), 1):
        for model, group in products.groupby("model_id"):
            total[model] += len(group)
            nonself[model] += int(group.nonself_flag.sum())
            native[model] += int(group.native_nonself_flag.sum())
        if index % 200 == 0:
            print(f"Checked {index} standard batches", flush=True)
    frame = pd.DataFrame(
        [
            {
                "model_id": model,
                "candidate_rows": total[model],
                "nonself_rows": nonself[model],
                "native_substrate_nonself_rows": native[model],
            }
            for model in sorted(total)
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "model_candidate_coverage.tsv", sep="\t", index=False)
    positive = frame.loc[
        frame.native_substrate_nonself_rows.gt(0), "native_substrate_nonself_rows"
    ]
    bins = [
        ("0", 0, 1),
        ("1-99", 1, 100),
        ("100-999", 100, 1000),
        ("1k-4.9k", 1000, 5000),
        ("5k-9.9k", 5000, 10000),
        (">=10k", 10000, float("inf")),
    ]
    summary = [
        {
            "candidate_row_bin": label,
            "models": int(
                (
                    frame.native_substrate_nonself_rows.ge(lower)
                    & frame.native_substrate_nonself_rows.lt(upper)
                ).sum()
            ),
        }
        for label, lower, upper in bins
    ]
    pd.DataFrame(summary).to_csv(
        args.output / "model_coverage_bins.tsv", sep="\t", index=False
    )
    report = {
        "models_with_standard_products": len(frame),
        "models_with_native_substrate_candidates": len(positive),
        "native_substrate_nonself_rows": int(positive.sum()),
        "median_positive_model_rows": float(positive.median()),
        "median_assessed_model_rows": float(
            frame.native_substrate_nonself_rows.median()
        ),
        "models_at_least_10000_rows": int(positive.ge(10000).sum()),
        "zero_native_substrate_models": int(
            frame.native_substrate_nonself_rows.eq(0).sum()
        ),
    }
    (args.output / "model_coverage.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


from rdkit import RDLogger
from biorxngen.database_reference import (
    load_agora_overlap_reference,
    validate_signature_table,
)


def database(argv=None):
    parser = argparse.ArgumentParser(
        description="Match candidate reaction signatures against the retained Rhea snapshot"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root, output = (args.data_root.resolve(), args.output.resolve())
    if output == root or root in output.parents:
        parser.error("Write outputs outside the immutable data deposit")
    RDLogger.DisableLog("rdApp.*")
    signatures = pd.read_parquet(root / "graph_analysis/signature_validation.parquet")
    inputs = signatures[
        [
            "signature_id",
            "ec_number",
            "reaction_smiles",
            "noncurrency_product_smiles",
            "product_smiles",
        ]
    ]
    matches = validate_signature_table(
        inputs,
        root / "reference_databases/rhea_snapshot",
        load_agora_overlap_reference(root),
    )
    if args.check:
        pd.testing.assert_frame_equal(
            matches.fillna(""),
            signatures[matches.columns].fillna(""),
            check_dtype=False,
        )
    output.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(output / "signature_validation.parquet", index=False)
    report = {
        "signatures": len(matches),
        "validation_tiers": matches.validation_tier.value_counts().to_dict(),
        "reference_check": "passed" if args.check else "not_requested",
    }
    (output / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


COMMANDS = {
    "classify": classify,
    "regenerated": regenerated,
    "coverage": coverage,
    "database": database,
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Summarize candidate reactions and database correspondence"
    )
    parser.add_argument("command", choices=COMMANDS)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.parse_args(arguments)
    command = parser.parse_args(arguments[:1]).command
    COMMANDS[command](arguments[1:])


if __name__ == "__main__":
    main()
