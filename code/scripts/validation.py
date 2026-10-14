from __future__ import annotations
import sys
import argparse
import hashlib
from pathlib import Path
import pandas as pd


def verify(root: Path, manifest: Path) -> int:
    root = root.resolve()
    records = pd.read_csv(manifest, sep="\t")
    if (
        not {"file", "bytes", "sha256"}.issubset(records.columns)
        or records["file"].duplicated().any()
    ):
        raise ValueError(
            "Manifest must contain unique file paths, byte sizes and SHA256 values"
        )
    for row in records.itertuples(index=False):
        path = (root / row.file).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or out-of-root file: {row.file}")
        if path.stat().st_size != int(row.bytes):
            raise ValueError(f"Size mismatch: {row.file}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != row.sha256:
            raise ValueError(f"Checksum mismatch: {row.file}")
    return len(records)


def verify_files(argv=None) -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Verify package files against a relative-path SHA256 manifest."
    )
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    manifest = args.manifest or args.root / "code" / "validation" / "checksums.tsv"
    print(f"Verified {verify(args.root, manifest)} files.")


import json
import pyarrow.parquet as pq


def audit_inputs(root: Path) -> dict:
    manifest = pd.read_csv(root / "worklist_manifest.csv")
    pair_count = partitions = 0
    models = set()
    seen_pairs = set()
    for record in manifest.itertuples(index=False):
        reference = pd.read_parquet(root / record.enzyme_reference_file)
        required = {"enzyme_key", "model_id", "protein_accession", "protein_sequence"}
        if not required.issubset(reference):
            raise ValueError(f"Incomplete enzyme-reference schema: {record.segment_id}")
        keys = set(reference.enzyme_key.dropna())
        for relative in record.pair_partition_files.split("|"):
            frame = pd.read_parquet(root / relative)
            required = {
                "candidate_pair_id",
                "model_id",
                "enzyme_key",
                "protein_accession",
                "substrate_smiles",
                "substrate_evidence_tier",
            }
            if not required.issubset(frame):
                raise ValueError(f"Incomplete pair schema: {relative}")
            if frame[list(required)].isna().any().any():
                raise ValueError(f"Missing pair annotations: {relative}")
            pair_ids = set(frame.candidate_pair_id)
            if len(pair_ids) != len(frame) or seen_pairs.intersection(pair_ids):
                raise ValueError(f"Duplicate pair identifiers: {relative}")
            if not set(frame.enzyme_key).issubset(keys):
                raise ValueError(f"Missing enzyme-reference links: {relative}")
            seen_pairs.update(pair_ids)
            models.update(frame.model_id)
            pair_count += len(frame)
            partitions += 1
    rules = sorted((root / "reaction_inputs/directional_rules").glob("*.parquet"))
    rule_rows = sum((pq.ParquetFile(path).metadata.num_rows for path in rules))
    return {
        "status": "passed",
        "pair_partitions": partitions,
        "pair_rows": pair_count,
        "pair_models": len(models),
        "directional_rule_partitions": len(rules),
        "directional_reaction_label_rows": rule_rows,
        "pair_enzyme_reference_links": "complete",
    }


def inputs(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit complete generation input schemas and pair-to-enzyme linkage"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_inputs(args.data_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


import csv

FIELDS = [
    "pair_id",
    "reaction_id",
    "model_id",
    "ec_number",
    "protein_reference_database",
    "protein_accession",
    "enzyme_source_namespace",
    "substrate_id",
    "substrate_smiles",
    "product_smiles",
    "reaction_smiles",
    "rule_id",
    "rule_rank",
    "product_rank_within_rule",
    "row_index",
]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit(root: Path, manifest: Path, check_hashes: bool = False) -> dict[str, int]:
    rows = list(
        csv.DictReader(manifest.open(encoding="utf-8", newline=""), delimiter="\t")
    )
    if not rows:
        raise ValueError("Candidate-product manifest is empty")
    seen: set[tuple[str, str]] = set()
    counts = {"standard": 0, "regenerated": 0}
    batches = {"standard": {}, "regenerated": {}}
    for row in rows:
        scope, batch_id = (row["scope"], row["batch_id"])
        if scope not in batches or (scope, batch_id) in seen:
            raise ValueError(f"Invalid or repeated batch: {scope}/{batch_id}")
        seen.add((scope, batch_id))
        relative = Path(row["file"])
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or nonlocal partition: {row['file']}")
        info = pq.ParquetFile(path)
        expected = int(row["rows"])
        if info.schema.names != FIELDS or info.metadata.num_rows != expected:
            raise ValueError(f"Schema or row count differs: {row['file']}")
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"File size differs: {row['file']}")
        if check_hashes and sha256(path) != row["sha256"]:
            raise ValueError(f"Checksum differs: {row['file']}")
        counts[scope] += expected
        batches[scope][batch_id] = expected
    missing = batches["regenerated"].keys() - batches["standard"].keys()
    if missing:
        raise ValueError(
            f"Regenerated batches absent from standard output: {len(missing)}"
        )
    return {
        "standard_batches": len(batches["standard"]),
        "standard_product_rows": counts["standard"],
        "regenerated_batches": len(batches["regenerated"]),
        "regenerated_product_rows": counts["regenerated"],
        "replaced_standard_rows": sum(
            (batches["standard"][key] for key in batches["regenerated"])
        ),
    }


def products(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit deposited candidate-product partitions"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--check-hashes", action="store_true")
    args = parser.parse_args(argv)
    manifest = (
        args.manifest or args.data_root / "metadata/candidate_product_manifest.tsv"
    )
    result = audit(args.data_root, manifest, args.check_hashes)
    if args.reference:
        expected = json.loads(args.reference.read_text(encoding="utf-8"))
        if result != expected:
            raise ValueError(f"Reference count mismatch: {result} != {expected}")
    print(json.dumps(result, indent=2))


COMMANDS = {"files": verify_files, "inputs": inputs, "products": products}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Validate repository files and generation data"
    )
    parser.add_argument("command", choices=COMMANDS)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.parse_args(arguments)
    command = parser.parse_args(arguments[:1]).command
    COMMANDS[command](arguments[1:])


if __name__ == "__main__":
    main()
