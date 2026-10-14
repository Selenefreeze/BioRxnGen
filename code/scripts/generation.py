from __future__ import annotations
import sys
import argparse
import os
from pathlib import Path
import shutil
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1] / "reproduction/backend/scripts"
sys.path.insert(0, str(BACKEND))
from worklist_builder import build_pair_worklist


def link_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def worklist(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a strain-resolved enzyme-substrate worklist."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-internal-chunks", type=int, default=0)
    parser.add_argument("--max-pairs-per-segment", type=int, default=0)
    args = parser.parse_args(argv)
    root = args.data_root.resolve()
    output = args.output.resolve()
    inputs = {
        "substrate_universe_path": root
        / "reaction_inputs/substrate_universe/full_cohort_substrate_universe.parquet",
        "external_sequence_scope_path": root
        / "reaction_inputs/sequence_scope/external_sequences.parquet",
        "internal_sequence_chunks_dir": root
        / "reaction_inputs/sequence_scope/internal_sequence_links",
    }
    for path in inputs.values():
        if not path.exists():
            parser.error(f"Missing input: {path}")
    backend_output = output / ".backend"
    config = {
        "artifact_root": str(root),
        "output_dir": str(backend_output),
        **{key: str(path) for key, path in inputs.items()},
        "max_internal_chunks": args.max_internal_chunks,
        "max_pairs_per_segment": args.max_pairs_per_segment,
    }
    summary = build_pair_worklist(config)
    if summary["metrics"]["status"] != "completed":
        raise RuntimeError("Worklist construction did not complete")
    source_manifest = backend_output / "worklist_manifest.csv"
    frame = pd.read_csv(source_manifest)
    pair_root = backend_output / "pairs"
    enzyme_root = backend_output / "enzyme_references"
    for source in pair_root.glob("*.parquet"):
        link_file(source, output / "pairs" / source.name)
    for source in enzyme_root.glob("*.parquet"):
        link_file(source, output / "enzyme_references" / source.name)
    frame["pair_partition_files"] = frame["pair_partition_files"].map(
        lambda value: "|".join(
            ("pairs/" + Path(entry).name for entry in str(value).split("|"))
        )
    )
    frame["enzyme_reference_file"] = frame["enzyme_reference_file"].map(
        lambda value: "enzyme_references/" + Path(str(value)).name
    )
    frame = frame.drop(columns=["source_path"], errors="ignore")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "worklist_manifest.csv", index=False)
    print(f"Worklist: {output / 'worklist_manifest.csv'}")
    print(f"Pair partitions: {len(list((output / 'pairs').glob('*.parquet')))}")


import json
from pathlib import Path, PureWindowsPath
import re
import pyarrow.parquet as pq

SCRIPTS = Path(__file__).resolve().parents[1] / "reproduction/backend/scripts"
sys.path.insert(0, str(SCRIPTS))
from generation_batch import run_generation_collection


def resolve_worklist_path(value: str, artifact_root: Path) -> str:
    entries = []
    for entry in str(value).split("|"):
        if not entry or entry.lower() == "nan":
            raise ValueError("Worklist contains an empty input path")
        path = Path(entry)
        if PureWindowsPath(entry).is_absolute() and (not path.is_absolute()):
            raise ValueError(f"Convert source-machine paths before running: {entry}")
        if not path.is_absolute() and ".." in path.parts:
            raise ValueError(
                f"Worklist path must stay under the artifact root: {entry}"
            )
        entries.append(str(path if path.is_absolute() else artifact_root / path))
    return "|".join(entries)


def prepare_rule_chunks(source: Path, output: Path) -> Path:
    if (source / "reaction_labels").is_dir():
        return source
    partitions = sorted(source.glob("*.parquet"))
    if not partitions:
        raise ValueError(f"No directional rule partitions under {source}")
    target = output / ".backend_inputs/reaction_labels"
    target.mkdir(parents=True, exist_ok=True)
    for partition in partitions:
        destination = target / partition.name
        if destination.exists():
            continue
        try:
            os.link(partition, destination)
        except OSError:
            shutil.copy2(partition, destination)
    return target.parent


PUBLIC_OUTPUTS = {
    "candidate_products.parquet": (
        "candidate_products.parquet",
        {
            "candidate_pair_id": "pair_id",
            "model_id": "model_id",
            "ec_number": "ec_number",
            "reaction_id": "reaction_id",
            "protein_ref_db": "protein_reference_database",
            "protein_accession": "protein_accession",
            "enzyme_source_namespace": "enzyme_source_namespace",
            "substrate_uid": "substrate_id",
            "substrate_smiles": "substrate_smiles",
            "generated_product_side_smiles": "product_smiles",
            "candidate_reaction_smiles": "reaction_smiles",
            "candidate_rule_directional_id": "rule_id",
            "rule_rank": "rule_rank",
            "product_rank_within_rule": "product_rank_within_rule",
        },
    ),
    "generation_attempts.parquet": (
        "generation_attempts.parquet",
        {
            "candidate_pair_id": "pair_id",
            "model_id": "model_id",
            "ec_number": "ec_number",
            "enzyme_source_namespace": "enzyme_source_namespace",
            "substrate_smiles": "substrate_smiles",
            "n_rule_candidates": "rule_candidates",
            "n_product_rows": "product_rows",
            "generation_status": "status",
        },
    ),
    "selected_pairs.parquet": (
        "selected_pairs.parquet",
        {
            "candidate_pair_id": "pair_id",
            "model_id": "model_id",
            "ec_number": "ec_number",
            "enzyme_source_namespace": "enzyme_source_namespace",
            "protein_accession": "protein_accession",
            "substrate_smiles": "substrate_smiles",
        },
    ),
}
PAIR_PREFIX = re.compile("^\\d{2,3}[a-z]\\d*:", re.IGNORECASE)
HISTORICAL_LABEL = re.compile(
    "(?:(?i:phase)\\d{2,3}[a-z]?(?=[_:-]|\\b)|(?<![A-Za-z0-9])\\d{2,3}[a-z]\\d*(?=$|[_:]))"
)
BIOLOGICAL_COLUMNS = {"model_id", "protein_accession", "substrate_id", "reaction_id"}


def public_table(source: Path, columns: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_parquet(source)
    required = {"candidate_pair_id", "model_id", "ec_number"}
    required.update(
        set(columns).intersection(
            {"generated_product_side_smiles", "generation_status"}
        )
    )
    if "reaction_id" in columns:
        required.add("reaction_id")
    missing = required.difference(frame.columns)
    if missing and (not frame.empty):
        raise ValueError(
            f"Missing required generation output columns in {source}: {sorted(missing)}"
        )
    for column in set(columns).difference(frame.columns):
        frame[column] = None
    frame = frame[list(columns)].rename(columns=columns)
    if frame["pair_id"].isna().any():
        raise ValueError(f"Missing pair identifier in {source}")
    frame["pair_id"] = (
        frame["pair_id"].astype(str).map(lambda value: PAIR_PREFIX.sub("pair:", value))
    )
    if "reaction_id" in columns:
        frame["row_index"] = range(len(frame))
    for column in frame.columns:
        if column in BIOLOGICAL_COLUMNS:
            continue
        if not (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
        ):
            continue
        if frame[column].dropna().astype(str).str.contains(HISTORICAL_LABEL).any():
            raise ValueError(
                f"Historical run label remains in public output column {column}"
            )
    return frame


def expose_outputs(backend_output: Path, public_output: Path) -> None:
    for source_name, (public_name, columns) in PUBLIC_OUTPUTS.items():
        for source in backend_output.rglob(source_name):
            relative = source.parent.relative_to(backend_output)
            target = public_output / "partitions" / relative / public_name
            target.parent.mkdir(parents=True, exist_ok=True)
            projected = public_table(source, columns)
            if target.exists():
                if not pd.read_parquet(target).equals(projected):
                    raise FileExistsError(f"Existing public output differs: {target}")
                continue
            projected.to_parquet(target, index=False)


def run(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the archived full-cohort generator on a portable data layout."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--worklist-manifest", type=Path, required=True)
    parser.add_argument("--rule-chunks", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument("--cofactor-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-pairs-per-task", type=int, default=0)
    parser.add_argument(
        "--profile", choices=["standard", "regenerated"], default="standard"
    )
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        help="Candidate-product manifest selecting regenerated batches",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute generation; otherwise only produce a work plan",
    )
    args = parser.parse_args(argv)
    if args.profile == "regenerated" and (not args.batch_manifest):
        parser.error(
            "Regeneration requires --batch-manifest for the recorded output-limited subset"
        )
    selected_batches = None
    if args.batch_manifest:
        batches = pd.read_csv(args.batch_manifest, sep="\t")
        if "scope" in batches:
            batches = batches.loc[batches["scope"].eq(args.profile)]
        selected_batches = {}
        for batch in batches["batch_id"]:
            match = re.fullmatch("(.+)__part(\\d+)__batch(\\d+)", str(batch))
            if match is None:
                parser.error(f"Invalid batch ID: {batch}")
            segment, part, chunk = match.groups()
            selected_batches.setdefault((segment, int(part)), set()).add(int(chunk))
        if not selected_batches:
            parser.error("Batch selection is empty")
    for path in (
        args.artifact_root,
        args.worklist_manifest,
        args.rule_chunks,
        args.split_assignments,
        args.cofactor_pool,
    ):
        if not path.exists():
            parser.error(f"Missing input: {path}")
    labels = (
        args.rule_chunks / "reaction_labels"
        if (args.rule_chunks / "reaction_labels").is_dir()
        else args.rule_chunks
    )
    if not labels.is_dir() or not any(
        (
            child.suffix.lower() in {".parquet", ".csv", ".tsv"}
            for child in labels.iterdir()
        )
    ):
        parser.error(f"No reaction-label partitions under {labels}")
    manifest = pd.read_csv(args.worklist_manifest)
    required = {"pair_partition_files", "enzyme_reference_file"}
    if not required.issubset(manifest.columns) or manifest.empty:
        parser.error(
            f"Worklist manifest must contain rows and columns: {sorted(required)}"
        )
    artifact_root = args.artifact_root.resolve()
    for column in required:
        try:
            manifest[column] = manifest[column].map(
                lambda value: resolve_worklist_path(value, artifact_root)
            )
        except ValueError as exc:
            parser.error(str(exc))
    for row in manifest.itertuples(index=False):
        for column in required:
            entries = str(getattr(row, column, "")).split("|")
            for entry in entries:
                if not entry or entry.lower() == "nan" or (not Path(entry).is_file()):
                    parser.error(f"Missing {column} input: {entry}")
    cofactor_columns = set(pq.ParquetFile(args.cofactor_pool).schema.names)
    expected_cofactor_columns = {"substrate_base_metabolite_id", "substrate_smiles"}
    if not expected_cofactor_columns.issubset(cofactor_columns):
        print(
            "Cofactor table lacks the original loader's optional lookup columns; the frozen generator will use its four built-in basic cofactors.",
            file=sys.stderr,
        )
    if args.workers < 1 or args.max_tasks < 0 or args.max_pairs_per_task < 0:
        parser.error("Workers must be positive and sampling limits non-negative")
    args.output.mkdir(parents=True, exist_ok=True)
    backend_output = args.output / ".backend"
    backend_output.mkdir(parents=True, exist_ok=True)
    try:
        rule_chunks = prepare_rule_chunks(args.rule_chunks, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    resolved_manifest = backend_output / "resolved_worklist_manifest.csv"
    manifest.to_csv(resolved_manifest, index=False)
    config = {
        "repo_root": str(Path(__file__).resolve().parents[1] / "reproduction"),
        "artifact_root": str(args.artifact_root.resolve()),
        "output_dir": str(backend_output.resolve()),
        "worklist_manifest_path": str(resolved_manifest),
        "rule_chunks_dir": str(rule_chunks.resolve()),
        "split_assignments_path": str(args.split_assignments.resolve()),
        "annotated_substrate_pool_path": str(args.cofactor_pool.resolve()),
        "parallel_workers": args.workers,
        "multiprocessing_start_method": "spawn" if os.name == "nt" else "fork",
        "max_partition_tasks": args.max_tasks,
        "max_pairs_per_partition": args.max_pairs_per_task,
        "chunk_rows": 5000,
        "rule_topk_per_ec": 40,
        "max_contexts_per_candidate": 3,
        "max_products_per_rule": 12,
        "max_reactant_permutations": 128,
        "max_product_rows": 100000 if args.profile == "standard" else 0,
        "selected_batches": selected_batches,
        "run_esp_scoring": False,
        "planner_only": not args.run,
    }
    (args.output / "generation_settings.json").write_text(
        json.dumps(
            {
                key: config[key]
                for key in (
                    "chunk_rows",
                    "rule_topk_per_ec",
                    "max_contexts_per_candidate",
                    "max_products_per_rule",
                    "max_reactant_permutations",
                    "max_product_rows",
                    "run_esp_scoring",
                )
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = run_generation_collection(config)
    if args.run and summary["status"] == "completed":
        expose_outputs(backend_output, args.output)
    (args.output / "run_summary.json").write_text(
        json.dumps(
            {
                "status": summary["status"],
                "task_count": int(summary["metrics"]["task_count"]),
                "executed": bool(args.run),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Status: {summary['status']}")
    print(f"Output: {args.output.resolve()}")
    if args.run and summary["status"] != "completed":
        raise RuntimeError(f"Full generation ended with status {summary['status']}")


def artifact_relative(entry: str, artifact_root: Path) -> str:
    normalized = entry.replace("\\", "/")
    marker = "/artifacts/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
    elif Path(entry).is_absolute():
        try:
            relative = (
                Path(entry).resolve().relative_to(artifact_root.resolve()).as_posix()
            )
        except ValueError as exc:
            raise ValueError(f"Path is outside the artifact root: {entry}") from exc
    elif re.match("^[A-Za-z]:/", normalized):
        root = str(artifact_root.resolve()).replace("\\", "/").rstrip("/")
        if not normalized.lower().startswith(root.lower() + "/"):
            raise ValueError(f"Path is outside the artifact root: {entry}")
        relative = normalized[len(root) + 1 :]
    else:
        relative = normalized
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe artifact-relative path: {entry}")
    return path.as_posix()


def relocate(value: str, artifact_root: Path, absolute: bool = False) -> str:
    text = str(value)
    if not text or text.lower() == "nan":
        return ""
    entries = [artifact_relative(entry, artifact_root) for entry in text.split("|")]
    if absolute:
        entries = [str(artifact_root.resolve() / Path(entry)) for entry in entries]
    return "|".join(entries)


def relocate_files(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Make the archived full-cohort worklist manifest portable."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Resolve paths against the specified local data root",
    )
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.source)
    for column in ("pair_partition_files", "enzyme_reference_file", "source_path"):
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: relocate(
                    value, args.artifact_root.resolve(), absolute=args.absolute
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} worklist segments: {args.output}")


COMMANDS = {"worklist": worklist, "run": run, "relocate": relocate_files}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Prepare inputs and generate candidate reactions"
    )
    parser.add_argument("command", choices=COMMANDS)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.parse_args(arguments)
    command = parser.parse_args(arguments[:1]).command
    COMMANDS[command](arguments[1:])


if __name__ == "__main__":
    main()
