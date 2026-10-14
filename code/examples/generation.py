from __future__ import annotations
import sys
import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

RULE = "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[O:3].[C:4]O"


def build_sample(root: Path) -> dict[str, Path]:
    root = root.resolve()
    rule_dir = root / "rules" / "reaction_labels"
    pair_dir = root / "worklist"
    rule_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame(
        [
            {
                "reaction_record_id": "synthetic_train_esterase",
                "mapped": "CC(=O)OC>>CC(=O)O.CO",
                "unmapped": "CC(=O)OC>>CC(=O)O.CO",
                "ec_num": "3.1.1.1",
                "ec1": "3",
                "ec2": "3.1",
                "ec3": "3.1.1",
                "ec4": "3.1.1.1",
                "rule": RULE,
                "rule_directional_id": "synthetic_esterase_rule",
                "rule_family_id": "synthetic_esterase_family",
                "rule_direction": "forward",
                "source": "synthetic",
                "steps": 1,
                "quality": 1.0,
                "n_reactants": 1,
                "n_products": 2,
                "n_bond_edits": 1,
                "n_changed_atoms": 2,
            }
        ]
    )
    labels.to_parquet(rule_dir / "part_00000.parquet", index=False)
    split = root / "split_assignments.parquet"
    pd.DataFrame(
        [{"reaction_record_id": "synthetic_train_esterase", "split": "train"}]
    ).to_parquet(split, index=False)
    pair_path = pair_dir / "pairs.parquet"
    pd.DataFrame(
        [
            {
                "candidate_pair_id": "synthetic_pair_1",
                "model_id": "synthetic_strain_model",
                "enzyme_key": "synthetic_enzyme_1",
                "enzyme_source_namespace": "internal_agora_gene_annotation_sequence",
                "substrate_smiles": "CC(=O)OC",
                "substrate_evidence_tier": "S2_agora_native_known_positive",
                "ec_number": "3.1.1.1",
            }
        ]
    ).to_parquet(pair_path, index=False)
    enzyme_path = pair_dir / "enzymes.parquet"
    pd.DataFrame(
        [
            {
                "enzyme_key": "synthetic_enzyme_1",
                "sequence_uid": "synthetic_sequence_1",
                "protein_sequence": "MKTLLVAAAGAA",
                "sequence_length": 12,
                "protein_sequence_source": "synthetic",
                "protein_acceptance_tier": "synthetic",
                "protein_confidence": 1.0,
            }
        ]
    ).to_parquet(enzyme_path, index=False)
    manifest = pair_dir / "worklist_manifest.csv"
    pd.DataFrame(
        [
            {
                "segment_id": "synthetic_segment",
                "source_namespace": "synthetic",
                "candidate_edges": 1,
                "pair_partitions": 1,
                "eligible_enzyme_rows": 1,
                "pair_partition_files": pair_path.relative_to(root).as_posix(),
                "enzyme_reference_file": enzyme_path.relative_to(root).as_posix(),
            }
        ]
    ).to_csv(manifest, index=False)
    cofactor_pool = root / "cofactor_pool.parquet"
    pd.DataFrame(
        [{"agora_base_metabolite_ids": "M_h2o", "canonical_smiles": "O"}]
    ).to_parquet(cofactor_pool, index=False)
    return {
        "artifact_root": root,
        "worklist_manifest": manifest,
        "rule_chunks": rule_dir.parent,
        "split_assignments": split,
        "cofactor_pool": cofactor_pool,
    }


import subprocess
import tempfile


def run(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the archived generator on synthetic inputs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Fresh output directory; defaults to a unique demo directory",
    )
    args = parser.parse_args(argv)
    if args.output_dir:
        root = args.output_dir.resolve()
        if root.exists():
            parser.error(f"Output directory already exists: {root}")
    else:
        demo_root = Path(__file__).resolve().parent / "demo_output"
        demo_root.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="full_generation_", dir=demo_root))
    paths = build_sample(root / "inputs")
    output = root / "result"
    generator = Path(__file__).resolve().parents[1] / "scripts/generation.py"
    subprocess.run(
        [
            sys.executable,
            str(generator),
            "run",
            "--artifact-root",
            str(paths["artifact_root"]),
            "--worklist-manifest",
            str(paths["worklist_manifest"]),
            "--rule-chunks",
            str(paths["rule_chunks"]),
            "--split-assignments",
            str(paths["split_assignments"]),
            "--cofactor-pool",
            str(paths["cofactor_pool"]),
            "--output",
            str(output),
            "--max-tasks",
            "1",
            "--max-pairs-per-task",
            "1",
            "--run",
        ],
        check=True,
    )
    product_path = (
        output
        / "partitions/synthetic_segment/pair_part_00000/chunk_00000/candidate_products.parquet"
    )
    with pq.ParquetFile(product_path) as source:
        products = source.read(use_threads=False).to_pandas(use_threads=False)
    acid_present = any(
        ("CC(=O)O" in str(side).split(".") for side in products["product_smiles"])
    )
    if products.empty or not acid_present:
        raise RuntimeError("The synthetic reference product was not generated")
    print(f"Synthetic product check passed: {len(products)} candidate rows")
    print(f"Product table: {product_path}")


COMMANDS = {"run": run}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Run the candidate-generation sample")
    parser.add_argument("command", choices=COMMANDS)
    if not arguments or arguments[0].startswith("--"):
        arguments.insert(0, "run")
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.parse_args(arguments)
    command = parser.parse_args(arguments[:1]).command
    COMMANDS[command](arguments[1:])


if __name__ == "__main__":
    main()
