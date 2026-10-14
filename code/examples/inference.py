from __future__ import annotations
import sys
import argparse
from pathlib import Path
import subprocess


def external(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Check frozen external-model inference on two archived reaction units"
    )
    parser.add_argument("--output", type=Path, default=root / "output/external_sample")
    args = parser.parse_args(argv)
    subprocess.run(
        [
            sys.executable,
            str(root / "code/scripts/benchmark.py"),
            "external",
            "--data-root",
            str(root / "data/raw_sample/external_workflow"),
            "--checkpoint",
            str(root / "code/models/external_product_reranker.pt"),
            "--operation",
            "inference",
            "--output",
            str(args.output),
            "--check",
        ],
        check=True,
    )


def esp(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Check frozen ESP inference on a real compatibility-cohort sample"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "output/compatibility_cohort"
    )
    args = parser.parse_args(argv)
    sample = root / "data/raw_sample/compatibility_cohort"
    subprocess.run(
        [
            sys.executable,
            str(root / "code/scripts/compatibility.py"),
            "score",
            "--pairs",
            str(sample / "pair_inputs.tsv"),
            "--embedding-map",
            str(sample / "protein_embedding_map.tsv"),
            "--embeddings",
            str(sample / "protein_embeddings.npy"),
            "--reference-scores",
            str(sample / "reference_scores.tsv"),
            "--output",
            str(args.output / "scores.tsv"),
            "--verification-report",
            str(args.output / "verification.json"),
            "--check",
        ],
        check=True,
    )


import pandas as pd
from biorxngen.graph_audit import (
    classify_signatures,
    prepare_products,
    profile_products,
)


def graph(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Check graph-context sample")
    parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2] / "data/raw_sample"
    products = pd.read_parquet(root / "graph_candidate_products.parquet")
    pairs = pd.read_parquet(root / "graph_pair_contexts.parquet")
    validation = pd.read_parquet(root / "graph_signature_validation.parquet")
    result = classify_signatures(
        profile_products(prepare_products(products, pairs)), validation
    )
    reference = pd.read_parquet(root / "graph_sample_reference.parquet")
    pd.testing.assert_frame_equal(result, reference)
    print(
        f"Checked {len(products)} archived product rows and {len(result)} reaction signatures"
    )
    print(result.groupby("candidate_class").product_rows.sum().to_string())


COMMANDS = {"external": external, "esp": esp, "graph": graph}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run inference and graph-analysis samples"
    )
    parser.add_argument("command", choices=COMMANDS)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.parse_args(arguments)
    command = parser.parse_args(arguments[:1]).command
    COMMANDS[command](arguments[1:])


if __name__ == "__main__":
    main()
