from __future__ import annotations
import sys
import argparse
from pathlib import Path
import pandas as pd
from biorxngen.reranker import train_reranker

ROOT = Path(__file__).resolve().parents[2]


def train(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the candidate-product reranker from feature rows and a fixed split."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "data/raw_sample/reranker_feature_sample.parquet",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data/benchmark_results/internal_split_manifest.csv",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    rows = pd.read_parquet(args.candidates)
    split = pd.read_csv(args.split)[["reaction_unit_id", "split"]]
    rows = rows.merge(split, on="reaction_unit_id", how="inner", validate="many_to_one")
    train = rows.loc[rows["split"].eq("train")]
    validation = rows.loc[rows["split"].eq("valid")]
    result = train_reranker(
        train, validation, args.checkpoint, epochs=args.epochs, device=args.device
    )
    print(result)


from biorxngen.reranker import load_model, score_candidates


def score(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Score candidate products with the frozen BioRxnGen reranker."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "data/raw_sample/reranker_feature_sample.parquet",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "code/models/product_reranker.pt"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    rows = pd.read_parquet(args.candidates)
    model, config = load_model(args.checkpoint, device=args.device)
    rows["biorxngen_score"] = score_candidates(rows, model, config, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(args.output, index=False)
    print(f"Scored {len(rows)} candidate products: {args.output}")


COMMANDS = {"train": train, "score": score}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Train or apply the product reranker")
    parser.add_argument("command", choices=COMMANDS)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.parse_args(arguments)
    command = parser.parse_args(arguments[:1]).command
    COMMANDS[command](arguments[1:])


if __name__ == "__main__":
    main()
