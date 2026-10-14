from __future__ import annotations
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
from biorxngen.compatibility import score_compatibility, verify_compatibility_scores


def read(path):
    return (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path, sep="\t", dtype={"substrate_ecfp_bitstring": str})
    )


def score(argv=None):
    root = Path(__file__).resolve().parents[2]
    sample = root / "data/raw_sample/compatibility_inference"
    parser = argparse.ArgumentParser(
        description="Score protein-substrate compatibility with the frozen ESP classifier"
    )
    parser.add_argument("--pairs", type=Path, default=sample / "pair_inputs.tsv")
    parser.add_argument(
        "--embedding-map", type=Path, default=sample / "protein_embedding_map.tsv"
    )
    parser.add_argument(
        "--embeddings", type=Path, default=sample / "protein_embeddings.npy"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "code/models/esp_compatibility_classifier.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--reference-scores",
        type=Path,
        help="Archived scores joined by pair_id for --check",
    )
    parser.add_argument(
        "--verification-report",
        type=Path,
        help="Write comparison counts and numerical errors as JSON",
    )
    args = parser.parse_args(argv)
    if (args.reference_scores or args.verification_report) and (not args.check):
        parser.error("Reference scores and verification reports require --check")
    inputs = [
        args.pairs,
        args.embedding_map,
        args.embeddings,
        args.checkpoint,
        args.reference_scores,
    ]
    protected = {path.resolve() for path in inputs if path is not None}
    if args.output.resolve() in protected or (
        args.verification_report and args.verification_report.resolve() in protected
    ):
        parser.error("Outputs must not overwrite input or checkpoint files")
    if (
        args.verification_report
        and args.verification_report.resolve() == args.output.resolve()
    ):
        parser.error("Score output and verification report must have different paths")
    pairs = read(args.pairs)
    pairs["compatibility_score"] = score_compatibility(
        pairs,
        read(args.embedding_map),
        args.embeddings,
        args.checkpoint,
        args.batch_size,
    )
    if args.check:
        if args.reference_scores:
            reference = read(args.reference_scores)
        elif "reference_score" in pairs:
            reference = pairs[["pair_id", "reference_score"]].rename(
                columns={"reference_score": "compatibility_score"}
            )
        else:
            parser.error("--check requires a reference_score column")
        report = verify_compatibility_scores(pairs, reference)
        if args.verification_report:
            args.verification_report.parent.mkdir(parents=True, exist_ok=True)
            args.verification_report.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        pairs.to_parquet(args.output, index=False)
    else:
        pairs.to_csv(args.output, sep="\t", index=False)
    print(
        f"Scored {len(pairs)} protein-substrate pairs"
        + ("; reference check passed" if args.check else "")
    )


from sklearn.metrics import average_precision_score, roc_auc_score

REQUIRED = {"pair_id", "label", "split", "decoy_class", "compatibility_score"}


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    if (
        frame.empty
        or frame["pair_id"].isna().any()
        or frame["pair_id"].duplicated().any()
    ):
        raise ValueError("Score table must contain unique, non-null pair identifiers")
    label = pd.to_numeric(frame["label"], errors="raise")
    score = pd.to_numeric(frame["compatibility_score"], errors="raise")
    if (
        not label.isin([0, 1]).all()
        or score.isna().any()
        or (not score.between(0, 1).all())
    ):
        raise ValueError(
            "Labels must be binary and scores must be finite probabilities"
        )

    def metrics(rows: pd.DataFrame) -> dict[str, float | int]:
        y = rows["label"].astype(int)
        s = rows["compatibility_score"].astype(float)
        if y.nunique() != 2:
            raise ValueError(
                "Each evaluated subset must contain positive and negative pairs"
            )
        return {
            "pairs": int(len(rows)),
            "known_pairs": int(y.sum()),
            "auroc": float(roc_auc_score(y, s)),
            "average_precision": float(average_precision_score(y, s)),
        }

    return {
        "all_pairs": metrics(frame),
        "split_metrics": {
            name: metrics(rows) for name, rows in frame.groupby("split", sort=True)
        },
        "decoy_counts": {
            name: int(count)
            for name, count in frame["decoy_class"].value_counts().items()
        },
    }


def metrics(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit saved protein-substrate compatibility scores."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help="Parquet or tab-separated score table",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Reference JSON for exact count and metric checks",
    )
    parser.add_argument("--output", type=Path, help="Write the audit summary as JSON")
    args = parser.parse_args(argv)
    if not args.scores.is_file():
        parser.error(f"Missing score table: {args.scores}")
    frame = (
        pd.read_parquet(args.scores)
        if args.scores.suffix.lower() == ".parquet"
        else pd.read_csv(args.scores, sep="\t")
    )
    result = summarize(frame)
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        for scope in ("all_pairs", "test"):
            observed = (
                result["all_pairs"]
                if scope == "all_pairs"
                else result["split_metrics"]["test"]
            )
            expected = reference[scope]
            for key in ("pairs", "known_pairs", "auroc", "average_precision"):
                tolerance = 1e-10 if isinstance(expected[key], float) else 0
                if abs(observed[key] - expected[key]) > tolerance:
                    raise ValueError(
                        f"Reference mismatch for {scope}.{key}: {observed[key]} != {expected[key]}"
                    )
        print("Saved-score metrics match the reference summary.")
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


COMMANDS = {"score": score, "metrics": metrics}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Score protein-substrate compatibility and summarize metrics"
    )
    parser.add_argument("command", choices=COMMANDS)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.parse_args(arguments)
    command = parser.parse_args(arguments[:1]).command
    COMMANDS[command](arguments[1:])


if __name__ == "__main__":
    main()
