"""Frozen ESP classifier inference using cached protein representations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biorxngen.production_rules import rule_modules


def verify_compatibility_scores(observed: pd.DataFrame, reference: pd.DataFrame,
                                tolerance: float = 1e-6) -> dict:
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Score tolerance must be finite and non-negative")
    for frame in (observed, reference):
        if not {"pair_id", "compatibility_score"}.issubset(frame) or frame.empty:
            raise ValueError("Score comparisons require non-empty identifier and score columns")
        if frame.pair_id.isna().any() or frame.pair_id.duplicated().any():
            raise ValueError("Score comparisons require unique, non-null pair identifiers")
    if set(observed.pair_id) != set(reference.pair_id):
        raise ValueError("Observed and reference pair identifiers must agree exactly")
    comparison = observed[["pair_id", "compatibility_score"]].merge(
        reference[["pair_id", "compatibility_score"]], on="pair_id", validate="one_to_one",
        suffixes=("_observed", "_reference"))
    values = comparison[["compatibility_score_observed", "compatibility_score_reference"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Compatibility scores must be finite probabilities")
    errors = np.abs(values[:, 0] - values[:, 1])
    report = {"pairs_checked": len(comparison), "maximum_score_difference": float(errors.max()),
              "mean_score_difference": float(errors.mean()), "absolute_tolerance": tolerance,
              "scores_outside_tolerance": int((errors > tolerance).sum())}
    if report["scores_outside_tolerance"]:
        raise ValueError(f"Compatibility score comparison failed: {report}")
    report["reference_check"] = "passed"
    return report


def score_compatibility(pairs: pd.DataFrame, mapping: pd.DataFrame, embeddings: Path,
                        checkpoint: Path, batch_size: int = 4096) -> np.ndarray:
    import xgboost as xgb

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not {"pair_id", "protein_ref"}.issubset(pairs) or pairs.pair_id.isna().any() or pairs.pair_id.duplicated().any() or pairs.protein_ref.isna().any():
        raise ValueError("Pairs require unique identifiers and protein references")
    if not {"protein_ref", "embedding_index"}.issubset(mapping) or mapping.protein_ref.isna().any() or mapping.protein_ref.duplicated().any() or mapping.embedding_index.isna().any():
        raise ValueError("Protein references require a unique embedding-row mapping")
    matrix = np.load(embeddings, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[1] != 1280:
        raise ValueError("Expected a two-dimensional, 1280-column protein embedding array")
    index = pd.to_numeric(mapping.embedding_index, errors="raise")
    if not (index == index.astype(int)).all() or not index.between(0, len(matrix) - 1).all():
        raise ValueError("Embedding indices must be valid integer row positions")
    selected = pairs.protein_ref.map(dict(zip(mapping.protein_ref, index.astype(int))))
    if selected.isna().any():
        raise ValueError("Some input proteins have no embedding-row mapping")
    booster = xgb.Booster()
    booster.load_model(checkpoint)
    if booster.num_features() != 2304:
        raise ValueError("The frozen ESP classifier expects 1024 substrate and 1280 protein features")
    rule_modules()
    from generator_preparation import _ecfp_bitstring_for_smiles

    scores = []
    names = [f"ECFP_{i}" for i in range(1024)] + [f"ESM1b_ts_{i}" for i in range(1280)]
    for start in range(0, len(pairs), batch_size):
        batch = pairs.iloc[start:start + batch_size]
        fingerprints = []
        for row in batch.to_dict("records"):
            bitstring = row.get("substrate_ecfp_bitstring")
            if not isinstance(bitstring, str) or not bitstring:
                bitstring, ok = _ecfp_bitstring_for_smiles(row.get("substrate_smiles"), 1024)
                if not ok:
                    raise ValueError(f"Invalid substrate structure for pair {row['pair_id']}")
            if len(bitstring) != 1024 or set(bitstring) - {"0", "1"}:
                raise ValueError("Substrate fingerprints must contain exactly 1024 binary characters")
            fingerprints.append(np.fromiter((int(value) for value in bitstring), dtype=np.float32, count=1024))
        features = np.concatenate([np.stack(fingerprints), matrix[selected.iloc[start:start + len(batch)].astype(int)]], axis=1).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Protein-substrate features must be finite")
        scores.append(booster.predict(xgb.DMatrix(features, feature_names=names)))
    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)
