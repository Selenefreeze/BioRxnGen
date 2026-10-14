from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


LABEL = "exact_agora_norm_no_proton_water"
UNIT = "reaction_unit_id"
SIMILARITY = "product_target_tanimoto"
POLICIES: dict[str, tuple[list[str], list[bool]]] = {
    "template_rank": (
        ["rule_rank", "product_rank_within_rule", "product_sets_returned"],
        [True, True, False],
    ),
    "enzymemap_reranker": (
        ["neural_score", "rule_rank", "product_rank_within_rule"],
        [False, True, True],
    ),
    "agora2_strict_reranker": (
        ["agora2_strict_reranker_score", "rule_rank", "product_rank_within_rule"],
        [False, True, True],
    ),
}


def replace_reranker_scores(scored: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Align fresh inference with the held-out candidates and comparison scores."""
    keys = [UNIT, "rule_rank", "product_rank_within_rule", "product_agora_norm_no_proton_water"]
    missing = set(keys + ["biorxngen_score"]).difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    test = predictions.loc[predictions[UNIT].isin(scored[UNIT])].reset_index(drop=True)
    if not test[keys].equals(scored[keys].reset_index(drop=True)):
        raise ValueError("Predictions must retain the released held-out candidate order")
    scores = pd.to_numeric(test["biorxngen_score"], errors="coerce").to_numpy()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Reranker scores must be finite probabilities")
    result = scored.copy()
    result["agora2_strict_reranker_score"] = scores
    return result


def summarize_heldout_policies(
    scored: pd.DataFrame,
    split_manifest: pd.DataFrame,
    topk: tuple[int, ...] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Recompute test metrics using every held-out reaction as denominator."""
    required_scored = {UNIT, LABEL, SIMILARITY}
    required_scored.update(column for columns, _ in POLICIES.values() for column in columns)
    missing_scored = required_scored.difference(scored.columns)
    missing_split = {UNIT, "split"}.difference(split_manifest.columns)
    if missing_scored or missing_split:
        raise ValueError(f"Missing columns: scored={sorted(missing_scored)}, split={sorted(missing_split)}")

    test_ids = split_manifest.loc[
        split_manifest["split"].eq("test"), UNIT
    ].astype(str)
    if test_ids.empty or test_ids.duplicated().any():
        raise ValueError("The held-out split must contain unique reaction-unit identifiers")
    test_set = set(test_ids)
    candidates = scored.copy()
    candidates[UNIT] = candidates[UNIT].astype(str)
    if not set(candidates[UNIT]).issubset(test_set):
        raise ValueError("Scored rows contain units outside the held-out split")
    candidates[LABEL] = candidates[LABEL].fillna(False).astype(bool)
    candidates[SIMILARITY] = pd.to_numeric(candidates[SIMILARITY], errors="coerce")

    groups = {key: part for key, part in candidates.groupby(UNIT, sort=False)}
    denominator = len(test_ids)
    rows: list[dict[str, Any]] = []

    def record(policy: str, rank: str | int, selected: dict[str, pd.DataFrame]) -> None:
        hits = 0
        similarities: list[float] = []
        for unit_id in test_ids:
            part = selected.get(unit_id)
            if part is None or part.empty:
                continue
            hits += int(part[LABEL].any())
            sim = part[SIMILARITY]
            if sim.notna().any():
                similarities.append(float(sim.max()))
        rows.append(
            {
                "dataset_scope": "test",
                "policy": policy,
                "topk": rank,
                "label_col": LABEL,
                "exact_hits": hits,
                "exact_rate": hits / denominator,
                "mean_max_tanimoto": float(np.mean(similarities)) if similarities else np.nan,
                "median_max_tanimoto": float(np.median(similarities)) if similarities else np.nan,
                "n_reactions": denominator,
            }
        )

    record("candidate_set", "all", groups)
    for policy, (columns, ascending) in POLICIES.items():
        ranked = {
            unit_id: part.sort_values(columns, ascending=ascending)
            for unit_id, part in groups.items()
        }
        for k in topk:
            record(policy, k, {unit_id: part.head(k) for unit_id, part in ranked.items()})
    return pd.DataFrame(rows)
