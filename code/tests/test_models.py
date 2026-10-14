from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")
from biorxngen.reranker import load_model, score_candidates, train_reranker

ROOT = Path(__file__).resolve().parents[2]
RERANKER_SAMPLE = ROOT / "data/raw_sample/reranker_feature_sample.parquet"
REFERENCE = ROOT / "data/raw_sample/reranker_score_reference.tsv"
RERANKER_CHECKPOINT = ROOT / "code/models/product_reranker.pt"


def test_frozen_checkpoint_matches_saved_scores():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
    rows = pd.read_parquet(RERANKER_SAMPLE).tail(40).copy()
    reference = pd.read_csv(REFERENCE, sep="\t")
    model, config = load_model(RERANKER_CHECKPOINT)
    rows["computed"] = score_candidates(rows, model, config)
    keys = [
        "reaction_unit_id",
        "rule_rank",
        "product_rank_within_rule",
        "product_agora_norm_no_proton_water",
    ]
    comparison = rows.merge(reference, on=keys, validate="one_to_one")
    assert len(comparison) == 40
    error = np.abs(comparison.computed - comparison.agora2_strict_reranker_score)
    assert error.max() < 0.001


def test_training_entry_point_saves_loadable_model(tmp_path):
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
    rows = pd.read_parquet(RERANKER_SAMPLE)
    split = pd.read_csv(ROOT / "data/benchmark_results/internal_split_manifest.csv")
    rows = rows.merge(
        split[["reaction_unit_id", "split"]],
        on="reaction_unit_id",
        validate="many_to_one",
    )
    train = rows.loc[rows["split"].eq("train")]
    validation = rows.loc[rows["split"].eq("valid")]
    checkpoint = tmp_path / "model.pt"
    report = train_reranker(train, validation, checkpoint, epochs=1, batch_size=64)
    assert report["train_rows"] == 92
    assert report["validation_rows"] == 92
    model, config = load_model(checkpoint)
    scores = score_candidates(validation.head(5), model, config)
    assert len(scores) == 5
    assert np.isfinite(scores).all()


pytest.importorskip("xgboost")
from biorxngen.compatibility import score_compatibility, verify_compatibility_scores

ESP_SAMPLE = ROOT / "data/raw_sample/compatibility_inference"
ESP_CHECKPOINT = ROOT / "code/models/esp_compatibility_classifier.json"


def inputs():
    return (
        pd.read_csv(ESP_SAMPLE / "pair_inputs.tsv", sep="\t"),
        pd.read_csv(ESP_SAMPLE / "protein_embedding_map.tsv", sep="\t"),
    )


def test_archived_pair_score():
    pairs, mapping = inputs()
    scores = score_compatibility(
        pairs, mapping, ESP_SAMPLE / "protein_embeddings.npy", ESP_CHECKPOINT
    )
    np.testing.assert_allclose(scores, pairs.reference_score, atol=1e-06, rtol=0)


def test_archived_cohort_sample():
    sample = ROOT / "data/raw_sample/compatibility_cohort"
    pairs = pd.read_csv(
        sample / "pair_inputs.tsv", sep="\t", dtype={"substrate_ecfp_bitstring": str}
    )
    mapping = pd.read_csv(sample / "protein_embedding_map.tsv", sep="\t")
    pairs["compatibility_score"] = score_compatibility(
        pairs, mapping, sample / "protein_embeddings.npy", ESP_CHECKPOINT
    )
    reference = pd.read_csv(sample / "reference_scores.tsv", sep="\t")
    assert verify_compatibility_scores(pairs, reference)["pairs_checked"] == 24


def test_missing_protein_mapping():
    pairs, mapping = inputs()
    pairs.loc[0, "protein_ref"] = "unmapped_protein"
    with pytest.raises(ValueError, match="no embedding-row mapping"):
        score_compatibility(
            pairs, mapping, ESP_SAMPLE / "protein_embeddings.npy", ESP_CHECKPOINT
        )


def test_out_of_range_embedding_index():
    pairs, mapping = inputs()
    mapping.loc[0, "embedding_index"] = 1
    with pytest.raises(ValueError, match="valid integer row positions"):
        score_compatibility(
            pairs, mapping, ESP_SAMPLE / "protein_embeddings.npy", ESP_CHECKPOINT
        )


def test_invalid_substrate_fingerprint():
    pairs, mapping = inputs()
    pairs["substrate_ecfp_bitstring"] = "010X"
    with pytest.raises(ValueError, match="1024 binary characters"):
        score_compatibility(
            pairs, mapping, ESP_SAMPLE / "protein_embeddings.npy", ESP_CHECKPOINT
        )


def test_score_comparison_joins_identifiers():
    observed = pd.DataFrame({"pair_id": ["a", "b"], "compatibility_score": [0.2, 0.4]})
    reference = observed.iloc[::-1].copy()
    assert verify_compatibility_scores(observed, reference)["pairs_checked"] == 2


def test_score_comparison_rejects_different_cohorts():
    observed = pd.DataFrame({"pair_id": ["a", "b"], "compatibility_score": [0.2, 0.4]})
    with pytest.raises(ValueError, match="agree exactly"):
        verify_compatibility_scores(observed, observed.head(1))


def test_score_comparison_rejects_duplicate_references():
    observed = pd.DataFrame({"pair_id": ["a"], "compatibility_score": [0.2]})
    with pytest.raises(ValueError, match="unique, non-null"):
        verify_compatibility_scores(observed, pd.concat([observed, observed]))


def test_score_comparison_reports_actual_difference():
    observed = pd.DataFrame({"pair_id": ["a"], "compatibility_score": [0.2]})
    reference = pd.DataFrame({"pair_id": ["a"], "compatibility_score": [0.3]})
    with pytest.raises(ValueError, match="comparison failed"):
        verify_compatibility_scores(observed, reference)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compatibility import summarize


def test_saved_score_summary_has_explicit_denominators():
    frame = pd.DataFrame(
        {
            "pair_id": ["pair-a", "pair-b", "pair-c", "pair-d"],
            "label": [1, 0, 1, 0],
            "split": ["test"] * 4,
            "decoy_class": [
                "known",
                "alternative enzyme",
                "known",
                "alternative substrate",
            ],
            "compatibility_score": [0.9, 0.1, 0.8, 0.2],
        }
    )
    result = summarize(frame)
    assert result["all_pairs"]["pairs"] == 4
    assert result["all_pairs"]["known_pairs"] == 2
    assert result["all_pairs"]["auroc"] == 1.0
    assert result["split_metrics"]["test"]["average_precision"] == 1.0


def test_saved_score_summary_rejects_duplicate_pair_ids():
    frame = pd.DataFrame(
        {
            "pair_id": ["pair-a", "pair-a"],
            "label": [1, 0],
            "split": ["test", "test"],
            "decoy_class": ["known", "alternative enzyme"],
            "compatibility_score": [0.9, 0.1],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        summarize(frame)
