from __future__ import annotations
import sys
import pandas as pd
import pytest
from biorxngen.paper_benchmark import (
    replace_reranker_scores,
    summarize_heldout_policies,
)


def test_heldout_denominator_includes_units_without_candidates() -> None:
    split = pd.DataFrame(
        {"reaction_unit_id": ["u1", "u2", "u3"], "split": ["test", "test", "test"]}
    )
    scored = pd.DataFrame(
        {
            "reaction_unit_id": ["u1", "u1", "u2"],
            "exact_agora_norm_no_proton_water": [False, True, False],
            "product_target_tanimoto": [0.2, 1.0, 0.5],
            "rule_rank": [1, 2, 1],
            "product_rank_within_rule": [1, 1, 1],
            "product_sets_returned": [1, 1, 1],
            "neural_score": [0.2, 0.9, 0.4],
            "agora2_strict_reranker_score": [0.1, 0.9, 0.4],
        }
    )
    result = summarize_heldout_policies(scored, split, topk=(1, 2))
    lookup = {
        (row.policy, str(row.topk)): row for row in result.itertuples(index=False)
    }
    assert lookup["candidate_set", "all"].exact_hits == 1
    assert lookup["template_rank", "1"].exact_hits == 0
    assert lookup["enzymemap_reranker", "1"].exact_hits == 1
    assert lookup["agora2_strict_reranker", "1"].exact_hits == 1
    assert all((row.n_reactions == 3 for row in result.itertuples(index=False)))


def score_frames():
    scored = pd.DataFrame(
        {
            "reaction_unit_id": ["u1", "u1"],
            "rule_rank": [1, 1],
            "product_rank_within_rule": [1, 1],
            "product_agora_norm_no_proton_water": ["CO", "CO"],
            "agora2_strict_reranker_score": [0.2, 0.3],
            "neural_score": [0.7, 0.8],
        }
    )
    predicted = scored.drop(columns=["agora2_strict_reranker_score", "neural_score"])
    predicted["biorxngen_score"] = [0.9, 0.4]
    return (scored, predicted)


def test_fresh_scores_preserve_candidates_and_baselines():
    scored, predicted = score_frames()
    result = replace_reranker_scores(scored, predicted)
    assert result.agora2_strict_reranker_score.tolist() == [0.9, 0.4]
    assert result.neural_score.equals(scored.neural_score)
    assert scored.agora2_strict_reranker_score.tolist() == [0.2, 0.3]


def test_fresh_scores_require_all_heldout_candidates():
    scored, predicted = score_frames()
    with pytest.raises(ValueError, match="candidate order"):
        replace_reranker_scores(scored, predicted.iloc[:1])


def test_fresh_scores_reject_invalid_probabilities():
    scored, predicted = score_frames()
    predicted.loc[0, "biorxngen_score"] = float("nan")
    with pytest.raises(ValueError, match="finite probabilities"):
        replace_reranker_scores(scored, predicted)


from pathlib import Path
import numpy as np
from biorxngen.external_workflow import make_prediction_rows, select_benchmark_units
from biorxngen.production_rules import rule_modules


def test_selection_deduplicates_models_and_preserves_ordered_ec_combination(tmp_path):
    rows = [
        dict(
            reaction_id=rid,
            model_id=model,
            in_micromap="True",
            ec_numbers_for_generation=ecs,
            reactant_side_smiles_all="CCO",
            product_side_smiles_all="CC=O",
        )
        for rid, model, ecs in [
            ("r1", "m1", "1.1.1.1|2.1.1.1"),
            ("r1", "m2", "1.1.1.1|2.1.1.1"),
            ("r2", "m3", "1.1.1.1|2.1.1.1"),
            ("r3", "m4", "2.1.1.1|1.1.1.1"),
        ]
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "worklist.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            dict(reaction_id=rid, complete_ready_strict_all_side_model_count=count)
            for rid, count in [("r1", 3), ("r2", 2), ("r3", 1)]
        ]
    ).to_csv(tmp_path / "support.tsv", sep="\t", index=False)
    selected, report = select_benchmark_units(
        pd,
        rule_modules(),
        tmp_path / "worklist.tsv",
        tmp_path / "support.tsv",
        {
            "max_benchmark_reactions": 1000,
            "max_per_ec_combo": 1,
            "require_micromap": True,
        },
    )
    assert selected.reaction_id.tolist() == ["r1", "r3"]
    assert selected.external_support_model_count_written_worklist.tolist() == [2, 1]
    assert report["ec_combo_selected"] == 2


def test_missing_smarts_never_enters_rule_prior(tmp_path):
    folder = tmp_path / "rules/reaction_labels"
    folder.mkdir(parents=True)
    columns = [
        "reaction_record_id",
        "mapped",
        "unmapped",
        "ec_num",
        "ec1",
        "ec2",
        "ec3",
        "ec4",
        "rule",
        "rule_directional_id",
        "rule_family_id",
        "rule_direction",
        "source",
        "steps",
        "quality",
        "n_reactants",
        "n_products",
        "n_bond_edits",
        "n_changed_atoms",
    ]
    rows = [{column: "" for column in columns} for _ in range(2)]
    for index, row in enumerate(rows):
        row.update(
            reaction_record_id=f"r{index}",
            ec_num="1.1.1.1",
            ec1="1",
            ec2="1.1",
            ec3="1.1.1",
            ec4="1.1.1.1",
            rule_directional_id=f"rule_{index}",
        )
    rows[0]["rule"] = "[C:1][O:2]>>[C:1]=[O:2]"
    rows[1]["rule"] = np.nan
    pd.DataFrame(rows).to_parquet(folder / "part_00000.parquet", index=False)
    pd.DataFrame(
        {"reaction_record_id": ["r0", "r1"], "split": ["train", "train"]}
    ).to_parquet(tmp_path / "split.parquet", index=False)
    prior, smarts, _, report = rule_modules()[
        "_load_rule_template_prior_and_eval_records"
    ](tmp_path / "rules", tmp_path / "split.parquet", [], 1, 42)
    assert set(smarts) == {"rule_0"}
    assert prior["ec4", "1.1.1.1"]["rule_0"] == 1
    assert report["loader_stats"]["train_missing_rule"] == 1


def test_external_ranking_deduplicates_normalized_products():
    rows = pd.DataFrame(
        {
            "benchmark_id": ["r1"] * 3,
            "rule_rank": [1, 2, 3],
            "product_rank_within_rule": [1, 1, 1],
            "rule_train_count": [100, 10, 1],
            "product_no_proton_water_nonisomeric": ["CC=O", "CC=O", "CCO"],
            "product_full_nonisomeric": ["CC=O", "CC=O.O", "CCO"],
            "neural_score": [0.1, 0.9, 0.5],
        }
    )
    result = pd.DataFrame(make_prediction_rows(rows))
    assert result.groupby("policy").size().to_dict() == {
        "hybrid_reranker": 2,
        "template_rank": 2,
    }
    assert result.loc[
        result.policy == "hybrid_reranker", "predicted_product_smiles"
    ].tolist() == ["CC=O.O", "CCO"]
