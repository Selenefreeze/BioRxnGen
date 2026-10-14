from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import pytest
from biorxngen.graph_audit import (
    DATABASE,
    IDENTITY,
    NATIVE,
    OVERLAP,
    UnionStats,
    classify_signatures,
    generation_side,
    prepare_products,
    profile_products,
    signature_id,
    summary_table,
)
from biorxngen.database_reference import validate_signature_table


def test_generation_identity_convention():
    assert generation_side("O.C[C@H](O)C") == "CC(C)O"
    assert signature_id("1.1.1.1", "CCO>>CC=O").startswith("signature:")


def test_archived_classification_and_identity_boundary():
    profile = pd.DataFrame(
        {
            "signature_id": ["a", "b", "c", "d"],
            "nonself_rows": [2, 0, 3, 0],
            "native_nonself_rows": [2, 0, 0, 0],
            "ec_number": ["1.1.1.1"] * 4,
        }
    )
    validation = pd.DataFrame(
        {
            "signature_id": ["a", "b", "c", "d"],
            "validation_tier": [
                "no_exact_rhea_match",
                "rhea_exact_secondary_non_agora",
                "agora_overlap",
                "no_exact_rhea_match",
            ],
        }
    )
    result = classify_signatures(profile, validation)
    assert result.candidate_class.tolist() == [NATIVE, DATABASE, OVERLAP, IDENTITY]
    assert not result.nonidentity_database_candidate.any()
    with pytest.raises(ValueError, match="incomplete"):
        classify_signatures(profile, validation.iloc[:1])


def test_real_graph_sample():
    root = Path(__file__).resolve().parents[2] / "data/raw_sample"
    products = pd.read_parquet(root / "graph_candidate_products.parquet")
    pairs = pd.read_parquet(root / "graph_pair_contexts.parquet")
    validation = pd.read_parquet(root / "graph_signature_validation.parquet")
    expected = pd.read_parquet(root / "graph_sample_reference.parquet")
    actual = classify_signatures(
        profile_products(prepare_products(products, pairs)), validation
    )
    pd.testing.assert_frame_equal(actual, expected)
    with pytest.raises(ValueError, match="matching pair"):
        prepare_products(products, pairs.iloc[:0])


def test_frozen_rhea_matching(tmp_path):
    pd.DataFrame([[100, "CCO>>CC=O"]]).to_csv(
        tmp_path / "rhea-reaction-smiles.tsv", sep="\t", header=False, index=False
    )
    pd.DataFrame(
        [[100, 101, 102, 103]],
        columns=["RHEA_ID_MASTER", "RHEA_ID_LR", "RHEA_ID_RL", "RHEA_ID_BI"],
    ).to_csv(tmp_path / "rhea-directions.tsv", sep="\t", index=False)
    for name, value in [
        ("rhea2ec.tsv", "1.1.1.1"),
        ("rhea2metacyc.tsv", "ALCOHOL-DEHYDROGENASE-RXN"),
        ("rhea2kegg_reaction.tsv", "R00001"),
    ]:
        pd.DataFrame(
            [[100, 100, value]], columns=["RHEA_ID", "MASTER_ID", "ID"]
        ).to_csv(tmp_path / name, sep="\t", index=False)
    signatures = pd.DataFrame(
        {
            "signature_id": ["a"],
            "ec_number": ["1.1.1.1"],
            "reaction_smiles": ["CCO>>CC=O"],
        }
    )
    reference = {
        "reaction_ids": set(),
        "metacyc_ids": set(),
        "kegg_reaction_ids": set(),
    }
    assert (
        validate_signature_table(signatures, tmp_path, reference).validation_tier.iloc[
            0
        ]
        == "rhea_exact_secondary_non_agora"
    )
    reference["kegg_reaction_ids"].add("R00001")
    assert (
        validate_signature_table(signatures, tmp_path, reference).validation_tier.iloc[
            0
        ]
        == "agora_overlap"
    )


def test_incomplete_cache_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        validate_signature_table(pd.DataFrame(), tmp_path, {})


def test_source_table_reports_global_model_overlap():
    internal, external = (UnionStats(), UnionStats())
    internal.models.update(["shared", "internal_only"])
    external.models.update(["shared", "external_only"])
    stats = {
        (
            "enzyme_source_namespace",
            "internal_agora_gene_annotation_sequence",
        ): internal,
        (
            "enzyme_source_namespace",
            "external_third_party_strain_ec_sequence",
        ): external,
    }
    assert summary_table(
        stats, "enzyme_source_namespace"
    ).both_source_models.tolist() == [1, 1]


import importlib.util


def script():
    path = Path(__file__).resolve().parents[1] / "scripts/resource.py"
    spec = importlib.util.spec_from_file_location("regenerated_map", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(root, records, scope="regenerated"):
    (root / "metadata").mkdir(exist_ok=True)
    pd.DataFrame(records).to_parquet(root / "products.parquet", index=False)
    pd.DataFrame(
        [
            {
                "scope": scope,
                "batch_id": "batch",
                "file": "products.parquet",
                "rows": len(records),
            }
        ]
    ).to_csv(root / "metadata/candidate_product_manifest.tsv", sep="\t", index=False)


def test_product_counts_and_ec_defined_signatures(tmp_path):
    inputs(
        tmp_path,
        [
            {"ec_number": "3.1.1.1", "reaction_smiles": "C>>O"},
            {"ec_number": "3.1.1.1", "reaction_smiles": "C>>O"},
            {"ec_number": "3.1.1.2", "reaction_smiles": "C>>O"},
            {"ec_number": "4.1.1.1", "reaction_smiles": "C>>O"},
        ],
    )
    result, batches = script().summarize(tmp_path)
    assert batches == 1
    assert result.set_index("ec_class").to_dict("index") == {
        "EC3": {"rows": 3, "unique_reaction_signatures": 2},
        "EC4": {"rows": 1, "unique_reaction_signatures": 1},
    }


def test_standard_records_are_not_counted(tmp_path):
    inputs(
        tmp_path,
        [{"ec_number": "3.1.1.1", "reaction_smiles": "C>>O"}],
        scope="standard",
    )
    with pytest.raises(ValueError, match="empty"):
        script().summarize(tmp_path)


def test_incorrect_partition_count_is_rejected(tmp_path):
    inputs(tmp_path, [{"ec_number": "3.1.1.1", "reaction_smiles": "C>>O"}])
    manifest = tmp_path / "metadata/candidate_product_manifest.tsv"
    frame = pd.read_csv(manifest, sep="\t")
    frame["rows"] = 2
    frame.to_csv(manifest, sep="\t", index=False)
    with pytest.raises(ValueError, match="count differs"):
        script().summarize(tmp_path)
