from __future__ import annotations
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import pandas as pd
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("tqdm")
CODE = Path(__file__).resolve().parents[1]
BUILDER = CODE / "examples/generation.py"
GENERATOR = CODE / "scripts/generation.py"
spec = spec_from_file_location("build_full_generation_sample", BUILDER)
sample_builder = module_from_spec(spec)
spec.loader.exec_module(sample_builder)
generator_spec = spec_from_file_location("run_full_generation", GENERATOR)
generation_entry = module_from_spec(generator_spec)
generator_spec.loader.exec_module(generation_entry)


def test_full_generation_cli_on_synthetic_inputs(tmp_path):
    paths = sample_builder.build_sample(tmp_path / "inputs")
    manifest = pd.read_csv(paths["worklist_manifest"])
    assert manifest.loc[0, "pair_partition_files"] == "worklist/pairs.parquet"
    assert manifest.loc[0, "enzyme_reference_file"] == "worklist/enzymes.parquet"
    output = tmp_path / "output"
    command = [
        sys.executable,
        str(GENERATOR),
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
        "--workers",
        "1",
        "--max-tasks",
        "1",
        "--max-pairs-per-task",
        "1",
        "--run",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "Status: completed" in result.stdout
    assert "four built-in basic cofactors" in result.stderr
    product_path = (
        output
        / "partitions/synthetic_segment/pair_part_00000/chunk_00000/candidate_products.parquet"
    )
    products = pd.read_parquet(product_path)
    assert not products.empty
    assert any(
        ("CC(=O)O" in str(side).split(".") for side in products["product_smiles"])
    )
    assert "pair_id" in products.columns
    assert not any((column.startswith("phase") for column in products.columns))
    assert (
        output
        / "partitions/synthetic_segment/pair_part_00000/chunk_00000/selected_pairs.parquet"
    ).is_file()


def test_flat_rule_directory_is_accepted(tmp_path):
    paths = sample_builder.build_sample(tmp_path / "inputs")
    output = tmp_path / "output"
    command = [
        sys.executable,
        str(GENERATOR),
        "run",
        "--artifact-root",
        str(paths["artifact_root"]),
        "--worklist-manifest",
        str(paths["worklist_manifest"]),
        "--rule-chunks",
        str(paths["rule_chunks"] / "reaction_labels"),
        "--split-assignments",
        str(paths["split_assignments"]),
        "--cofactor-pool",
        str(paths["cofactor_pool"]),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "Status: planned_only" in result.stdout
    assert (output / ".backend_inputs/reaction_labels/part_00000.parquet").is_file()


def test_public_product_projection_uses_semantic_pair_ids(tmp_path):
    source = tmp_path / "products.parquet"
    pd.DataFrame(
        [
            {
                "candidate_pair_id": "pair:example",
                "model_id": "strain_a",
                "ec_number": "3.1.1.1",
                "reaction_id": "reaction:example",
                "substrate_smiles": "CC(=O)OC",
                "generated_product_side_smiles": "CC(=O)O.CO",
                "candidate_rule_directional_id": "hydrolysis_rule",
                "pair_priority_score": 1.0,
            }
        ]
    ).to_parquet(source, index=False)
    columns = generation_entry.PUBLIC_OUTPUTS["candidate_products.parquet"][1]
    public = generation_entry.public_table(source, columns)
    assert public.loc[0, "pair_id"] == "pair:example"
    assert public.loc[0, "product_smiles"] == "CC(=O)O.CO"
    assert public.loc[0, "reaction_id"] == "reaction:example"
    assert "pair_priority_score" not in public.columns


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/generation.py"
spec = spec_from_file_location("prepare_full_data_manifest", SCRIPT)
generation_entry = module_from_spec(spec)
spec.loader.exec_module(generation_entry)


def test_relocate_archived_input_paths(tmp_path):
    old = "/research/artifacts/worklist/pairs.parquet"
    assert generation_entry.relocate(old, tmp_path) == "worklist/pairs.parquet"
    assert (
        generation_entry.relocate(old + "|" + old, tmp_path)
        == "worklist/pairs.parquet|worklist/pairs.parquet"
    )
    assert generation_entry.relocate(old, tmp_path, absolute=True) == str(
        tmp_path / "worklist/pairs.parquet"
    )
    assert generation_entry.relocate("", tmp_path) == ""


def test_relocate_rejects_paths_outside_artifact_root(tmp_path):
    with pytest.raises(ValueError, match="Unsafe artifact-relative"):
        generation_entry.relocate("../outside.parquet", tmp_path)
    with pytest.raises(ValueError, match="outside the artifact root"):
        generation_entry.relocate("D:/other/place/file.parquet", tmp_path)


import importlib.util


def test_biological_identifiers_are_not_run_labels(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/generation.py"
    spec = importlib.util.spec_from_file_location("generation_entry", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    isolate = str(34) + "a"
    frame = pd.DataFrame(
        {
            "candidate_pair_id": ["pair:abc"],
            "model_id": [f"M_Bacterium_{isolate}_isolate"],
            "ec_number": ["1.1.1.1"],
            "protein_accession": [f"gene_{isolate}_locus"],
            "substrate_uid": [f"metabolite_{isolate}_variant"],
            "reaction_id": [f"R_{isolate}_variant"],
            "generated_product_side_smiles": ["CC=O"],
        }
    )
    source = tmp_path / "products.parquet"
    frame.to_parquet(source, index=False)
    result = module.public_table(
        source, module.PUBLIC_OUTPUTS["candidate_products.parquet"][1]
    )
    assert result.model_id.iloc[0] == frame.model_id.iloc[0]
    assert result.protein_accession.iloc[0] == frame.protein_accession.iloc[0]
    assert result.substrate_id.iloc[0] == frame.substrate_uid.iloc[0]


def test_production_example_reads_public_product_schema(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/generation.py"
    for index in range(3):
        result = subprocess.run(
            [
                sys.executable,
                str(example),
                "--output-dir",
                str(tmp_path / f"example_{index}"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Synthetic product check passed" in result.stdout


from collections import Counter

SCRIPTS = Path(__file__).resolve().parents[1] / "reproduction/backend/scripts"
sys.path.insert(0, str(SCRIPTS))
from reaction_generator import generate_products


def test_reference_generator_runs_on_synthetic_reaction():
    smarts = "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[O:3].[C:4]O"
    prior = {("ec4", "3.1.1.1"): Counter({"esterase": 1})}
    rules = {"esterase": Counter({smarts: 1})}
    batch = pd.DataFrame(
        [
            {
                "candidate_edge_id": "sample_edge",
                "model_id": "sample_model",
                "ec_number": "3.1.1.1",
                "substrate_smiles": "CC(=O)OC",
                "pair_priority_score": 1.0,
            }
        ]
    )
    products, attempts, metrics = generate_products(
        batch,
        {},
        {
            "_preloaded_rule_context": (prior, rules),
            "_preloaded_cofactor_panel": {"h2o": "O"},
            "rule_topk_per_ec": 40,
            "max_contexts_per_candidate": 3,
            "max_products_per_rule": 12,
            "max_reactant_permutations": 128,
            "max_product_rows": 100000,
            "max_generation_pairs": 0,
        },
    )
    assert metrics["status"] == "ok"
    assert len(attempts) == 1
    assert not products.empty
    assert "CC(=O)O" in products.generated_product_side_smiles.iloc[0]


sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "reproduction/backend/scripts")
)
from generation_batch import build_partition_tasks


def test_regeneration_only_selects_recorded_partitions():
    manifest = pd.DataFrame(
        [
            {
                "segment_id": "strain_part",
                "source_namespace": "internal",
                "pair_partition_files": "a.parquet|b.parquet",
                "enzyme_reference_file": "enzymes.parquet",
            }
        ]
    )
    tasks = build_partition_tasks(
        manifest, {"selected_batches": {("strain_part", 1): {2, 5}}}
    )
    assert len(tasks) == 1
    assert tasks[0]["pair_partition_file"] == "b.parquet"
    assert tasks[0]["pair_partition_index"] == 1
