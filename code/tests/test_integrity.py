from __future__ import annotations
import sys
import subprocess
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import hashlib
import importlib.util
from pathlib import Path
import pandas as pd
import pytest


def load_verifier():
    path = Path(__file__).resolve().parents[1] / "scripts/validation.py"
    spec = importlib.util.spec_from_file_location("release_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checksum_verification_and_corruption(tmp_path: Path) -> None:
    payload = tmp_path / "input.tsv"
    payload.write_bytes(b"x\t1\n")
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [
            {
                "file": "input.tsv",
                "bytes": 4,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ]
    ).to_csv(manifest, sep="\t", index=False)
    verifier = load_verifier()
    assert verifier.verify(tmp_path, manifest) == 1
    payload.write_bytes(b"x\t2\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verifier.verify(tmp_path, manifest)


def test_manifest_rejects_parent_path(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([{"file": "../outside.tsv", "bytes": 0, "sha256": ""}]).to_csv(
        manifest, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="out-of-root"):
        load_verifier().verify(tmp_path, manifest)


@pytest.mark.parametrize("custom_manifest", [False, True])
def test_manifest_cli_paths(
    tmp_path: Path, monkeypatch, capsys, custom_manifest: bool
) -> None:
    payload = tmp_path / "input.tsv"
    payload.write_bytes(b"x\t1\n")
    manifest = tmp_path / (
        "file_manifest.tsv" if custom_manifest else "code/validation/checksums.tsv"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "file": "input.tsv",
                "bytes": 4,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ]
    ).to_csv(manifest, sep="\t", index=False)
    args = ["validation.py", "--root", str(tmp_path)]
    if custom_manifest:
        args.extend(["--manifest", str(manifest)])
    monkeypatch.setattr(sys, "argv", args)
    load_verifier().verify_files()
    assert capsys.readouterr().out.strip() == "Verified 1 files."


import csv
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from validation import FIELDS, audit


def test_replacement_batches_are_counted_once(tmp_path: Path, monkeypatch) -> None:
    rows = []
    for scope, count in (("standard", 2), ("regenerated", 3)):
        relative = f"candidate_products/{scope}/batch_00001.parquet"
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        columns = {
            name: (
                pa.array(list(range(count)) if name == "row_index" else [1] * count)
                if name in {"rule_rank", "product_rank_within_rule", "row_index"}
                else pa.array(["value"] * count)
            )
            for name in FIELDS
        }
        pq.write_table(pa.table(columns), path)
        rows.append(
            {
                "scope": scope,
                "batch_id": "batch_00001",
                "file": relative,
                "rows": count,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "metadata/candidate_product_manifest.tsv"
    manifest.parent.mkdir()
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    result = audit(tmp_path, manifest, check_hashes=True)
    assert result == {
        "standard_batches": 1,
        "standard_product_rows": 2,
        "regenerated_batches": 1,
        "regenerated_product_rows": 3,
        "replaced_standard_rows": 2,
    }
    import validation as audit_candidate_products

    monkeypatch.setattr(
        sys,
        "argv",
        ["validation.py", "--data-root", str(tmp_path), "--check-hashes"],
    )
    audit_candidate_products.products()


@pytest.mark.parametrize(
    "group,command",
    [
        ("generation", "worklist"),
        ("generation", "run"),
        ("generation", "relocate"),
        ("reranker", "train"),
        ("reranker", "score"),
        ("benchmark", "internal"),
        ("benchmark", "external"),
        ("benchmark", "recovery"),
        ("compatibility", "score"),
        ("compatibility", "metrics"),
        ("resource", "classify"),
        ("resource", "regenerated"),
        ("resource", "coverage"),
        ("resource", "database"),
        ("validation", "files"),
        ("validation", "inputs"),
        ("validation", "products"),
    ],
)
def test_task_entry_points(group, command):
    script = Path(__file__).resolve().parents[1] / "scripts" / f"{group}.py"
    result = subprocess.run(
        [sys.executable, str(script), command, "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_source_data_workbook_index():
    root = Path(__file__).resolve().parents[2] / "data"
    index = pd.read_csv(root / "results/table_index.tsv", sep="\t")
    assert len(index) == 35
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    for filename, group in index.groupby("file"):
        with ZipFile(root / filename) as archive:
            assert archive.testzip() is None
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            names = {
                sheet.attrib["name"]
                for sheet in workbook.findall("s:sheets/s:sheet", namespace)
            }
            assert set(group.sheet).issubset(names)
