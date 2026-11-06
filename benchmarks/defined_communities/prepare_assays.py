"""Extract condition-level molecular inventories from published supplementary tables."""
import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
from hashlib import file_digest
from io import BytesIO, TextIOWrapper
import json
from math import ceil, isfinite
from pathlib import Path
import re
from statistics import median
import warnings
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import openpyxl
from rdkit import Chem
from structures import structure_key

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

warnings.filterwarnings("ignore", message="Unknown extension is not supported.*")
IDENTITIES = []
OBSERVATIONS = []
CONDITIONS = []

SIHUMIX = {
    "Anaerostipes caccae DSM 14662": "Anaerostipes_caccae_DSM_14662",
    "Bacteroides thetaiotaomicron DSM 2079": "Bacteroides_thetaiotaomicron_VPI_5482",
    "Bifidobacterium longum NCC 2705": "Bifidobacterium_longum_NCC2705",
    "Blautia producta DSM 2950": "Blautia_producta_DSM_2950",
    "Clostridium butyricum DSM 10702": "Clostridium_butyricum_DSM_10702",
    "Thomasclavelia ramosa DSM 1402": "Clostridium_ramosum_VPI_0427_DSM_1402",
    "Escherichia coli K-12 MG1655": "Escherichia_coli_str_K_12_substr_MG1655",
    "Lactiplantibacillus plantarum DSM 20174": "Lactobacillus_plantarum_subsp_plantarum_ATCC_14917",
}
HAN = {
    "Bacteroides thetaiotaomicron VPI 5482": "Bacteroides_thetaiotaomicron_VPI_5482",
    "Collinsella aerofaciens ATCC 25986": "Collinsella_aerofaciens_ATCC_25986",
    "Eubacterium rectale ATCC 33656": "Eubacterium_rectale_ATCC_33656",
    "Parabacteroides distasonis ATCC 8503": "Parabacteroides_distasonis_ATCC_8503",
    "Edwardsiella tarda ATCC 23685": "Edwardsiella_tarda_ATCC_23685",
    "Clostridium sporogenes ATCC 15579": "Clostridium_sporogenes_ATCC_15579",
}


def records(sheet):
    header, *rows = sheet.values
    return [dict(zip(header, row)) for row in rows]


def number(value):
    if isinstance(value, (int, float)) and isfinite(value) and value >= 0:
        return value
    return None


@lru_cache(maxsize=None)
def key_and_inchikey(smiles):
    key = structure_key(smiles)
    return (key, Chem.MolToInchiKey(Chem.MolFromSmiles(key))) if key else ("", "")


def add_identity(dataset, feature, name, smiles="", basis="", cids="", keys=""):
    canonical, key = key_and_inchikey(smiles)
    row = {
        "dataset": dataset, "source_feature": feature, "source_label": name,
        "assignment": basis, "source_parent_smiles": smiles,
        "source_pubchem_cids": cids, "source_inchikeys": keys,
        "connectivity_smiles": canonical, "connectivity_inchikey": key,
    }
    IDENTITIES.append(row)
    return row


def compare(culture, control, expected_culture, expected_control):
    if (expected_culture < 3 or expected_control < 3
            or len(culture) != expected_culture or len(control) != expected_control):
        return ""
    cm, bm = median(culture), median(control)
    return int(sum(v > max(control) for v in culture) >= ceil(len(culture) / 2)
               and (cm >= 1.25 * bm if bm > 0 else cm > 0))


def add_measurement(dataset, condition, identity, culture, control=None, *, comparison=True, plotted=False):
    x = [number(v) for v in culture]
    y = [number(v) for v in control] if control is not None else []
    xv = [v for v in x if v is not None]
    yv = [v for v in y if v is not None]
    n_pos = sum(v > 0 for v in xv)
    OBSERVATIONS.append({
        "dataset": dataset, "condition": condition,
        "source_feature": identity["source_feature"],
        "source_label": identity["source_label"],
        "connectivity_smiles": identity["connectivity_smiles"],
        "connectivity_inchikey": identity["connectivity_inchikey"],
        "detected": int(n_pos > 0 or plotted),
        "positive_signal_n": "" if plotted else n_pos, "valid_sample_n": "" if plotted else len(xv),
        "valid_control_n": len(yv), "sample_median": median(xv) if xv else "",
        "control_median": median(yv) if yv else "",
        "above_control": compare(xv, yv, len(x), len(y)) if comparison else "",
    })


def add_condition(dataset, condition, members, matrix, sample_ids, control_ids=(),
                  scale="replicate", sample_n=None, control_n=None):
    CONDITIONS.append({
        "dataset": dataset, "condition": condition, "strains": len(members),
        "matrix": matrix, "source_sample_n": sample_n if sample_n is not None else (len(sample_ids) or ""),
        "source_control_n": control_n if control_n is not None else len(control_ids), "sampling_unit": scale,
        "samples": "|".join(map(str, sample_ids)),
        "controls": "|".join(map(str, control_ids)),
        "agora2_model_ids": "|".join("M_" + x for x in members),
    })


def read_farag(source):
    dataset = "Farag2020_SIHUMIx"
    t = ET.fromstring(ZipFile(source / "farag_supplement.docx").read("word/document.xml"))
    tab = t.findall(".//{*}tbl")[2]
    rows = [[" ".join(v.text or "" for v in c.findall(".//{*}t"))
             for c in row.findall("./{*}tc")] for row in tab.findall("./{*}tr")]
    assert rows[0][:2] == ["Category", "Metabolite"] and len(rows) == 144
    parent = {}
    for line in (DATA / "farag_parent_structures.txt").read_text(encoding="utf-8-sig").splitlines():
        if line and not line.startswith("#"):
            i, name, smiles = line.split("|")
            parent[int(i)] = (name, smiles)
    groups = [f"{food} {hour} h" for food in rows[0][2:] for hour in ("0.5", "24")]
    assert len(groups) == 16
    for group in groups:
        add_condition(dataset, group, list(SIHUMIX.values()), "GC-MS relative peak area",
                      (group,), scale="published group mean")
    for i, row in enumerate(rows[2:], 3):
        if row[1] == "Total":
            continue
        name, smiles = parent.get(i, ("", ""))
        identity = add_identity(dataset, f"Table S3 row {i}", row[1], smiles,
                                f"GC-MS putative parent: {name}" if smiles else "unresolved annotation")
        if smiles and not identity["connectivity_smiles"]:
            raise ValueError((row[1], name, smiles))
        for condition, value in zip(groups, row[2:], strict=True):
            if value.strip() in {"-", "–", "—", ""}:
                value = None
            else:
                value = float(value.split("±", 1)[0].strip())
            add_measurement(dataset, condition, identity, (value,))


def han_identities(source):
    dataset = "Han2021"
    library = openpyxl.load_workbook(source / "han_table1.xlsx", read_only=True, data_only=True)
    grouped = defaultdict(list)
    for row in records(library["chemical_info"]):
        grouped[row["dname"]].append(row)
    result = {}
    for feature, entries in grouped.items():
        if not isinstance(feature, str):
            continue
        names = sorted({str(r["Compound"]) for r in entries if r["Compound"] is not None})
        smiles = [r["canonical_smiles"] for r in entries]
        key_set = {structure_key(x) for x in smiles}
        cids = sorted({str(r["PubChem_CID"]) for r in entries if r["PubChem_CID"] is not None})
        keys = sorted({str(r["inchikey"]) for r in entries if r["inchikey"] is not None})
        if any(name.startswith("IS_") for name in names):
            status = "spiked internal standard"
            parent = ""
        elif len(key_set) == 1 and "" not in key_set:
            status = "reference standard, unambiguous connectivity"
            parent = smiles[0]
        else:
            status = "co-elution across connectivities or no resolvable structure"
            parent = ""
        result[feature] = add_identity(dataset, feature, " | ".join(names), parent,
                                       status, "|".join(cids), "|".join(keys))
    library.close()
    return result


def collect_han(dataset, matrix, identity, combinations):
    features, *rows = matrix.values
    values = {r[0]: r for r in rows}
    assert len(values) == len(rows)
    for condition, matrix_label, strains, samples, controls in combinations:
        assert all(i in values for i in samples + controls)
        add_condition(dataset, condition, strains, matrix_label, samples, controls)
        for col, feature in enumerate(features[1:], 1):
            annotation = identity.get(feature)
            if annotation is None:
                annotation = add_identity("Han2021", feature, feature, basis="unlisted library feature")
                identity[feature] = annotation
            add_measurement(dataset, condition, annotation, [values[i][col] for i in samples],
                            [values[i][col] for i in controls] if controls else None)


def read_han(source):
    identity = han_identities(source)
    vivo = openpyxl.load_workbook(source / "han_table8.xlsx", read_only=True, data_only=True)
    metadata = [r for r in records(vivo["metadata"]) if r["experiment"] == "community"]
    labels = {"Bt_Ca_Er_Pd_Et": "5-strain", "Cs_Bt_Ca_Er_Pd_Et": "6-strain"}
    combos = []
    for source_label, group in labels.items():
        members = list(HAN.values())[:int(group[0])]
        for matrix in ("caecal", "feces", "serum", "urine"):
            samples = [r["sample_id"] for r in metadata
                       if r["sample_type"] == matrix and r["colonization"] == source_label]
            blanks = [r["sample_id"] for r in metadata
                      if r["sample_type"] == matrix and r["colonization"] == "germ-free"]
            combos.append((f"{group} {matrix}", matrix, members, samples, blanks))
    collect_han("Han2021_defined_in_vivo", vivo["istd_corr_ion_count_matrix"], identity, combos)
    vivo.close()
    vitro = openpyxl.load_workbook(source / "han_table7.xlsx", read_only=True, data_only=True)
    metadata = records(vitro["aggregated_md"])
    groups = defaultdict(list)
    blanks = defaultdict(list)
    sample_times = {}
    for r in metadata:
        run = (str(r["experiment"]), r["media"])
        sample_times[r[None]] = r["subculture_time"]
        if r["sample_type"] == "media_blank":
            blanks[run].append(r[None])
        if r["sample_type"] == "supernatant" and r["taxonomy"] in HAN:
            groups[(r["taxonomy"], *run)].append(r[None])
    combos = []
    for (taxonomy, experiment, medium), samples in sorted(groups.items()):
        measured_times = {sample_times[i] for i in samples if sample_times[i] is not None}
        controls = [i for i in blanks[(experiment, medium)]
                    if not measured_times or sample_times[i] is None or sample_times[i] in measured_times]
        combos.append((f"{taxonomy} {medium} {experiment}", f"culture supernatant in {medium}",
                       [HAN[taxonomy]], samples, controls))
    collect_han("Han2021_matched_monocultures", vitro["count.ps"], identity, combos)
    vitro.close()


def read_weiss_untargeted(source, members):
    dataset = "Weiss2022_OMM_monocultures"
    props = {str(row["CID"]): row for row in
             json.loads((DATA / "weiss2022_pubchem.json").read_text(encoding="utf8"))}
    blanks = [f"SM{i}_AAM" for i in range(1, 4)]
    for label, model in sorted(members.items()):
        samples = [f"SM{i}_{label}" for i in range(1, 4)]
        add_condition(dataset, label, [model], "culture supernatant",
                      samples, blanks, scale="deposited processed intensity")
    with ZipFile(source / "weiss2022_intensity_matrix.zip") as archive:
        for name in sorted(n for n in archive.namelist() if n.startswith("m_MTBLS3535_")):
            mode = re.search(r"LC-MS_(negative|positive)_hilic", name)[1]
            with archive.open(name) as raw:
                reader = csv.DictReader(TextIOWrapper(raw, encoding="utf-8-sig"), delimiter="\t")
                rows = list(reader)
            columns = reader.fieldnames[21:]
            features = defaultdict(list)
            for i, row in enumerate(rows, 2):
                features[tuple(row[k] for k in columns)].append((i, row))
            for records_with_same_trace in features.values():
                data = [r for _, r in records_with_same_trace]
                cids = sorted({r["database_identifier"] for r in data})
                keys = sorted({r["inchi"] for r in data})
                assert all(props[r["database_identifier"]]["InChIKey"][:14] == r["inchi"][:14] for r in data)
                parent_candidates = [props[cid]["SMILES"] for cid in cids]
                candidates = {structure_key(s) for s in parent_candidates}
                good = len(candidates) == 1 and "" not in candidates
                feature = mode + " MAF rows " + ",".join(str(i) for i, _ in records_with_same_trace)
                identity = add_identity(dataset, feature,
                                        " | ".join(sorted({r["metabolite_identification"].strip() for r in data})),
                                        parent_candidates[0] if good else "",
                                        "deposited putative assignment"
                                        if good else "identical intensity trace with multiple candidate connectivities",
                                        "|".join(cids), "|".join(keys))
                trace = data[0]
                intensity = lambda name: None if trace[name] in {"NA", ""} else float(trace[name])
                for label in sorted(members):
                    samples = [f"SM{i}_{label}" for i in range(1, 4)]
                    add_measurement(dataset, label, identity, [intensity(k) for k in samples],
                                    [intensity(k) for k in blanks], comparison=False)


def write_tsv(out, name, rows):
    with (out / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    lookup = defaultdict(list)
    for row in OBSERVATIONS:
        lookup[row["dataset"], row["condition"]].append(row)
    for row in CONDITIONS:
        rr = lookup[row["dataset"], row["condition"]]
        sample_features = [x for x in rr if x["detected"]]
        # Repeated detection is calculated on the same feature rather than pooling signals between modes.
        detected = {x["connectivity_inchikey"] for x in sample_features if x["connectivity_inchikey"]}
        repeated = {x["connectivity_inchikey"] for x in sample_features
                    if x["connectivity_inchikey"] and (x["positive_signal_n"] or 0) >= 2}
        assessed = {x["connectivity_inchikey"] for x in rr
                    if x["connectivity_inchikey"] and x["above_control"] != ""}
        above = {x["connectivity_inchikey"] for x in rr
                 if x["connectivity_inchikey"] and x["above_control"] == 1}
        assert above <= detected and repeated <= detected
        row["detected_source_features"] = len(sample_features)
        row["unique_structures_detected"] = len(detected)
        row["unique_structures_in_2plus_replicates"] = len(repeated) if (row["source_sample_n"] or 0) >= 2 else ""
        row["unique_structures_assessable_against_control"] = len(assessed) if assessed else ""
        row["unique_structures_above_control"] = len(above) if assessed else ""


def prepare(source, out):
    for record in json.loads((DATA / "source_data.json").read_text(encoding="utf-8")):
        with (source / record["file"]).open("rb") as handle:
            if file_digest(handle, "sha256").hexdigest() != record["sha256"]:
                raise ValueError(f"Source checksum mismatch: {record['file']}")
    with (DATA / "conditions.tsv").open(encoding="utf-8", newline="") as handle:
        members = {
            row["condition"]: row["agora2_model_ids"].removeprefix("M_")
            for row in csv.DictReader(handle, delimiter="\t")
            if row["dataset"] == "Weiss2022_OMM_monocultures"
        }
    read_farag(source)
    read_han(source)
    read_weiss_untargeted(source, members)
    summarize()
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out, "conditions.tsv", CONDITIONS)
    write_tsv(out, "compound_observations.tsv", OBSERVATIONS)
    write_tsv(out, "compound_identities.tsv", IDENTITIES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
