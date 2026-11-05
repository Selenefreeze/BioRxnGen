"""Freeze the assay panel and recover condition-matched original replicates."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import re

import numpy as np
import openpyxl
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structures import structure_key

ROOT = Path.cwd()
STRUCTURES = {
    "Serine": "N[C@@H](CO)C(=O)O",
    "Arginine": "N[C@@H](CCCNC(=N)N)C(=O)O",
    "Glutamine": "N[C@@H](CCC(N)=O)C(=O)O",
    "Glutamic acid": "N[C@@H](CCC(=O)O)C(=O)O",
    "Histidine": "N[C@@H](Cc1cnc[nH]1)C(=O)O",
    "Isoleucine/Leucine": "CC[C@H](C)[C@H](N)C(=O)O|CC(C)C[C@H](N)C(=O)O",
    "Lysine": "N[C@@H](CCCCN)C(=O)O",
    "Methionine": "N[C@@H](CCSC)C(=O)O",
    "Phenylalanine": "N[C@@H](Cc1ccccc1)C(=O)O",
    "Proline": "O=C(O)[C@@H]1CCCN1",
    "Threonine": "C[C@@H](O)[C@H](N)C(=O)O",
    "Tryptophane": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
    "Valine": "N[C@@H](C(C)C)C(=O)O",
    "Alanine": "N[C@@H](C)C(=O)O",
    "Tyrosine": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "Aspartic acid": "N[C@@H](CC(=O)O)C(=O)O",
    "Asparagine": "N[C@@H](CC(N)=O)C(=O)O",
    "Acetic acid": "CC(=O)O",
    "Propionic acid": "CCC(=O)O",
    "Butyric acid": "CCCC(=O)O",
    "Isobutyric acid": "CC(C)C(=O)O",
    "2-Methylbutyric acid": "CCC(C)C(=O)O",
    "Isovaleric acid": "CC(C)CC(=O)O",
    "Valeric acid": "CCCCC(=O)O",
    "4-Methyl valeric acid": "CC(C)CCC(=O)O",
    "Lactic acid": "CC(O)C(=O)O",
    "Desaminotyrosine": "O=C(O)CCc1ccc(O)cc1",
    "Succinic acid": "O=C(O)CCC(=O)O",
}


def main():
    with ZipFile(ROOT / "sources" / "OMM12_SourceData.zip") as archive:
        name, = [n for n in archive.namelist() if n.endswith("table_1_revision.xlsx") and not n.startswith("__")]
        workbook = openpyxl.load_workbook(BytesIO(archive.read(name)), read_only=True, data_only=True)
    records = []
    for tab in ("Tab 9", "Tab10"):
        rows = list(workbook[tab].values)
        names = [str(value).strip() for value in rows[0][1:]]
        for medium in ("AF", "APF"):
            for group in ("blank", "spent"):
                if tab == "Tab 9":
                    pattern = rf"{medium}_Blank_[1-3]" if group == "blank" else rf"OMM12_{medium}_E[1-3]"
                else:
                    pattern = rf"Blank_{medium}_S[1-3]_W1" if group == "blank" else rf"OMM12_{medium}_S[1-3]_W1"
                samples = [row for row in rows[1:] if re.fullmatch(pattern, str(row[0]))]
                if len(samples) != 3:
                    raise ValueError((tab, medium, group, pattern, [r[0] for r in rows[1:25]]))
                for sample in samples:
                    for analyte, value in zip(names, sample[1:]):
                        records.append(dict(tab=tab, medium=medium, group=group, sample=sample[0],
                                            analyte=analyte, value=value))
    data = pd.DataFrame(records)
    targets = []
    for (medium, analyte), frame in data.groupby(["medium","analyte"], sort=True):
        blank = pd.to_numeric(frame[frame.group.eq("blank")].value, errors="coerce").to_numpy()
        spent = pd.to_numeric(frame[frame.group.eq("spent")].value, errors="coerce").to_numpy()
        valid = len(blank) == len(spent) == 3 and np.isfinite(np.r_[blank,spent]).all() and min(*blank,*spent) >= 0
        fold = np.median(spent)/np.median(blank) if valid and np.median(blank)>0 else np.nan
        distinct = int(sum(spent>max(blank))) if valid else 0
        keys = [structure_key(x) for x in STRUCTURES[analyte].split("|")]
        assert all(keys)
        targets.append(dict(medium=medium, analyte=analyte, tab=frame.tab.iloc[0],
                            smiles=STRUCTURES[analyte], structure_keys="|".join(keys), single_structure=len(keys)==1,
                            valid=valid, blank_median=np.median(blank), spent_median=np.median(spent),
                            fold_change=fold, spent_above_max_blank=distinct,
                            enriched=bool(valid and distinct>=2 and (fold>=1.25 if np.median(blank)>0 else np.median(spent)>0))))
    data.to_csv(ROOT / "inputs" / "assay_replicates.tsv", sep="\t", index=False)
    pd.DataFrame(targets).to_csv(ROOT / "inputs" / "assay_targets.tsv", sep="\t", index=False)
    print(pd.DataFrame(targets).groupby("medium").enriched.agg(["size","sum"]).to_string())


if __name__ == "__main__":
    main()
