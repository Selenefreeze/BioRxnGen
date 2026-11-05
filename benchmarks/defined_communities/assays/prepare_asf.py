"""Derive an assay panel for the exact-model ASF co-cultures."""
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from prepare_omm12 import ROOT, STRUCTURES, structure_key

MODELS = {
    "356": "Clostridium_sp_ASF356",
    "361": "Lactobacillus_murinus_ASF361",
    "492": "Eubacterium_plexicaudatum_ASF492",
    "519": "Parabacteroides_sp_ASF519",
}
IDENTITIES = {
    **{name: STRUCTURES[name] for name in (
        "Valine", "Alanine", "Methionine", "Asparagine", "Lysine", "Proline", "Serine",
        "Threonine", "Tyrosine", "Histidine", "Phenylalanine")},
    "Butyrate": "CCCC(=O)O",
    "Isovalerate": "CC(C)CC(=O)O",
    "Leucine": "CC(C)C[C@H](N)C(=O)O",
    "Isoleucine": "CC[C@H](C)[C@H](N)C(=O)O",
    "Propionate": "CCC(=O)O",
    "Ethanol": "CCO",
    "Lactate": "CC(O)C(=O)O",
    "Isocaproate": "CC(C)CCC(=O)O",
    "Acetate": "CC(=O)O",
    "Glutamate": STRUCTURES["Glutamic acid"],
    "Isobutyrate": "CC(C)C(=O)O",
    "Succinate": "O=C(O)CCC(=O)O",
    "2-Oxoisocaproate": "CC(C)CC(=O)C(=O)O",
    "Aspartate": STRUCTURES["Aspartic acid"],
    "Choline": "C[N+](C)(C)CCO",
    "Cystine": "NC(CSSCC(N)C(=O)O)C(=O)O",
    "Glycine": "NCC(=O)O",
    "Glycerol": "OCC(O)CO",
    "Betaine": "C[N+](C)(C)CC(=O)[O-]",
    "Isopropanol": "CC(C)O",
    "Pyroglutamate": "O=C(O)C1CCC(=O)N1",
    "Trehalose": "OCC1OC(OC2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O",
    "Lactose": "OCC1OC(OC2C(O)C(O)C(O)OC2CO)C(O)C(O)C1O",
    "Uracil": "O=c1cc[nH]c(=O)[nH]1",
    "Uridine": "O=c1ccn(C2OC(CO)C(O)C2O)c(=O)[nH]1",
    "Cytosine": "Nc1cc[nH]c(=O)n1",
    "Cytidine": "Nc1ccn(C2OC(CO)C(O)C2O)c(=O)n1",
    "Adenosine": "Nc1ncnc2c1ncn2C1OC(CO)C(O)C1O",
    "Inosine": "O=c1[nH]cnc2c1ncn2C1OC(CO)C(O)C1O",
    "Urocanate": "O=C(O)C=Cc1cnc[nH]1",
    "Fumarate": "O=C(O)/C=C/C(=O)O",
    "Tyramine": "NCCc1ccc(O)cc1",
    "Tryptophan": STRUCTURES["Tryptophane"],
    "UMP": "O=c1ccn(C2OC(COP(=O)(O)O)C(O)C2O)c(=O)[nH]1",
    "Hypoxanthine": "O=c1[nH]cnc2[nH]cnc12",
    "Adenine": "Nc1ncnc2[nH]cnc12",
    "Formate": "O=CO",
    "AMP": "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)O)C(O)C1O",
    "Nicotinate": "O=C(O)c1cccnc1",
    "Nicotinamide": "NC(=O)c1cccnc1",
}


def main():
    folder = ROOT / "cohorts" / "asf"
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "sources/asf_integrals_with_metadata.txt", sep="\t", dtype={"species":str})
    groups, target, raw, qc, panels = [], [], [], [], []
    for sample_species in sorted(set(df.species.dropna()) - {"0"}):
        ids = sample_species.split(",")
        if len(ids)<2:
            continue
        missing = [x for x in ids if x not in MODELS]
        profile = f"ASF_{'_'.join(ids)}"
        qc.append(dict(cohort=profile,members=sample_species,matched=len(ids)-len(missing),
                       total=len(ids),missing="|".join(missing),eligible=not missing))
        if missing:
            continue
        frame = df[df.species.eq(sample_species)]
        runs = sorted(frame.Run.unique())
        models = ["M_" + MODELS[x] for x in ids]
        groups.append(dict(study="Medlock2018", cohort=profile, models=models, baseline="medium_blank"))
        for label, smi in IDENTITIES.items():
            vals = []
            for run in runs:
                blank = df[(df.Run==run)&(df.species=="0")][["experiment_id",label]]
                spent = frame[frame.Run==run][["experiment_id",label]]
                for key, values in [("blank",blank),("spent",spent)]:
                    for sample, value in values.itertuples(index=False,name=None):
                        raw.append(dict(cohort=profile,run=run,analyte=label,group=key,sample=sample,value=value))
                x,y = blank[label].dropna().to_numpy(), spent[label].dropna().to_numpy()
                valid = len(x)==len(y)==3 and min(x.min(),y.min())>=0 and np.isfinite(np.r_[x,y]).all()
                folds = np.median(y)/np.median(x) if valid and np.median(x)>0 else np.nan
                above = int(sum(y>x.max())) if valid else None
                pos = bool(valid and above>=2 and (folds>=1.25 if np.median(x)>0 else np.median(y)>0))
                panels.append(dict(cohort=profile,run=run,analyte=label,valid=valid,enriched=pos,
                                   fold=folds,spent_above_blank=above,
                                   blank_median=np.median(x) if len(x) else None,spent_median=np.median(y) if len(y) else None))
                if valid:
                    vals.append(pos)
            required = len([r for r in runs if r!=3]) if len(runs)==1 else 2
            target.append(dict(study="Medlock2018",cohort=profile,analyte=label,smiles=smi,structure_key=structure_key(smi),
                               valid_runs=len(vals),supporting_runs=sum(vals),valid=len(vals)>=required,
                               enriched=len(vals)>=required and sum(vals)>=required))
    result = pd.DataFrame(target)
    assert result.structure_key.ne("").all()
    for name, records in [("targets",target),("sample_values",raw),("run_scores",panels),("cohort_eligibility",qc)]:
        pd.DataFrame(records).to_csv(folder/f"{name}.tsv",sep="\t",index=False)
    import json
    (folder/"cohorts.json").write_text(json.dumps(groups,indent=2),encoding="utf-8")
    print(result.groupby("cohort")[["valid","enriched"]].sum().to_string())


if __name__ == "__main__":
    main()
