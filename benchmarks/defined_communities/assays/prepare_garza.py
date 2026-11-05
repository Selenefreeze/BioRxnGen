"""Construct paired 0-to-120 h panels from the authors' batch HPLC data."""
import json
import math

import pandas as pd
from prepare_asf import ROOT, IDENTITIES, structure_key

MODELS = {"bh":"M_Blautia_hydrogenotrophica_DSM_10507",
          "bt":"M_Bacteroides_thetaiotaomicron_VPI_5482",
          "ri":"M_Roseburia_intestinalis_L1_82"}
MEMBERS = {"bhbt":["bh","bt"],"bhri":["bh","ri"],"btri":["bt","ri"],"bhbtri":["bh","bt","ri"]}
STRUCTURES = {name.lower():IDENTITIES[name] for name in ("Acetate","Butyrate","Formate","Lactate","Succinate","Trehalose")}
STRUCTURES.update(glucose="OCC1OC(O)C(O)C(O)C1O",pyruvate="CC(=O)C(=O)O")


if __name__=="__main__":
    folder=ROOT/"cohorts/garza"
    folder.mkdir(parents=True,exist_ok=True)
    targets, groups, records, paired = [], [], [], []
    for name,members in MEMBERS.items():
        cohort="GARZA_"+name.upper()
        groups.append(dict(study="RiosGarza2026",cohort=cohort,models=[MODELS[m] for m in members],
                           baseline="early_culture",minimum_valid_pairs=3))
        for file in sorted((ROOT/"sources/garza"/name).glob("*.tsv")):
            label=file.stem
            smi=STRUCTURES[label]
            source=pd.read_csv(file,sep="\t")
            for row in source.itertuples(index=False):
                records.append(dict(cohort=cohort,analyte=label,replicate=row.replicate,
                                    time_h=row.time,value=getattr(row,label),source=file.relative_to(ROOT).as_posix()))
            frame=source.pivot(index="replicate",columns="time",values=label)
            obs=frame.reindex(columns=[0,120]).dropna()
            valid=obs[(obs[0]>=0)&(obs[120]>=0)]
            pos=valid[120].gt(valid[0])&valid[120].ge(1.25*valid[0])
            for sample,row in valid.iterrows():
                paired.append(dict(cohort=cohort,analyte=label,replicate=sample,
                                   initial=row[0],endpoint=row[120],enriched=bool(pos.loc[sample])))
            n,k=len(valid),int(pos.sum())
            targets.append(dict(study="RiosGarza2026",cohort=cohort,analyte=label,smiles=smi,
                                structure_key=structure_key(smi),valid_runs=n,supporting_runs=k,
                                valid=n>=3,enriched=n>=3 and k>=math.ceil(2*n/3)))
    df=pd.DataFrame(targets)
    assert df.structure_key.ne("").all()
    for name,rows in [("targets",targets),("sample_values",records),("pair_scores",paired)]:
        pd.DataFrame(rows).to_csv(folder/f"{name}.tsv",sep="\t",index=False)
    (folder/"cohorts.json").write_text(json.dumps(groups,indent=2),encoding="utf-8")
    print(df.groupby("cohort")[["valid","enriched"]].sum().to_string())
