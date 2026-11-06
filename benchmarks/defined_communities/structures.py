"""Standardize covalent structures and audited assay representations."""
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

DATA = Path(__file__).resolve().parent / "data"
RDLogger.DisableLog("rdApp.*")
UNCHARGER = rdMolStandardize.Uncharger()
AMIDES = {
    "N=C(O)CC(N)C(=O)O": "NC(=O)CC(N)C(=O)O",
    "N=C(O)CCC(N)C(=O)O": "NC(=O)CCC(N)C(=O)O",
}


@lru_cache(maxsize=200000)
def structure_key(smiles):
    if not isinstance(smiles, str) or not smiles or "." in smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms() or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return ""
    key = Chem.MolToSmiles(UNCHARGER.uncharge(mol), isomericSmiles=False)
    return AMIDES.get(key, key)


@lru_cache(maxsize=1)
def crosswalk():
    return pd.read_csv(DATA / "assay_structure_crosswalk.tsv", sep="\t").fillna("")


@lru_cache(maxsize=1)
def aliases():
    return {structure_key(r.archive_smiles): structure_key(r.assay_smiles)
            for r in crosswalk().itertuples(index=False) if r.archive_smiles}


def side_keys(side, label):
    mapping = {} if label == "omm12" else aliases()
    return {mapping.get(key, key) for token in side.split(".") if (key := structure_key(token))}


def native_structure_keys(hints, label):
    keys = hints.smiles_from_inchi.map(structure_key)
    base = pd.read_csv(DATA / "structure_crosswalk.tsv", sep="\t").rename(columns={
        "metabolite_base": "base", "kegg_id": "kegg", "chebi_id": "chebi", "neutral_smiles": "assay_smiles",
    })
    base["archive_smiles"] = ""
    entries = base if label == "omm12" else pd.concat([base, crosswalk()])
    for row in entries.itertuples(index=False):
        selected = (
            hints.metabolite_id.str.startswith(row.base + "__91__")
            & hints.kegg_compound_list_json.map(lambda x: row.kegg in json.loads(x))
            & hints.chebi_list_json.map(lambda x: row.chebi in json.loads(x))
        )
        target = structure_key(row.assay_smiles)
        allowed = {"", target, structure_key(row.archive_smiles)}
        if not set(keys[selected]) <= allowed:
            raise ValueError(f"Unexpected structure for {row.base}")
        keys.loc[selected] = target
    return keys


@lru_cache(maxsize=65536)
def composition(smiles):
    elements = Counter()
    charge = 0
    for part in smiles.split("."):
        mol = Chem.MolFromSmiles(part)
        if mol is None or mol.GetNumAtoms() == 0 or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            return None
        elements.update(atom.GetAtomicNum() for atom in Chem.AddHs(mol).GetAtoms())
        charge += Chem.GetFormalCharge(mol)
    heavy = tuple(sorted((element, count) for element, count in elements.items() if element != 1))
    return heavy, elements[1], charge
