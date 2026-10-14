from __future__ import annotations

from functools import lru_cache
from typing import Iterable


SPECTATOR_COMPONENTS = {"[H+]", "[H]", "O", "[OH-]"}


def _rdkit_chem():
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise ImportError("RDKit is required for chemistry-dependent functions.") from exc
    return Chem


def split_components(smiles: str | None) -> list[str]:
    if smiles is None:
        return []
    return [part for part in str(smiles).split(".") if part]


@lru_cache(maxsize=200000)
def canonicalize_smiles(smiles: str | None, isomeric: bool = True) -> str:
    """Return a canonical SMILES string, or an empty string if parsing fails."""
    if smiles is None or str(smiles).strip() == "":
        return ""
    Chem = _rdkit_chem()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def canonical_component_set(smiles: str | None, remove_spectators: bool = False) -> tuple[str, ...]:
    parts = []
    for part in split_components(smiles):
        can = canonicalize_smiles(part)
        if not can:
            continue
        if remove_spectators and can in SPECTATOR_COMPONENTS:
            continue
        parts.append(can)
    return tuple(sorted(parts))


def product_identity_key(smiles: str | None, remove_spectators: bool = True) -> str:
    return ".".join(canonical_component_set(smiles, remove_spectators=remove_spectators))


def molecular_formula_set(smiles: str | None, remove_spectators: bool = True) -> str:
    Chem = _rdkit_chem()
    from rdkit.Chem import rdMolDescriptors

    formulas: list[str] = []
    for part in canonical_component_set(smiles, remove_spectators=remove_spectators):
        mol = Chem.MolFromSmiles(part)
        if mol is not None:
            formulas.append(rdMolDescriptors.CalcMolFormula(mol))
    return ".".join(sorted(formulas))


def morgan_tanimoto(smiles_a: str | None, smiles_b: str | None, radius: int = 2, n_bits: int = 2048) -> float | None:
    Chem = _rdkit_chem()
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    can_a = canonicalize_smiles(smiles_a)
    can_b = canonicalize_smiles(smiles_b)
    if not can_a or not can_b:
        return None
    mol_a = Chem.MolFromSmiles(can_a)
    mol_b = Chem.MolFromSmiles(can_b)
    if mol_a is None or mol_b is None:
        return None
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=n_bits)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=n_bits)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def first_valid_smiles(values: Iterable[str | None]) -> str:
    for value in values:
        can = canonicalize_smiles(value)
        if can:
            return can
    return ""
