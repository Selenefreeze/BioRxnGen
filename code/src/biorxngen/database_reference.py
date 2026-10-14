"""Frozen-database matching and conservative AGORA2 overlap checks."""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any
import pandas as pd
from rdkit import Chem

def rhea_canonical_smiles(value: Any, *, allow_generic: bool=False, isomeric: bool=False) -> tuple[str | None, int, str | None]:
    if Chem is None:
        return (None, 0, 'rdkit_unavailable')
    text = '' if value is None else str(value).strip()
    if not text:
        return (None, 0, 'empty')
    if '*' in text and (not allow_generic):
        return (None, 0, 'generic_wildcard')
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return (None, 0, 'parse_failed')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return (None, 0, f'sanitize_failed:{type(exc).__name__}')
    return (Chem.MolToSmiles(mol, isomericSmiles=isomeric), int(mol.GetNumHeavyAtoms()), None)

@lru_cache(maxsize=1)
def currency_smiles() -> frozenset[str]:
    raw = ['[H+]', '[H-]', '[OH-]', '[H]O[H]', 'O', 'O=O', 'N', '[NH4+]', 'O=C=O', 'O=P([O-])([O-])O', 'O=P([O-])([O-])[O-]', 'O=P(O)(O)O', 'O=S(=O)([O-])[O-]', 'O=C([O-])[O-]']
    out: set[str] = set()
    for item in raw:
        cleaned, heavy, err = rhea_canonical_smiles(item)
        if not err and cleaned and (heavy >= 0):
            out.add(cleaned)
    return frozenset(out)

def canonical_fragments(value: Any, *, drop_currency: bool=False, allow_generic: bool=False) -> tuple[list[str], bool]:
    text = '' if value is None else str(value).strip()
    if not text:
        return ([], False)
    currencies = currency_smiles() if drop_currency else set()
    frags: list[str] = []
    generic_seen = False
    for raw in text.split('.'):
        cleaned, heavy, err = rhea_canonical_smiles(raw, allow_generic=allow_generic)
        if err:
            if 'generic' in err:
                generic_seen = True
            continue
        if not cleaned or heavy <= 0:
            continue
        if cleaned in currencies:
            continue
        frags.append(cleaned)
    return (sorted(frags), generic_seen)

def rhea_canonical_side(value: Any, *, drop_currency: bool=False, allow_generic: bool=False) -> str | None:
    frags, _generic = canonical_fragments(value, drop_currency=drop_currency, allow_generic=allow_generic)
    if not frags:
        return None
    return '.'.join(frags)

def reaction_sides(reaction_smiles: Any) -> tuple[str, str] | None:
    text = '' if reaction_smiles is None else str(reaction_smiles).strip()
    if '>>' not in text:
        return None
    left, right = text.split('>>', 1)
    return (left, right)

def load_rhea_index(cache_dir: Path) -> dict[str, Any]:
    reaction_path = cache_dir / 'rhea-reaction-smiles.tsv'
    directions_path = cache_dir / 'rhea-directions.tsv'
    rhea2ec_path = cache_dir / 'rhea2ec.tsv'
    if not reaction_path.exists():
        raise FileNotFoundError(reaction_path)
    if not directions_path.exists():
        raise FileNotFoundError(directions_path)
    reaction_df = pd.read_csv(reaction_path, sep='\t', header=None, names=['rhea_id', 'reaction_smiles'])
    reaction_by_id: dict[int, dict[str, Any]] = {}
    for row in reaction_df.to_dict('records'):
        rid = int(row['rhea_id'])
        sides = reaction_sides(row.get('reaction_smiles'))
        if not sides:
            continue
        left_raw, right_raw = sides
        left_full = rhea_canonical_side(left_raw, drop_currency=False)
        right_full = rhea_canonical_side(right_raw, drop_currency=False)
        left_clean = rhea_canonical_side(left_raw, drop_currency=True)
        right_clean = rhea_canonical_side(right_raw, drop_currency=True)
        left_frags, left_generic = canonical_fragments(left_raw, drop_currency=True)
        right_frags, right_generic = canonical_fragments(right_raw, drop_currency=True)
        reaction_by_id[rid] = {'rhea_id': rid, 'reaction_smiles': row.get('reaction_smiles'), 'left_raw': left_raw, 'right_raw': right_raw, 'left_full_clean': left_full, 'right_full_clean': right_full, 'left_noncurrency_clean': left_clean, 'right_noncurrency_clean': right_clean, 'left_noncurrency_fragments': set(left_frags), 'right_noncurrency_fragments': set(right_frags), 'has_generic_wildcard': bool(left_generic or right_generic or '*' in str(row.get('reaction_smiles')))}
    directions_df = pd.read_csv(directions_path, sep='\t')
    family_by_any_id: dict[int, dict[str, int | None]] = {}
    master_by_any_id: dict[int, int] = {}
    for rec in directions_df.to_dict('records'):
        family = {'master': int(rec['RHEA_ID_MASTER']), 'lr': int(rec['RHEA_ID_LR']), 'rl': int(rec['RHEA_ID_RL']), 'bi': int(rec['RHEA_ID_BI'])}
        for rid in family.values():
            family_by_any_id[int(rid)] = family
            master_by_any_id[int(rid)] = int(family['master'])
    ec_by_master: dict[int, set[str]] = defaultdict(set)
    ec_by_any_id: dict[int, set[str]] = defaultdict(set)
    if rhea2ec_path.exists():
        rhea2ec = pd.read_csv(rhea2ec_path, sep='\t')
        for rec in rhea2ec.to_dict('records'):
            ec = str(rec.get('ID') or '').strip()
            if not ec:
                continue
            rid = int(rec['RHEA_ID'])
            master = int(rec.get('MASTER_ID') or master_by_any_id.get(rid, rid))
            ec_by_master[master].add(ec)
            ec_by_any_id[rid].add(ec)
    xrefs_by_master: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for key, filename in [('metacyc', 'rhea2metacyc.tsv'), ('kegg_reaction', 'rhea2kegg_reaction.tsv')]:
        path = cache_dir / filename
        if not path.exists():
            continue
        xdf = pd.read_csv(path, sep='\t')
        for rec in xdf.to_dict('records'):
            rid = int(rec['RHEA_ID'])
            master = int(rec.get('MASTER_ID') or master_by_any_id.get(rid, rid))
            value = str(rec.get('ID') or '').strip()
            if value:
                xrefs_by_master[master][key].add(value)
    return {'reaction_by_id': reaction_by_id, 'family_by_any_id': family_by_any_id, 'master_by_any_id': master_by_any_id, 'ec_by_master': ec_by_master, 'ec_by_any_id': ec_by_any_id, 'xrefs_by_master': xrefs_by_master}
MANUAL_AGORA_BLOCKS = [{'label': 'AGORA2_R_GLYD_glycerate_dehydrogenase', 'reaction_ids': ['R_GLYD', 'R_GLYDy'], 'reaction_name_tokens': ['glycerate dehydrogenase'], 'mnx_reaction_ids': ['MNXR2769'], 'seed_reaction_ids': ['rxn01011'], 'metacyc_ids': ['GLYCERATE-DEHYDROGENASE-RXN', 'HYDROXYPYRUVATE-REDUCTASE-RXN', 'RXN0-300'], 'kegg_reaction_ids': ['R01388', 'R01392'], 'substrate_smiles': ['O=C([O-])C(O)CO'], 'product_smiles': ['O=C([O-])C(=O)CO']}]

def split_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        raw = value
    else:
        text = str(value).strip()
        if not text or text.lower() == 'nan':
            return []
        try:
            loaded = json.loads(text)
            raw = loaded if isinstance(loaded, list) else [loaded]
        except Exception:
            raw = re.split('[|,;]\\s*', text)
    out: list[str] = []
    for item in raw:
        token = str(item).strip().strip('"').strip("'")
        if token and token.lower() != 'nan':
            out.append(token)
    return out

def norm_token(value: Any) -> str:
    return str(value or '').strip().upper()

def canonical_smiles(value: Any) -> str | None:
    text = '' if value is None else str(value).strip()
    if not text:
        return None
    if Chem is None:
        return text
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def canonical_side_fragments(value: Any) -> set[str]:
    text = '' if value is None else str(value).strip()
    out: set[str] = set()
    for frag in text.split('.'):
        cleaned = canonical_smiles(frag)
        if cleaned:
            out.add(cleaned)
    return out

def manual_block_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    metacyc = {norm_token(x) for x in split_tokens(row.get('rhea_metacyc_ids'))}
    kegg = {norm_token(x) for x in split_tokens(row.get('rhea_kegg_reaction_ids'))}
    side_fragments = canonical_side_fragments(row.get('rhea_reactant_side_noncurrency')) | canonical_side_fragments(row.get('rhea_product_side_noncurrency'))
    generated_fragments = canonical_side_fragments(row.get('generated_product_side_noncurrency'))
    side_fragments |= generated_fragments
    for block in MANUAL_AGORA_BLOCKS:
        block_metacyc = {norm_token(x) for x in block.get('metacyc_ids', [])}
        block_kegg = {norm_token(x) for x in block.get('kegg_reaction_ids', [])}
        if metacyc & block_metacyc:
            reasons.append(f"manual_block:{block['label']}:metacyc={','.join(sorted(metacyc & block_metacyc))}")
        if kegg & block_kegg:
            reasons.append(f"manual_block:{block['label']}:kegg={','.join(sorted(kegg & block_kegg))}")
        substrates = {canonical_smiles(x) for x in block.get('substrate_smiles', [])}
        products = {canonical_smiles(x) for x in block.get('product_smiles', [])}
        substrates.discard(None)
        products.discard(None)
        if substrates and products and side_fragments & substrates and side_fragments & products:
            reasons.append(f"manual_block:{block['label']}:substrate_product_pair")
    return sorted(set(reasons))

def reference_overlap_reasons(row: dict[str, Any], ref: dict[str, set[str]]) -> list[str]:
    reasons = manual_block_reasons(row)
    metacyc = {norm_token(x) for x in split_tokens(row.get('rhea_metacyc_ids'))}
    kegg = {norm_token(x) for x in split_tokens(row.get('rhea_kegg_reaction_ids'))}
    if metacyc and metacyc & ref['metacyc_ids']:
        reasons.append(f"agora_reaction_xref_overlap:metacyc={','.join(sorted(metacyc & ref['metacyc_ids']))}")
    if kegg and kegg & ref['kegg_reaction_ids']:
        reasons.append(f"agora_reaction_xref_overlap:kegg={','.join(sorted(kegg & ref['kegg_reaction_ids']))}")
    rhea_master = norm_token(row.get('rhea_master_id'))
    directed = norm_token(row.get('directed_rhea_id'))
    if rhea_master in ref['reaction_ids']:
        reasons.append(f'agora_reaction_id_overlap:{rhea_master}')
    if directed in ref['reaction_ids']:
        reasons.append(f'agora_reaction_id_overlap:{directed}')
    return sorted(set(reasons))

def build_rhea_signature_index(rhea_index: dict[str, Any]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    records: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    reaction_by_id = rhea_index.get('reaction_by_id', {})
    ec_by_master = rhea_index.get('ec_by_master', {})
    ec_by_any_id = rhea_index.get('ec_by_any_id', {})
    master_by_any_id = rhea_index.get('master_by_any_id', {})
    xrefs_by_master = rhea_index.get('xrefs_by_master', {})
    for rid, reaction in reaction_by_id.items():
        rid = int(rid)
        master = int(master_by_any_id.get(rid, rid))
        ecs = set(ec_by_master.get(master, set())) | set(ec_by_any_id.get(rid, set()))
        if not ecs:
            continue
        xrefs = xrefs_by_master.get(master, {})
        orientations = [('left_to_right', reaction.get('left_noncurrency_clean'), reaction.get('right_noncurrency_clean')), ('right_to_left', reaction.get('right_noncurrency_clean'), reaction.get('left_noncurrency_clean'))]
        for ec in sorted(ecs):
            for orientation, reactant_side, product_side in orientations:
                reactant_side = as_text(reactant_side)
                product_side = as_text(product_side)
                if not reactant_side or not product_side:
                    continue
                rec = {'rhea_id': f'RHEA:{rid}', 'rhea_master_id': f'RHEA:{master}', 'rhea_orientation': orientation, 'rhea_reactant_side_noncurrency': reactant_side, 'rhea_product_side_noncurrency': product_side, 'rhea_metacyc_ids': '|'.join(sorted(xrefs.get('metacyc', set()))), 'rhea_kegg_reaction_ids': '|'.join(sorted(xrefs.get('kegg_reaction', set()))), 'rhea_reaction_smiles': reaction.get('reaction_smiles')}
                records.setdefault((str(ec), reactant_side, product_side), []).append(rec)
    return records

def as_text(value):
    return '' if value is None or pd.isna(value) else str(value).strip()


def load_agora_overlap_reference(root: Path) -> dict[str, set[str]]:
    reference = {name: set() for name in ('reaction_ids', 'metacyc_ids', 'kegg_reaction_ids')}
    reaction_paths = sorted((root / 'processed_reconstructions/reactions').glob('*.parquet'))
    xref_paths = sorted((root / 'processed_reconstructions/reaction_xrefs').glob('*.parquet'))
    if not reaction_paths or not xref_paths:
        raise FileNotFoundError('Processed AGORA2 reaction and cross-reference partitions are required')
    for path in reaction_paths:
        rows = pd.read_parquet(path, columns=['reaction_id'])
        reference['reaction_ids'].update(norm_token(value) for value in rows.reaction_id.dropna().unique())
    for path in xref_paths:
        rows = pd.read_parquet(path, columns=['db', 'accession']).drop_duplicates()
        for database, field in [('metacyc', 'metacyc_ids'), ('kegg', 'kegg_reaction_ids')]:
            reference[field].update(norm_token(value) for value in rows.loc[rows.db.str.lower().str.contains(database, na=False), 'accession'].dropna().unique())
    for block in MANUAL_AGORA_BLOCKS:
        for field in reference:
            reference[field].update(norm_token(value) for value in block.get(field, []))
    return reference


def validate_signature_table(signatures: pd.DataFrame, rhea_cache: Path, reference: dict[str, set[str]]) -> pd.DataFrame:
    required = ['rhea-reaction-smiles.tsv', 'rhea-directions.tsv', 'rhea2ec.tsv', 'rhea2metacyc.tsv', 'rhea2kegg_reaction.tsv']
    for name in required:
        if not (rhea_cache / name).is_file():
            raise FileNotFoundError(f'Frozen Rhea cache is incomplete: {name}')
    index = build_rhea_signature_index(load_rhea_index(rhea_cache))
    output = []
    for row in signatures.to_dict('records'):
        sides = reaction_sides(row['reaction_smiles'])
        if sides is None:
            raise ValueError(f"Invalid reaction representation: {row['signature_id']}")
        left = rhea_canonical_side(sides[0], drop_currency=True) or sides[0]
        right_input = row.get('noncurrency_product_smiles') or row.get('product_smiles') or sides[1]
        right = rhea_canonical_side(right_input, drop_currency=True) or right_input
        matches = index.get((as_text(row['ec_number']), left, right), [])
        best = matches[0] if matches else {}
        overlap = reference_overlap_reasons({
            **best, 'directed_rhea_id': best.get('rhea_id'), 'rhea_reactant_side_noncurrency': left,
            'rhea_product_side_noncurrency': right, 'generated_product_side_noncurrency': right,
        }, reference) if matches else []
        secondary = bool(best.get('rhea_metacyc_ids') or best.get('rhea_kegg_reaction_ids'))
        tier = 'no_exact_rhea_match'
        if matches:
            tier = 'agora_overlap' if overlap else 'rhea_exact_secondary_non_agora' if secondary else 'rhea_exact_non_agora'
        output.append({'signature_id': row['signature_id'], 'validation_tier': tier,
            'exact_rhea_reaction_match': bool(matches), 'n_exact_rhea_reaction_matches': len(matches),
            **{name: best.get(name, '') for name in ('rhea_id', 'rhea_master_id', 'rhea_orientation', 'rhea_metacyc_ids', 'rhea_kegg_reaction_ids')},
            'agora_overlap_reasons': ';'.join(overlap), 'has_agora_overlap_signal': bool(overlap)})
    return pd.DataFrame(output)
