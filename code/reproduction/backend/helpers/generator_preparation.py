from __future__ import annotations
import itertools
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from common import ensure_dir
from input_preparation import iter_table, read_data_file, table_path
try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
except Exception:
    Chem = None
    AllChem = None
    DataStructs = None

def _clean_token(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and math.isnan(value):
        return ''
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'null', 'na', 'n/a'}:
        return ''
    return text

def _side_parts(side: str) -> list[str]:
    return [part for part in str(side).split('.') if part.strip()]

def _lenient_mol_from_smiles(smiles: str) -> Any:
    if Chem is None:
        raise RuntimeError('RDKit is required for product construction')
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f'RDKit failed to parse SMILES: {smiles[:180]}')
    Chem.SanitizeMol(mol, catchErrors=True)
    return mol

def _rdkit_error_category(exc_or_text: Any) -> str:
    text = str(exc_or_text or '').lower()
    if 'kekul' in text:
        return 'kekulization'
    if 'valence' in text:
        return 'valence'
    if 'aromatic' in text:
        return 'aromaticity'
    if 'charge' in text:
        return 'charge'
    if 'sanitize' in text:
        return 'sanitize'
    if 'parse' in text or 'smiles' in text:
        return 'parse'
    if 'range' in text or 'atom' in text:
        return 'atom_index'
    return 'other'

def _ec_level_dict_from_text(ec_num: Any) -> dict[str, str]:
    text = _clean_token(ec_num)
    parts = text.split('.') if text else []
    return {'ec1': parts[0] if len(parts) >= 1 else text, 'ec2': '.'.join(parts[:2]) if len(parts) >= 2 else text, 'ec3': '.'.join(parts[:3]) if len(parts) >= 3 else text, 'ec4': '.'.join(parts[:4]) if len(parts) >= 4 else text}
_RULE_TEMPLATE_COMMON_SPECTATORS = {'[H+]', 'O', '[OH-]'}

def _canonical_side_from_mols(mols: list[Any] | tuple[Any, ...], isomeric: bool=False) -> dict[str, Any]:
    try:
        smiles: list[str] = []
        for mol in mols:
            if mol is None:
                continue
            rw = Chem.RWMol(mol)
            for atom in rw.GetAtoms():
                atom.SetAtomMapNum(0)
            mol2 = rw.GetMol()
            if mol2.GetNumAtoms() == 0:
                continue
            Chem.SanitizeMol(mol2)
            frags = Chem.GetMolFrags(mol2, asMols=True, sanitizeFrags=True)
            smiles.extend((Chem.MolToSmiles(frag, canonical=True, isomericSmiles=bool(isomeric)) for frag in frags if frag.GetNumAtoms() > 0))
        return {'ok': True, 'smiles': '.'.join(sorted(smiles)), 'n_fragments': int(len(smiles))}
    except Exception as exc:
        return {'ok': False, 'smiles': None, 'n_fragments': None, 'error': f'{type(exc).__name__}: {exc}', 'error_category': _rdkit_error_category(exc)}

def _strip_rule_template_spectators(smiles: Any) -> str | None:
    if smiles is None:
        return None
    parts = [part for part in str(smiles).split('.') if part and part not in _RULE_TEMPLATE_COMMON_SPECTATORS]
    return '.'.join(sorted(parts))

def _ec_rule_prior_keys(meta: dict[str, Any]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for level in ['ec4', 'ec3', 'ec2', 'ec1']:
        value = _clean_token(meta.get(level))
        if value:
            keys.append((level, value))
    keys.append(('global', '*'))
    return list(dict.fromkeys(keys))

def _load_rule_template_prior_and_eval_records(rule_chunks_dir: Path, split_assignments_path: Path, eval_splits: list[str], max_rows_per_split: int | None, seed: int) -> tuple[dict[tuple[str, str], Counter], dict[str, Counter], list[dict[str, Any]], dict[str, Any]]:
    split_df = read_data_file(split_assignments_path, columns=['reaction_record_id', 'split'])
    split_map = dict(zip(split_df['reaction_record_id'].astype(str), split_df['split'].astype(str)))
    eval_split_set = {str(x) for x in eval_splits}
    rule_prior: dict[tuple[str, str], Counter] = defaultdict(Counter)
    rule_smarts_by_id: dict[str, Counter] = defaultdict(Counter)
    eval_records_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = Counter()
    reaction_cols = ['reaction_record_id', 'mapped', 'unmapped', 'ec_num', 'ec1', 'ec2', 'ec3', 'ec4', 'rule', 'rule_directional_id', 'rule_family_id', 'rule_direction', 'source', 'steps', 'quality', 'n_reactants', 'n_products', 'n_bond_edits', 'n_changed_atoms']
    for _, df in iter_table(table_path(rule_chunks_dir, 'reaction_labels'), columns=reaction_cols, desc='load rule template labels'):
        for rec in df.to_dict(orient='records'):
            rid = str(rec.get('reaction_record_id'))
            split = split_map.get(rid)
            if not split:
                stats['missing_split'] += 1
                continue
            rec['split'] = str(split)
            rule_id = _clean_token(rec.get('rule_directional_id'))
            rule_smarts = _clean_token(rec.get('rule'))
            if split == 'train':
                stats['train_rows'] += 1
                if not rule_id or not rule_smarts:
                    stats['train_missing_rule'] += 1
                    continue
                rule_smarts_by_id[rule_id][rule_smarts] += 1
                for key in _ec_rule_prior_keys(rec):
                    rule_prior[key][rule_id] += 1
            elif split in eval_split_set:
                stats[f'{split}_rows_seen'] += 1
                eval_records_by_split[str(split)].append(rec)
    rng = random.Random(int(seed))
    eval_records: list[dict[str, Any]] = []
    for split in eval_splits:
        rows = list(eval_records_by_split.get(str(split), []))
        if max_rows_per_split is not None and len(rows) > int(max_rows_per_split):
            rng.shuffle(rows)
            rows = rows[:int(max_rows_per_split)]
        rows = sorted(rows, key=lambda x: str(x.get('reaction_record_id')))
        eval_records.extend(rows)
        stats[f'{split}_rows_selected'] = int(len(rows))
    context = {'split_assignments_rows': int(len(split_df)), 'train_rows': int(stats.get('train_rows', 0)), 'n_rule_prior_keys': int(len(rule_prior)), 'n_rule_ids_with_smarts': int(len(rule_smarts_by_id)), 'eval_rows_selected': int(len(eval_records)), 'rows_by_split': {str(k).replace('_rows_selected', ''): int(v) for k, v in stats.items() if str(k).endswith('_rows_selected')}, 'loader_stats': {str(k): int(v) for k, v in stats.items()}}
    return (rule_prior, rule_smarts_by_id, eval_records, context)

def _resolve_rule_template_candidates(rule_prior: dict[tuple[str, str], Counter], rule_smarts_by_id: dict[str, Counter], meta: dict[str, Any], topk: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for level, value in _ec_rule_prior_keys(meta):
        counter = rule_prior.get((level, value))
        if not counter:
            continue
        for rule_id, train_count in counter.most_common():
            if str(rule_id) in seen:
                continue
            smarts_counter = rule_smarts_by_id.get(str(rule_id))
            if not smarts_counter:
                continue
            rule_smarts, smarts_count = smarts_counter.most_common(1)[0]
            seen.add(str(rule_id))
            candidates.append({'rule_rank': int(len(candidates) + 1), 'rule_directional_id': str(rule_id), 'rule': str(rule_smarts), 'rule_source_key': f'{level}:{value}', 'rule_train_count': int(train_count), 'rule_smarts_count': int(smarts_count)})
            if len(candidates) >= int(topk):
                return candidates
    return candidates

def _run_rule_smarts_on_reactant_side(rule_smarts: str, reactant_side: str, rxn_cache: dict[str, Any], max_products: int, max_reactant_permutations: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats = Counter()
    products: list[dict[str, Any]] = []
    seen_products: set[str] = set()
    try:
        if AllChem is None:
            raise RuntimeError('RDKit AllChem is unavailable')
        if rule_smarts not in rxn_cache:
            rxn = AllChem.ReactionFromSmarts(str(rule_smarts))
            if rxn is None:
                raise ValueError('ReactionFromSmarts returned None')
            rxn.Initialize()
            rxn_cache[str(rule_smarts)] = rxn
        rxn = rxn_cache[str(rule_smarts)]
        required_reactants = int(rxn.GetNumReactantTemplates())
        reactant_mols = [_lenient_mol_from_smiles(part) for part in _side_parts(reactant_side)]
        stats['required_reactants'] = required_reactants
        stats['available_reactants'] = len(reactant_mols)
        if required_reactants <= 0:
            stats['no_reactant_templates'] += 1
            return (products, {str(k): int(v) for k, v in stats.items()})
        if required_reactants > len(reactant_mols):
            stats['insufficient_reactants'] += 1
            return (products, {str(k): int(v) for k, v in stats.items()})
        perm_iter = itertools.permutations(range(len(reactant_mols)), required_reactants)
        for perm_idx, perm in enumerate(perm_iter):
            if perm_idx >= int(max_reactant_permutations):
                stats['permutation_cap_hit'] += 1
                break
            stats['permutations_tried'] += 1
            try:
                product_sets = rxn.RunReactants(tuple((reactant_mols[i] for i in perm)))
            except Exception as exc:
                stats[f'run_error_{_rdkit_error_category(exc)}'] += 1
                continue
            stats['product_sets_returned'] += int(len(product_sets))
            for product_tuple in product_sets:
                full_noniso = _canonical_side_from_mols(product_tuple, isomeric=False)
                full_iso = _canonical_side_from_mols(product_tuple, isomeric=True)
                if not full_noniso.get('ok'):
                    stats[f"product_canonical_error_{full_noniso.get('error_category', 'other')}"] += 1
                    continue
                key = str(full_noniso.get('smiles') or '')
                if not key or key in seen_products:
                    continue
                seen_products.add(key)
                products.append({'full_nonisomeric': key, 'full_isomeric': full_iso.get('smiles') if full_iso.get('ok') else None, 'no_proton_water_nonisomeric': _strip_rule_template_spectators(key), 'n_product_fragments': int(full_noniso.get('n_fragments') or 0), 'reactant_permutation': list(map(int, perm))})
                if len(products) >= int(max_products):
                    stats['product_cap_hit'] += 1
                    return (products, {str(k): int(v) for k, v in stats.items()})
        return (products, {str(k): int(v) for k, v in stats.items()})
    except Exception as exc:
        stats[f'template_error_{_rdkit_error_category(exc)}'] += 1
        return (products, {**{str(k): int(v) for k, v in stats.items()}, 'error': f'{type(exc).__name__}: {exc}', 'error_category': _rdkit_error_category(exc)})

def _rule_source_depth(rule_source_key: Any) -> int:
    level = str(rule_source_key or '').split(':', 1)[0]
    return {'global': 0, 'ec1': 1, 'ec2': 2, 'ec3': 3, 'ec4': 4}.get(level, 0)

def _ecfp_bitstring_for_smiles(smiles: Any, n_bits: int=1024) -> tuple[str, bool]:
    if Chem is None or AllChem is None:
        return ('', False)
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return ('', False)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=int(n_bits))
        return (fp.ToBitString(), True)
    except Exception:
        return ('', False)

def _bitstring_to_float_array(bitstring: Any, n_bits: int) -> np.ndarray:
    text = str(bitstring or '')
    if len(text) < int(n_bits):
        text = text + '0' * (int(n_bits) - len(text))
    return np.fromiter((1.0 if ch == '1' else 0.0 for ch in text[:int(n_bits)]), dtype=np.float32, count=int(n_bits))

def _native_esmc_identifiers_for_model_path(model_path: Path, cfg: dict[str, Any]) -> list[str]:
    configured = cfg.get('native_esmc_identifiers')
    if configured:
        return [str(x) for x in configured]
    identifiers: list[str] = []
    if model_path.exists():
        identifiers.append(str(model_path))
    name = str(model_path.name).lower()
    if '600m' in name:
        identifiers.append('esmc_600m')
    elif '300m' in name:
        identifiers.append('esmc_300m')
    elif '6b' in name:
        identifiers.append('esmc_6b')
    else:
        identifiers.extend(['esmc_600m', 'esmc_300m'])
    return list(dict.fromkeys(identifiers))

def _pool_native_esmc_embeddings_1d(embeddings: Any, cfg: dict[str, Any]) -> np.ndarray:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and hasattr(embeddings, 'detach'):
        arr = embeddings.detach().float().cpu().numpy()
    else:
        arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f'Unexpected native ESMC embedding ndim: {arr.ndim}')
    if bool(cfg.get('native_esmc_drop_special_tokens', True)) and arr.shape[0] > 2:
        arr = arr[1:-1]
    return arr.mean(axis=0).astype(np.float32)

def _try_compute_native_esmc_protein_embeddings(protein_df: pd.DataFrame, model_path: Path, output_dir: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], Path | None]:
    report: dict[str, Any] = {'backend': 'native_esmc', 'model_path': str(model_path), 'model_path_exists': bool(model_path.exists()), 'attempted': False, 'status': 'not_attempted'}
    metadata_cols = ['protein_ref', 'embedding_index', 'embedding_dim', 'embedding_source', 'embedding_available']
    if protein_df.empty:
        return (pd.DataFrame(columns=metadata_cols), {**report, 'status': 'empty_input'}, None)
    if not bool(cfg.get('try_native_esmc', True)):
        return (pd.DataFrame(columns=metadata_cols), {**report, 'status': 'disabled_by_config'}, None)
    try:
        import torch
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
    except Exception as exc:
        return (pd.DataFrame(columns=metadata_cols), {**report, 'status': 'native_esmc_unavailable', 'error': repr(exc)}, None)
    report['attempted'] = True
    device = torch.device('cuda:0' if torch.cuda.is_available() and (not bool(cfg.get('force_cpu_embeddings', False))) else 'cpu')
    load_attempts: list[dict[str, Any]] = []
    client = None
    loaded_identifier = None
    for identifier in _native_esmc_identifiers_for_model_path(model_path, cfg):
        attempt = {'identifier': str(identifier)}
        try:
            client = ESMC.from_pretrained(str(identifier)).to(device)
            client.eval()
            attempt['status'] = 'ok'
            loaded_identifier = str(identifier)
            load_attempts.append(attempt)
            break
        except Exception as exc:
            attempt['status'] = 'failed'
            attempt['error'] = repr(exc)
            load_attempts.append(attempt)
    report['load_attempts'] = load_attempts
    if client is None:
        return (pd.DataFrame(columns=metadata_cols), {**report, 'status': 'native_esmc_load_failed'}, None)
    seq_rows = protein_df[['protein_ref', 'protein_sequence']].drop_duplicates('protein_ref').copy()
    seq_rows = seq_rows[seq_rows['protein_sequence'].fillna('').astype(str).str.len() > 0].reset_index(drop=True)
    max_length = int(cfg.get('esmc_max_length', 1024))
    embeddings: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for row in tqdm(seq_rows.itertuples(index=False), total=len(seq_rows), desc='native ESMC embeddings'):
                seq = re.sub('[^A-Za-z]', '', str(row.protein_sequence or '').upper())
                if max_length > 0:
                    seq = seq[:max_length]
                protein = ESMProtein(sequence=seq)
                protein_tensor = client.encode(protein)
                logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
                embedding = getattr(logits_output, 'embeddings', None)
                if embedding is None and isinstance(logits_output, dict):
                    embedding = logits_output.get('embeddings')
                if embedding is None:
                    raise RuntimeError('Native ESMC logits output has no embeddings')
                pooled = _pool_native_esmc_embeddings_1d(embedding, cfg)
                embeddings.append(pooled)
                meta_rows.append({'protein_ref': str(row.protein_ref), 'embedding_index': int(len(embeddings) - 1), 'embedding_dim': int(pooled.shape[0]), 'embedding_source': loaded_identifier or str(model_path), 'embedding_available': True})
    except Exception as exc:
        return (pd.DataFrame(columns=metadata_cols), {**report, 'status': 'native_esmc_embedding_failed', 'loaded_identifier': loaded_identifier, 'error': repr(exc)}, None)
    finally:
        try:
            del client
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            pass
    matrix = np.stack(embeddings, axis=0).astype(np.float32) if embeddings else np.zeros((0, 0), dtype=np.float32)
    ensure_dir(output_dir)
    embedding_path = output_dir / 'protein_native_esmc_embeddings.npy'
    np.save(embedding_path, matrix)
    report.update({'status': 'ok', 'device': str(device), 'loaded_identifier': loaded_identifier, 'n_embeddings': int(matrix.shape[0]), 'embedding_dim': int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[0] else 0, 'embedding_path': str(embedding_path)})
    return (pd.DataFrame(meta_rows), report, embedding_path)

def _try_compute_transformer_protein_embeddings(protein_df: pd.DataFrame, model_path: Path, output_dir: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], Path | None]:
    report: dict[str, Any] = {'backend': 'transformers_hf', 'model_path': str(model_path), 'model_path_exists': bool(model_path.exists()), 'attempted': False, 'status': 'not_attempted'}
    metadata_cols = ['protein_ref', 'embedding_index', 'embedding_dim', 'embedding_source', 'embedding_available']
    if protein_df.empty:
        return (pd.DataFrame(columns=metadata_cols), {**report, 'status': 'empty_input'}, None)
    if not bool(cfg.get('compute_esmc_embeddings', True)):
        return (pd.DataFrame(columns=metadata_cols), {**report, 'status': 'disabled_by_config'}, None)
    backend_mode = str(cfg.get('esmc_backend', 'native_first')).lower()
    if backend_mode in {'native', 'native_first', 'auto'}:
        native_df, native_report, native_path = _try_compute_native_esmc_protein_embeddings(protein_df, model_path, output_dir, cfg)
        if native_report.get('status') == 'ok':
            native_report['backend_mode'] = backend_mode
            return (native_df, native_report, native_path)
        if backend_mode == 'native':
            native_report['backend_mode'] = backend_mode
            return (native_df, native_report, native_path)
        report['native_esmc_fallback_report'] = native_report
    if not model_path.exists():
        return (pd.DataFrame(columns=metadata_cols), {**report, 'status': 'missing_model_path'}, None)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        return (pd.DataFrame(columns=metadata_cols), {**report, 'status': 'missing_transformers_or_torch', 'error': repr(exc)}, None)
    report['attempted'] = True
    device = torch.device('cuda:0' if torch.cuda.is_available() and (not bool(cfg.get('force_cpu_embeddings', False))) else 'cpu')
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        model_kwargs: dict[str, Any] = {'trust_remote_code': True, 'local_files_only': True}
        if not bool(cfg.get('transformer_add_pooling_layer', False)):
            model_kwargs['add_pooling_layer'] = False
        try:
            model = AutoModel.from_pretrained(str(model_path), **model_kwargs)
        except TypeError:
            model_kwargs.pop('add_pooling_layer', None)
            model = AutoModel.from_pretrained(str(model_path), **model_kwargs)
        model.eval()
        model.to(device)
    except Exception as exc:
        return (pd.DataFrame(columns=metadata_cols), {**report, 'status': 'load_failed', 'error': repr(exc)}, None)
    seq_rows = protein_df[['protein_ref', 'protein_sequence']].drop_duplicates('protein_ref').copy()
    seq_rows = seq_rows[seq_rows['protein_sequence'].fillna('').astype(str).str.len() > 0].reset_index(drop=True)
    batch_size = int(cfg.get('esmc_batch_size', 4))
    max_length = int(cfg.get('esmc_max_length', 1024))
    embeddings: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for start in tqdm(range(0, len(seq_rows), batch_size), desc='11 ESMC embeddings'):
                part = seq_rows.iloc[start:start + batch_size]
                encoded = tokenizer(part['protein_sequence'].astype(str).tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                output = model(**encoded)
                hidden = getattr(output, 'last_hidden_state', None)
                if hidden is None and isinstance(output, dict):
                    hidden = output.get('last_hidden_state')
                if hidden is None:
                    raise RuntimeError('ESMC/transformer output has no last_hidden_state')
                mask = encoded.get('attention_mask')
                if mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    pool_mask = mask
                    if bool(cfg.get('transformer_drop_special_tokens', True)):
                        input_ids = encoded.get('input_ids')
                        special_ids = set((int(x) for x in getattr(tokenizer, 'all_special_ids', []) if x is not None))
                        if input_ids is not None and special_ids:
                            special_mask = torch.zeros_like(pool_mask, dtype=torch.bool)
                            for special_id in special_ids:
                                special_mask = special_mask | input_ids.eq(int(special_id))
                            cleaned_mask = pool_mask.masked_fill(special_mask, 0)
                            empty_rows = cleaned_mask.sum(dim=1).eq(0)
                            if bool(empty_rows.any()):
                                cleaned_mask[empty_rows] = pool_mask[empty_rows]
                            pool_mask = cleaned_mask
                    mask_f = pool_mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
                pooled_np = pooled.detach().cpu().numpy().astype(np.float32)
                for idx, protein_ref in enumerate(part['protein_ref'].astype(str).tolist()):
                    embeddings.append(pooled_np[idx])
                    meta_rows.append({'protein_ref': protein_ref, 'embedding_index': int(len(embeddings) - 1), 'embedding_dim': int(pooled_np.shape[1]), 'embedding_source': str(model_path), 'embedding_available': True})
    except Exception as exc:
        return (pd.DataFrame(columns=metadata_cols), {**report, 'status': 'embedding_failed', 'error': repr(exc)}, None)
    if embeddings:
        matrix = np.stack(embeddings, axis=0).astype(np.float32)
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)
    ensure_dir(output_dir)
    embedding_path = output_dir / 'protein_esmc_embeddings.npy'
    np.save(embedding_path, matrix)
    report.update({'status': 'ok', 'device': str(device), 'n_embeddings': int(matrix.shape[0]), 'embedding_dim': int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[0] else 0, 'embedding_path': str(embedding_path)})
    return (pd.DataFrame(meta_rows), report, embedding_path)

def _try_official_xgboost_esp_scores(pair_df: pd.DataFrame, embedding_df: pd.DataFrame, embedding_path: Path | None, esp_root: Path, cfg: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    report: dict[str, Any] = {'attempted': False, 'status': 'not_attempted'}
    model_path = Path(cfg.get('esp_xgboost_model_path') or esp_root / 'data' / 'model_weights' / 'xgboost_model_production_mode.dat')
    report['model_path'] = str(model_path)
    report['model_path_exists'] = bool(model_path.exists())
    if not bool(cfg.get('try_official_esp_xgboost', True)):
        report['status'] = 'disabled_by_config'
        return (None, report)
    if pair_df.empty:
        report['status'] = 'empty_pair_df'
        return (None, report)
    if not model_path.exists():
        report['status'] = 'missing_model'
        return (None, report)
    try:
        import xgboost as xgb
    except Exception as exc:
        report['status'] = 'xgboost_unavailable'
        report['error'] = repr(exc)
        return (None, report)
    if embedding_path is None or embedding_df.empty:
        report['status'] = 'missing_embeddings'
        return (None, report)
    report['attempted'] = True
    booster, booster_report = _load_legacy_esp_booster(esp_root, cfg)
    report['booster_report'] = booster_report
    if booster is None:
        report['status'] = str(booster_report.get('status') or 'load_failed')
        report['error'] = booster_report.get('pickle_error') or booster_report.get('load_model_error') or booster_report.get('error')
        return (None, report)
    expected_features = int(booster_report.get('expected_features') or 0)
    matrix = np.load(embedding_path)
    emb_map = {str(row.protein_ref): int(row.embedding_index) for row in embedding_df.itertuples(index=False)}
    n_bits = int(cfg.get('substrate_ecfp_bits', 1024))
    feature_rows: list[np.ndarray] = []
    valid_indices: list[int] = []
    for idx, row in pair_df.reset_index(drop=True).iterrows():
        emb_idx = emb_map.get(str(row.get('protein_ref') or ''))
        if emb_idx is None or emb_idx >= len(matrix):
            continue
        ecfp = _bitstring_to_float_array(row.get('substrate_ecfp_bitstring'), n_bits=n_bits)
        feature_rows.append(np.concatenate([ecfp, matrix[emb_idx].astype(np.float32)], axis=0))
        valid_indices.append(int(idx))
    if not feature_rows:
        report['status'] = 'no_feature_rows'
        report['expected_features'] = expected_features
        return (None, report)
    X = np.stack(feature_rows, axis=0).astype(np.float32)
    report['expected_features'] = expected_features
    report['available_features'] = int(X.shape[1])
    report['n_feature_rows'] = int(X.shape[0])
    if expected_features and expected_features != int(X.shape[1]):
        report['status'] = 'feature_dimension_mismatch'
        return (None, report)
    try:
        feature_names = [f'ECFP_{i}' for i in range(n_bits)] + [f'ESM1b_ts_{i}' for i in range(X.shape[1] - n_bits)]
        if booster.__class__.__name__ == 'Booster':
            dmat = xgb.DMatrix(X, feature_names=feature_names if len(feature_names) == X.shape[1] else None)
            pred_part = booster.predict(dmat)
        elif hasattr(booster, 'predict_proba'):
            pred_part = booster.predict_proba(X)[:, 1]
        else:
            pred_part = booster.predict(X)
        pred = np.full((len(pair_df),), np.nan, dtype=np.float32)
        pred[np.array(valid_indices, dtype=int)] = np.asarray(pred_part, dtype=np.float32).reshape(-1)
    except Exception as exc:
        report['status'] = 'predict_failed'
        report['error'] = repr(exc)
        return (None, report)
    report['status'] = 'ok'
    report['booster_load_status'] = booster_report.get('status')
    report['predicted_rows'] = int(np.isfinite(pred).sum())
    return (pred, report)

def _path_head_text(path: Path, n: int=128) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='ignore')[:int(n)]
    except Exception:
        return ''

def _looks_like_lfs_pointer(path: Path) -> bool:
    return bool(path.exists() and path.is_file() and _path_head_text(path).startswith('version https://git-lfs.github.com/spec'))

def _load_legacy_esp_booster(esp_root: Path, cfg: dict[str, Any]) -> tuple[Any | None, dict[str, Any]]:
    model_path = Path(cfg.get('esp_xgboost_model_path') or esp_root / 'data' / 'model_weights' / 'xgboost_model_production_mode.dat')
    report: dict[str, Any] = {'model_path': str(model_path), 'model_path_exists': bool(model_path.exists()), 'status': 'not_attempted'}
    if not model_path.exists():
        report['status'] = 'missing_model'
        return (None, report)
    if _looks_like_lfs_pointer(model_path):
        report['status'] = 'git_lfs_pointer_only'
        report['size_bytes'] = int(model_path.stat().st_size)
        return (None, report)
    try:
        import xgboost as xgb
        report['xgboost_version'] = str(getattr(xgb, '__version__', 'available'))
    except Exception as exc:
        report['status'] = 'xgboost_unavailable'
        report['error'] = repr(exc)
        return (None, report)
    load_model_error = None
    try:
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        report['status'] = 'ok_load_model'
    except Exception as exc:
        booster = None
        load_model_error = repr(exc)
        report['load_model_error'] = load_model_error
    if booster is None:
        try:
            with open(model_path, 'rb') as handle:
                booster = pickle.load(handle)
            report['status'] = 'ok_legacy_pickle'
        except Exception as exc:
            report['status'] = 'legacy_pickle_load_failed'
            report['pickle_error'] = repr(exc)
            return (None, report)
    try:
        report['booster_class'] = booster.__class__.__name__
        report['expected_features'] = int(booster.num_features())
    except Exception as exc:
        report['num_features_error'] = repr(exc)
    try:
        feature_names = getattr(booster, 'feature_names', None)
        report['feature_names_count'] = int(len(feature_names)) if feature_names is not None else None
        report['feature_names_head'] = list(feature_names[:8]) if feature_names is not None else []
    except Exception:
        pass
    try:
        report['booster_attributes'] = {str(k): str(v) for k, v in booster.attributes().items()}
    except Exception:
        pass
    return (booster, report)
