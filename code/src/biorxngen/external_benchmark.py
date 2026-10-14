from __future__ import annotations
import math
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
TOPKS = [1, 5, 10, 20, 50, 100]
POLICY_ORDER = ['hybrid_reranker', 'template_rank']
METRIC_ORDER = ['string_full', 'string_no_proton_water', 'rdkit_component_no_hw', 'neutral_component_no_hw', 'neutral_formula_no_hw']
METRIC_LABELS = {'string_full': 'Full product string identity', 'string_no_proton_water': 'No H+/H2O product string identity', 'rdkit_component_no_hw': 'Canonical component identity', 'neutral_component_no_hw': 'Charge-normalized component identity', 'neutral_formula_no_hw': 'Charge-normalized formula-equivalent upper bound'}

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')

def clean(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and math.isnan(value):
        return ''
    text = str(value).strip()
    return '' if text.lower() in {'', 'nan', 'none', 'null', '<na>', 'nat'} else text

def to_int(value: Any, default: int=0) -> int:
    text = clean(value)
    if not text:
        return default
    try:
        return int(float(text))
    except Exception:
        return default

def truthy(value: Any) -> bool:
    return clean(value).lower() in {'true', '1', 'yes', 'y'}

def load_pandas_and_rdkit():
    import pandas as pd
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    uncharger = rdMolStandardize.Uncharger()
    return (pd, Chem, rdMolDescriptors, uncharger)

def split_components(smiles: str) -> list[str]:
    text = clean(smiles)
    if not text:
        return []
    return [part for part in text.split('.') if clean(part)]

def build_remove_component_keys(Chem: Any, rdMolDescriptors: Any) -> tuple[set[str], set[str]]:
    remove_smiles: set[str] = set()
    remove_formula = {'H', 'H2', 'H2O', 'HO', 'OH'}
    for smi in ['[H+]', '[H]', '[H-]', 'O', '[OH2]', '[OH-]']:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        remove_smiles.add(Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True))
        try:
            remove_formula.add(rdMolDescriptors.CalcMolFormula(mol))
        except Exception:
            pass
    return (remove_smiles, remove_formula)

class RepresentationKeyer:

    def __init__(self, Chem: Any, rdMolDescriptors: Any, uncharger: Any):
        self.Chem = Chem
        self.rdMolDescriptors = rdMolDescriptors
        self.uncharger = uncharger
        self.remove_smiles, self.remove_formula = build_remove_component_keys(Chem, rdMolDescriptors)
        self.invalid_counter: Counter[str] = Counter()

    @lru_cache(maxsize=300000)
    def component_key(self, smiles: str, metric: str) -> str:
        text = clean(smiles)
        if not text:
            return ''
        mol = self.Chem.MolFromSmiles(text)
        if mol is None:
            self.invalid_counter[f'{metric}:parse_failed'] += 1
            return f'INVALID:{text}'
        try:
            if metric in {'neutral_component_no_hw', 'neutral_formula_no_hw'}:
                mol = self.uncharger.uncharge(mol)
            if metric == 'neutral_component_no_hw':
                return self.Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            if metric == 'neutral_formula_no_hw':
                return self.rdMolDescriptors.CalcMolFormula(mol)
            return self.Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        except Exception:
            self.invalid_counter[f'{metric}:key_failed'] += 1
            return f'INVALID:{text}'

    @lru_cache(maxsize=300000)
    def side_key(self, smiles: str, metric: str) -> str:
        if metric in {'string_full', 'string_no_proton_water'}:
            return clean(smiles)
        components: list[str] = []
        invalids: list[str] = []
        for comp in split_components(smiles):
            key = self.component_key(comp, metric)
            if not key:
                continue
            if metric == 'neutral_formula_no_hw':
                if key in self.remove_formula:
                    continue
            elif key in self.remove_smiles:
                continue
            if key.startswith('INVALID:'):
                invalids.append(key)
            else:
                components.append(key)
        if components:
            return '||'.join(sorted(components))
        if invalids:
            return 'INVALID_SIDE:' + '||'.join(sorted(invalids))
        return ''

    def component_set(self, smiles: str, metric: str) -> set[str]:
        key = self.side_key(smiles, metric)
        if not key:
            return set()
        if key.startswith('INVALID_SIDE:'):
            return set()
        return set(key.split('||'))

def ec_class_from_row(row: dict[str, Any]) -> str:
    for field in ['source_ec_num', 'ec_ids_json', 'ec_num']:
        text = clean(row.get(field))
        if not text:
            continue
        match = re.search('([1-7])(?:\\.|$)', text)
        if match:
            return f'EC{match.group(1)}'
    return 'unclassified'

def support_bin(value: Any) -> str:
    n = to_int(value)
    if n < 10:
        return '1-9'
    if n < 50:
        return '10-49'
    if n < 100:
        return '50-99'
    if n < 200:
        return '100-199'
    if n < 500:
        return '200-499'
    return '>=500'

def build_benchmark_metadata(reaction_df: Any) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in reaction_df.to_dict(orient='records'):
        bid = clean(row.get('benchmark_id'))
        if not bid:
            continue
        if bid not in meta:
            ec_class = ec_class_from_row(row)
            meta[bid] = {'benchmark_id': bid, 'reaction_id': clean(row.get('reaction_id')) or bid, 'reaction_name': clean(row.get('reaction_name')), 'ec_class': ec_class, 'ec_ids_json': clean(row.get('ec_ids_json')), 'external_support_model_count_total': to_int(row.get('external_support_model_count_total')), 'support_bin': support_bin(row.get('external_support_model_count_total')), 'candidate_rows': to_int(row.get('candidate_rows')), 'candidate_positive': truthy(row.get('candidate_positive')), 'target_product_full_nonisomeric': clean(row.get('target_product_full_nonisomeric')), 'target_product_no_proton_water_nonisomeric': clean(row.get('target_product_no_proton_water_nonisomeric')), 'reactant_base_ids': clean(row.get('reactant_base_ids')), 'product_base_ids': clean(row.get('product_base_ids')), 'signature_noncurrency': clean(row.get('signature_noncurrency'))}
    return meta

def add_metric_keys(prediction_df: Any, keyer: RepresentationKeyer) -> Any:
    df = prediction_df.copy()
    metric_specs = {'string_full': ('predicted_product_smiles', 'target_product_smiles'), 'string_no_proton_water': ('predicted_product_no_proton_water_smiles', 'target_product_no_proton_water_smiles'), 'rdkit_component_no_hw': ('predicted_product_smiles', 'target_product_smiles'), 'neutral_component_no_hw': ('predicted_product_smiles', 'target_product_smiles'), 'neutral_formula_no_hw': ('predicted_product_smiles', 'target_product_smiles')}
    for metric, (pred_col, target_col) in metric_specs.items():
        print(f'[{now_iso()}] representation metric: {metric}', flush=True)
        df[f'{metric}__pred_key'] = df[pred_col].map(lambda value, m=metric: keyer.side_key(value, m))
        df[f'{metric}__target_key'] = df[target_col].map(lambda value, m=metric: keyer.side_key(value, m))
        df[f'{metric}__match'] = df[f'{metric}__pred_key'].eq(df[f'{metric}__target_key']) & df[f'{metric}__target_key'].astype(bool)
    return df

def compute_rank_maps(scored_df: Any) -> dict[tuple[str, str], dict[str, int]]:
    rank_maps: dict[tuple[str, str], dict[str, int]] = {}
    for policy in POLICY_ORDER:
        part = scored_df[scored_df['policy'] == policy]
        if part.empty:
            continue
        for metric in METRIC_ORDER:
            hit_col = f'{metric}__match'
            hits = part[part[hit_col]][['benchmark_id', 'rank']]
            if hits.empty:
                rank_maps[policy, metric] = {}
                continue
            best = hits.groupby('benchmark_id', as_index=False)['rank'].min()
            rank_maps[policy, metric] = {str(row['benchmark_id']): int(row['rank']) for row in best.to_dict(orient='records')}
    return rank_maps

def build_topk_summary(rank_maps: dict[tuple[str, str], dict[str, int]], benchmark_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    benchmark_ids = sorted(benchmark_meta)
    all_n = len(benchmark_ids)
    candidate_positive_ids = {bid for bid, row in benchmark_meta.items() if truthy(row.get('candidate_positive'))}
    rows: list[dict[str, Any]] = []
    for policy in POLICY_ORDER:
        if not any(((policy, metric) in rank_maps for metric in METRIC_ORDER)):
            continue
        for metric in METRIC_ORDER:
            ranks = rank_maps.get((policy, metric), {})
            for denom_name, denom_ids in [('all_benchmark_units', benchmark_ids), ('candidate_positive_units', sorted(candidate_positive_ids))]:
                denom = len(denom_ids)
                for k in TOPKS:
                    hits = sum((1 for bid in denom_ids if ranks.get(bid, 10 ** 9) <= k))
                    rows.append({'policy': policy, 'metric': metric, 'metric_label': METRIC_LABELS[metric], 'denominator': denom_name, 'topk': k, 'n_eval': denom, 'hits': hits, 'rate': hits / denom if denom else 0.0})
                oracle_hits = sum((1 for bid in denom_ids if ranks.get(bid, 10 ** 9) < 10 ** 9))
                rows.append({'policy': policy, 'metric': metric, 'metric_label': METRIC_LABELS[metric], 'denominator': denom_name, 'topk': 'candidate_oracle', 'n_eval': denom, 'hits': oracle_hits, 'rate': oracle_hits / denom if denom else 0.0})
    return rows
