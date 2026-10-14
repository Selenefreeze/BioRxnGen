"""External reaction-unit selection, EC-compatible generation and ranking."""
from __future__ import annotations
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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

def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open('r', encoding='utf-8', newline='') as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter='\t')]

def read_tsv_iter(path: Path):
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            yield dict(row)

def split_values(value: Any) -> list[str]:
    return [clean(x) for x in clean(value).replace(';', '|').replace(',', '|').split('|') if clean(x)]

def load_reaction_support_summary(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_tsv(path):
        rid = clean(row.get('reaction_id'))
        if rid:
            out[rid] = row
    return out

def canonicalize_side(mods: dict[str, Any], smiles: str) -> tuple[str, str, bool]:
    canonical_side = mods['_canonical_unmapped_side']
    strip_spectators = mods['_strip_rule_template_spectators']
    bundle = canonical_side(clean(smiles), isomeric=False)
    full = clean(bundle.get('smiles') if bundle.get('ok') else '')
    no_pw = clean(strip_spectators(full))
    return (full, no_pw, bool(full))

def select_benchmark_units(pd: Any, mods: dict[str, Any], worklist_path: Path, reaction_summary_path: Path, cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    support_summary = load_reaction_support_summary(reaction_summary_path)
    max_units = int(cfg.get('max_benchmark_reactions', 1000))
    max_per_ec_combo = int(cfg.get('max_per_ec_combo', 0))
    min_support_models = int(cfg.get('min_external_support_models', 1))
    require_micromap = bool(cfg.get('require_micromap', False))
    rows_seen = 0
    selected_by_key: dict[str, dict[str, Any]] = {}
    support_models_in_written_worklist: dict[str, set[str]] = defaultdict(set)
    candidate_by_key: dict[str, dict[str, Any]] = {}
    for row in read_tsv_iter(worklist_path):
        rows_seen += 1
        rid = clean(row.get('reaction_id'))
        if not rid:
            continue
        support_models_in_written_worklist[rid].add(clean(row.get('model_id')))
        key = rid
        if key in candidate_by_key:
            continue
        full_target, no_pw_target, target_ok = canonicalize_side(mods, row.get('product_side_smiles_all') or row.get('product_side_smiles_noncurrency'))
        full_reactant, _no_pw_reactant, reactant_ok = canonicalize_side(mods, row.get('reactant_side_smiles_all') or row.get('reactant_side_smiles_noncurrency'))
        ec_ids = split_values(row.get('ec_numbers_for_generation'))
        support = support_summary.get(rid, {})
        total_support = to_int(support.get('complete_ready_strict_all_side_model_count') or support.get('complete_ready_noncurrency_core_model_count'))
        if total_support < min_support_models:
            continue
        if require_micromap and (not truthy(row.get('in_micromap'))):
            continue
        candidate_by_key[key] = {'benchmark_id': rid, 'reaction_id': rid, 'reaction_name': clean(row.get('reaction_name')), 'holdout_split': 'apollo_external', 'candidate_tier': 'external_strain_agora_overlap', 'theme_ids': clean(row.get('theme_ids')), 'in_micromap': truthy(row.get('in_micromap')), 'novelty_layer': clean(row.get('novelty_layer')), 'ec_ids_json': json.dumps(ec_ids, ensure_ascii=False), 'ec_count': len(ec_ids), 'ec_source_for_generation': clean(row.get('ec_source_for_generation')), 'leakage_ec_from_reaction_id': truthy(row.get('leakage_ec_from_reaction_id')), 'reactant_side_smiles': full_reactant, 'target_product_full_nonisomeric': full_target, 'target_product_no_proton_water_nonisomeric': no_pw_target, 'reactant_side_evaluable': reactant_ok, 'target_product_evaluable': target_ok, 'reactant_base_ids': clean(row.get('reactant_base_ids')), 'product_base_ids': clean(row.get('product_base_ids')), 'signature_noncurrency': clean(row.get('signature_noncurrency')), 'external_support_model_count_total': total_support, 'external_support_model_count_written_worklist': 0, 'agora_native_model_count': to_int(row.get('agora_native_model_count')), 'mapped_ec_count': to_int(row.get('mapped_ec_count')), 'native_ec_count': to_int(row.get('native_ec_count'))}
    ec_combo_counts = Counter()
    selected: list[dict[str, Any]] = []
    for key, row in sorted(candidate_by_key.items(), key=lambda kv: (not bool(kv[1].get('in_micromap')), -to_int(kv[1].get('external_support_model_count_total')), clean(kv[0]))):
        ec_combo = clean(row.get('ec_ids_json'))
        if max_per_ec_combo > 0 and ec_combo_counts[ec_combo] >= max_per_ec_combo:
            continue
        row['external_support_model_count_written_worklist'] = len(support_models_in_written_worklist.get(row['reaction_id'], set()))
        selected.append(row)
        ec_combo_counts[ec_combo] += 1
        selected_by_key[key] = row
        if max_units > 0 and len(selected) >= max_units:
            break
    report = {'worklist_path': str(worklist_path), 'worklist_rows_seen': rows_seen, 'unique_candidate_reactions_seen': len(candidate_by_key), 'benchmark_units_selected': len(selected), 'max_benchmark_reactions': max_units, 'max_per_ec_combo': max_per_ec_combo, 'min_external_support_models': min_support_models, 'require_micromap': require_micromap, 'ec_combo_selected': len(ec_combo_counts), 'selection_note': 'Reaction-level deduplication prevents repeated APOLLO model support rows from inflating accuracy.'}
    return (pd.DataFrame(selected), report)

def make_prediction_rows(candidates: Any) -> list[dict[str, Any]]:
    if candidates.empty:
        return []
    work = candidates.copy()
    for col in ['rule_rank', 'product_rank_within_rule', 'rule_train_count']:
        if col in work.columns:
            work[col] = work[col].fillna(0).astype(float)
    work['_template_score'] = work.get('rule_train_count', 0).map(lambda x: math.log1p(float(x))) - work.get('rule_rank', 0) * 0.05 - work.get('product_rank_within_rule', 0) * 0.001
    policies = [('template_rank', '_template_score', False)]
    if 'neural_score' in work.columns and work['neural_score'].notna().any():
        policies.append(('hybrid_reranker', 'neural_score', False))
    out: list[dict[str, Any]] = []
    for policy, score_col, ascending in policies:
        sorted_df = work.sort_values(['benchmark_id', score_col, 'rule_rank', 'product_rank_within_rule'], ascending=[True, ascending, True, True]).copy()
        sorted_df = sorted_df.drop_duplicates(['benchmark_id', 'product_no_proton_water_nonisomeric'], keep='first')
        sorted_df['rank'] = sorted_df.groupby('benchmark_id').cumcount() + 1
        for row in sorted_df.to_dict(orient='records'):
            out.append({'benchmark_id': row.get('benchmark_id'), 'reaction_id': row.get('reaction_id'), 'policy': policy, 'rank': int(row.get('rank') or 0), 'score': row.get(score_col), 'predicted_product_smiles': row.get('product_full_nonisomeric'), 'predicted_product_no_proton_water_smiles': row.get('product_no_proton_water_nonisomeric'), 'target_product_smiles': row.get('target_full_nonisomeric'), 'target_product_no_proton_water_smiles': row.get('target_no_proton_water_nonisomeric'), 'source_ec_num': row.get('source_ec_num') or row.get('ec_num'), 'rule_directional_id': row.get('rule_directional_id'), 'rule_rank': row.get('rule_rank'), 'product_rank_within_rule': row.get('product_rank_within_rule')})
    return out

def parse_json_list(value: Any) -> list[str]:
    text = clean(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [clean(x) for x in parsed if clean(x)]
    except Exception:
        pass
    return [clean(x) for x in text.replace(';', '|').replace(',', '|').split('|') if clean(x)]

def generate_external_candidates(pd: Any, mods: dict[str, Any], units: Any, paths: dict[str, Path | None], cfg: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    load_rule_context = mods['_load_rule_template_prior_and_eval_records']
    resolve_rules = mods['_resolve_rule_template_candidates']
    ec_level = mods['_ec_level_dict_from_text']
    run_smarts = mods['_run_rule_smarts_on_reactant_side']
    strip_spectators = mods['_strip_rule_template_spectators']
    rule_source_depth = mods['_rule_source_depth']
    rule_chunks_dir = paths['rule_chunks_dir']
    split_path = paths['split_assignments_path']
    if rule_chunks_dir is None or not rule_chunks_dir.exists() or split_path is None or (not split_path.exists()):
        return (pd.DataFrame(), units.assign(candidate_generation_status='missing_rule_context'), {'status': 'missing_rule_context', 'rule_chunks_dir': str(rule_chunks_dir), 'split_assignments_path': str(split_path)})
    rule_prior, rule_smarts_by_id, _eval_unused, prior_context = load_rule_context(rule_chunks_dir=rule_chunks_dir, split_assignments_path=split_path, eval_splits=['valid'], max_rows_per_split=1, seed=int(cfg.get('random_seed', 20260628)))
    max_reactions = int(cfg.get('max_reactions', 0))
    max_total_rules = int(cfg.get('max_total_rules', 40))
    rule_topk_per_ec = int(cfg.get('rule_topk_per_ec', 20))
    max_products_per_rule = int(cfg.get('max_products_per_rule', 24))
    max_products_per_reaction = int(cfg.get('max_products_per_reaction', 256))
    max_reactant_permutations = int(cfg.get('max_reactant_permutations', 96))
    work = units.copy()
    work['_ready'] = work['reactant_side_evaluable'].fillna(False).astype(bool) & work['target_product_evaluable'].fillna(False).astype(bool) & (work['ec_count'] > 0)
    work = work.sort_values(['_ready', 'holdout_split', 'reaction_id'], ascending=[False, True, True])
    if max_reactions > 0:
        work = work.head(max_reactions).copy()
    rxn_cache: dict[str, Any] = {}
    candidate_rows = []
    attempt_rows = []
    status_counts = Counter()
    for rec in work.to_dict(orient='records'):
        ec_ids = parse_json_list(rec.get('ec_ids_json'))
        rule_candidates = []
        seen_rules = set()
        for ec in ec_ids:
            meta = {'ec_num': ec, **ec_level(ec)}
            for cand in resolve_rules(rule_prior, rule_smarts_by_id, meta, topk=rule_topk_per_ec):
                rule_id = clean(cand.get('rule_directional_id'))
                if rule_id in seen_rules:
                    continue
                seen_rules.add(rule_id)
                rule_candidates.append({**cand, 'source_ec_num': ec, **meta})
                if len(rule_candidates) >= max_total_rules:
                    break
            if len(rule_candidates) >= max_total_rules:
                break
        if not bool(rec.get('reactant_side_evaluable')):
            status = 'no_reactant_smiles'
        elif not bool(rec.get('target_product_evaluable')):
            status = 'no_target_product_smiles'
        elif not ec_ids:
            status = 'no_ec_examples'
        elif not rule_candidates:
            status = 'no_rule_candidates'
        else:
            status = 'no_template_product'
        n_products = 0
        exact_full_any = False
        exact_no_pw_any = False
        if status == 'no_template_product':
            for cand in rule_candidates:
                products, template_stats = run_smarts(rule_smarts=clean(cand.get('rule')), reactant_side=clean(rec.get('reactant_side_smiles')), rxn_cache=rxn_cache, max_products=max_products_per_rule, max_reactant_permutations=max_reactant_permutations)
                for product_idx, product in enumerate(products, start=1):
                    product_full = clean(product.get('full_nonisomeric'))
                    if not product_full:
                        continue
                    product_no_pw = clean(strip_spectators(product_full))
                    exact_full = bool(product_full and product_full == clean(rec.get('target_product_full_nonisomeric')))
                    exact_no_pw = bool(product_no_pw and product_no_pw == clean(rec.get('target_product_no_proton_water_nonisomeric')))
                    exact_full_any = exact_full_any or exact_full
                    exact_no_pw_any = exact_no_pw_any or exact_no_pw
                    candidate_rows.append({'split': 'micromap_external', 'reaction_record_id': clean(rec.get('reaction_id')), 'reaction_id': clean(rec.get('reaction_id')), 'holdout_split': clean(rec.get('holdout_split')), 'theme_ids': clean(rec.get('theme_ids')), 'candidate_tier': clean(rec.get('candidate_tier')), 'ec_ids_json': clean(rec.get('ec_ids_json')), 'ec_num': clean(cand.get('source_ec_num')), 'source_ec_num': clean(cand.get('source_ec_num')), 'ec1': clean(cand.get('ec1')), 'ec2': clean(cand.get('ec2')), 'ec3': clean(cand.get('ec3')), 'ec4': clean(cand.get('ec4')), 'rule_directional_id': clean(cand.get('rule_directional_id')), 'candidate_rule_directional_id': clean(cand.get('rule_directional_id')), 'rule_rank': int(cand.get('rule_rank') or 0), 'rule_source_key': clean(cand.get('rule_source_key')), 'rule_source_depth': int(rule_source_depth(cand.get('rule_source_key'))), 'rule_train_count': int(cand.get('rule_train_count') or 0), 'rule_smarts_count': int(cand.get('rule_smarts_count') or 0), 'product_rank_within_rule': int(product_idx), 'reactant_side_smiles': clean(rec.get('reactant_side_smiles')), 'product_full_nonisomeric': product_full, 'product_no_proton_water_nonisomeric': product_no_pw, 'product_full_isomeric': clean(product.get('full_isomeric')), 'target_full_nonisomeric': clean(rec.get('target_product_full_nonisomeric')), 'target_no_proton_water_nonisomeric': clean(rec.get('target_product_no_proton_water_nonisomeric')), 'target_product_evaluable': bool(rec.get('target_product_evaluable')), 'reactant_side_evaluable': bool(rec.get('reactant_side_evaluable')), 'required_reactants': int(template_stats.get('required_reactants') or 0), 'available_reactants': int(template_stats.get('available_reactants') or 0), 'permutations_tried': int(template_stats.get('permutations_tried') or 0), 'product_sets_returned': int(template_stats.get('product_sets_returned') or 0), 'template_product_cap_hit': int(template_stats.get('product_cap_hit') or 0), 'exact_full_nonisomeric': exact_full, 'exact_no_proton_water_nonisomeric': exact_no_pw})
                    n_products += 1
                    if n_products >= max_products_per_reaction:
                        break
                if n_products >= max_products_per_reaction:
                    break
            status = 'ok' if n_products else 'no_template_product'
        status_counts[status] += 1
        attempt_rows.append({**rec, 'n_rule_candidates': len(rule_candidates), 'n_product_candidate_rows': n_products, 'candidate_generation_status': status, 'template_candidate_exact_full_any': exact_full_any, 'template_candidate_exact_no_proton_water_any': exact_no_pw_any})
    candidates = pd.DataFrame(candidate_rows)
    attempts = pd.DataFrame(attempt_rows)
    report = {'status': 'ok', 'prior_context': prior_context, 'evaluated_reactions': int(len(attempts)), 'product_candidate_rows': int(len(candidates)), 'reactions_with_products': int((attempts.get('n_product_candidate_rows', pd.Series(dtype=int)) > 0).sum()) if not attempts.empty else 0, 'generation_status_counts': {str(k): int(v) for k, v in status_counts.items()}}
    return (candidates, attempts, report)
