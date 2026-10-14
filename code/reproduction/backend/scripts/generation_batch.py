from __future__ import annotations
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from pandas.errors import EmptyDataError
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None
REPO_ROOT = Path(__file__).resolve().parents[2]
from reaction_generator import artifact_paths as generator_artifact_paths, default_artifact_root, generate_products, load_cofactor_panel, load_rule_context, score_batch_with_esp
from exploratory_generation import annotate_products, write_json

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')

def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}

def first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if path.exists() and (path.is_dir() or path.stat().st_size > 0):
            return path
    return None

def read_table(path: Path | None, columns: list[str] | None=None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        if path.suffix.lower() == '.parquet':
            return pd.read_parquet(path, columns=columns)
        if path.suffix.lower() in {'.csv', '.tsv', '.txt'}:
            sep = '\t' if path.suffix.lower() in {'.tsv', '.txt'} else ','
            if columns is None:
                return pd.read_csv(path, sep=sep)
            wanted = set(columns)
            return pd.read_csv(path, sep=sep, usecols=lambda col: col in wanted)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception:
        if path.suffix.lower() == '.parquet':
            try:
                return pd.read_parquet(path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()
    return pd.DataFrame()

def as_text(value: Any) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    return str(value).strip()

def parquet_num_rows(path: Path | None) -> int:
    if path is None:
        return 0
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    if pq is not None:
        try:
            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            pass
    try:
        cols = pd.read_parquet(path).columns.tolist()
        if cols:
            return int(pd.read_parquet(path, columns=[cols[0]]).shape[0])
        return int(pd.read_parquet(path).shape[0])
    except Exception:
        return 0

def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False

def value_to_parquet_text(value: Any) -> str:
    if is_missing(value):
        return ''
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value)

def prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype == 'object':
            safe[col] = safe[col].map(value_to_parquet_text)
    return safe

def write_table(df: pd.DataFrame, path: Path, empty_columns: list[str] | None=None) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = df.copy()
    if safe.empty and len(safe.columns) == 0:
        safe = pd.DataFrame({col: pd.Series(dtype='object') for col in empty_columns or ['empty_result']})
    safe = prepare_for_parquet(safe)
    safe.to_parquet(path, index=False)
    return {'path': str(path), 'rows': int(len(df))}

def write_sample(df: pd.DataFrame, path: Path, sample_rows: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = df.copy()
    if safe.empty and len(safe.columns) == 0:
        safe = pd.DataFrame({'empty_result': pd.Series(dtype='object')})
    safe.head(max(0, int(sample_rows))).to_csv(path, index=False)
    return str(path)

def artifact_paths(repo_root: Path, artifact_root: Path, cfg: dict[str, Any]) -> dict[str, Path | None]:
    worklist_root = artifact_root / 'reaction_inputs/enzyme_substrate_worklist'
    return {'worklist_manifest': first_existing([Path(cfg['worklist_manifest_path']) if cfg.get('worklist_manifest_path') else None, worklist_root / 'worklist_manifest.csv']), 'worklist_metrics': first_existing([worklist_root / 'worklist_metrics.json']), 'worklist_summary': first_existing([worklist_root / 'worklist_summary.json'])}

def parse_pipe_paths(value: Any) -> list[Path]:
    text = as_text(value)
    if not text:
        return []
    return [Path(x) for x in text.split('|') if x.strip()]

def load_manifest(path: Path | None, cfg: dict[str, Any]) -> pd.DataFrame:
    manifest = read_table(path)
    if manifest.empty:
        return manifest
    manifest = manifest.copy()
    for col in ['candidate_edges', 'pair_partitions', 'eligible_enzyme_rows']:
        if col in manifest.columns:
            manifest[col] = pd.to_numeric(manifest[col], errors='coerce').fillna(0).astype('int64')
    namespaces = cfg.get('source_namespaces')
    if namespaces:
        allowed = {str(x) for x in namespaces}
        manifest = manifest[manifest['source_namespace'].astype(str).isin(allowed)].copy()
    segment_ids = cfg.get('segment_ids')
    if segment_ids:
        allowed_segments = {str(x) for x in segment_ids}
        manifest = manifest[manifest['segment_id'].astype(str).isin(allowed_segments)].copy()
    max_segments = int(cfg.get('max_segments', 0) or 0)
    if max_segments > 0:
        manifest = manifest.head(max_segments).copy()
    return manifest.reset_index(drop=True)

def build_partition_tasks(manifest: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for _, rec in manifest.iterrows():
        pair_paths = parse_pipe_paths(rec.get('pair_partition_files'))
        max_parts = int(cfg.get('max_pair_partitions_per_segment', 0) or 0)
        if max_parts > 0:
            pair_paths = pair_paths[:max_parts]
        for part_index, pair_path in enumerate(pair_paths):
            selection = cfg.get('selected_batches')
            if selection is not None and (as_text(rec.get('segment_id')), part_index) not in selection:
                continue
            tasks.append({'segment_id': as_text(rec.get('segment_id')), 'source_namespace': as_text(rec.get('source_namespace')), 'pair_partition_index': int(part_index), 'pair_partition_file': str(pair_path), 'enzyme_reference_file': as_text(rec.get('enzyme_reference_file'))})
    max_tasks = int(cfg.get('max_partition_tasks', 0) or 0)
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    return tasks

def estimate_runtime(worklist_metrics: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    candidate_edges = int(worklist_metrics.get('candidate_edges', 0) or 0)
    chunk_rows = int(cfg.get('chunk_rows', 250) or 250)
    workers = max(1, int(cfg.get('parallel_workers', 1) or 1))
    chunks = math.ceil(candidate_edges / max(1, chunk_rows))
    return {'candidate_edges': candidate_edges, 'chunk_rows': chunk_rows, 'estimated_chunks': int(chunks), 'parallel_workers': workers}

def prepare_generation_batch(pair_df: pd.DataFrame, enzyme_ref: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if pair_df.empty or enzyme_ref.empty:
        return pd.DataFrame()
    keep_ref = ['enzyme_key', 'sequence_uid', 'protein_sequence', 'sequence_length', 'protein_sequence_source', 'protein_acceptance_tier', 'protein_confidence']
    for col in keep_ref:
        if col not in enzyme_ref.columns:
            enzyme_ref[col] = ''
    work = pair_df.merge(enzyme_ref[keep_ref].drop_duplicates('enzyme_key'), on='enzyme_key', how='left')
    work = work[work['protein_sequence'].map(as_text).str.len().gt(0) & work['substrate_smiles'].map(as_text).str.len().gt(0)].copy()
    if work.empty:
        return work
    work['candidate_edge_id'] = work['candidate_pair_id'].astype(str)
    work['round_id'] = 1
    work['enzyme_sequence_uid'] = work['enzyme_key'].astype(str)
    work['protein_sequence_len'] = pd.to_numeric(work.get('sequence_length'), errors='coerce').fillna(work['protein_sequence'].astype(str).str.len())
    work['substrate_source_tiers'] = work.get('substrate_evidence_tier', '')
    work['sampling_stratum'] = work['enzyme_source_namespace'].astype(str) + '|' + work.get('substrate_evidence_tier', pd.Series('', index=work.index)).astype(str)
    tier_bonus = work.get('substrate_evidence_tier', pd.Series('', index=work.index)).map({'S1_external_reaction_db_ec_supported': 2.0, 'S2_agora_native_known_positive': 1.0}).fillna(0.0)
    source_bonus = work['enzyme_source_namespace'].astype(str).map({'external_third_party_strain_ec_sequence': 1.0, 'internal_agora_gene_annotation_sequence': 0.5}).fillna(0.0)
    work['pair_priority_score'] = 1.0 + tier_bonus + source_bonus
    work['passes_esp_main_gate'] = False
    work['passes_esp_relaxed_gate'] = False
    work['official_esp_xgboost_score'] = pd.NA
    return work.reset_index(drop=True)

def generation_cfg(cfg: dict[str, Any], generator_paths: dict[str, Path | None]) -> dict[str, Any]:
    gen_cfg = dict(cfg)
    gen_cfg.setdefault('run_product_generation', True)
    gen_cfg.setdefault('max_generation_pairs', 0)
    gen_cfg.setdefault('max_generation_pairs_per_stratum', 0)
    gen_cfg.setdefault('rule_topk_per_ec', 40)
    gen_cfg.setdefault('max_products_per_rule', 12)
    gen_cfg.setdefault('max_reactant_permutations', 128)
    gen_cfg.setdefault('max_product_rows', 0)
    gen_cfg.setdefault('max_contexts_per_candidate', 3)
    if bool(cfg.get('preload_rule_context_per_worker', True)):
        try:
            gen_cfg['_preloaded_rule_context'] = load_rule_context(generator_paths, gen_cfg)
            gen_cfg['_preloaded_cofactor_panel'] = load_cofactor_panel(generator_paths.get('annotated_substrate_pool'))
        except Exception as exc:
            gen_cfg['_preload_error'] = repr(exc)
    return gen_cfg

def process_partition_task(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    base_cfg, task = payload
    cfg = dict(base_cfg)
    gpu_id = cfg.get('gpu_id')
    if gpu_id is not None and str(gpu_id) != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    repo_root = Path(cfg.get('repo_root') or REPO_ROOT).resolve()
    artifact_root = Path(cfg.get('artifact_root') or default_artifact_root(repo_root)).resolve()
    output_dir = Path(cfg.get('output_dir') or artifact_root / 'generation').resolve()
    generator_paths = generator_artifact_paths(repo_root, artifact_root, cfg)
    gen_cfg = generation_cfg(cfg, generator_paths)
    sample_rows = int(cfg.get('sample_rows', 2000) or 2000)
    chunk_rows = int(cfg.get('chunk_rows', 250) or 250)
    skip_existing = bool(cfg.get('skip_existing_chunks', True))
    segment_id = task['segment_id']
    part_index = int(task['pair_partition_index'])
    pair_path = Path(task['pair_partition_file'])
    enzyme_path = Path(task['enzyme_reference_file'])
    part_dir = output_dir / segment_id / f'pair_part_{part_index:05d}'
    part_dir.mkdir(parents=True, exist_ok=True)
    status_path = part_dir / 'partition_status.json'
    write_json(status_path, {'status': 'started', 'started_at': now_iso(), 'task': task})
    pair_df = read_table(pair_path)
    enzyme_ref = read_table(enzyme_path)
    batch_all = prepare_generation_batch(pair_df, enzyme_ref, cfg)
    max_pairs_per_partition = int(cfg.get('max_pairs_per_partition', 0) or 0)
    if max_pairs_per_partition > 0:
        batch_all = batch_all.head(max_pairs_per_partition).copy()
    total_chunks = math.ceil(len(batch_all) / max(1, chunk_rows)) if len(batch_all) else 0
    chunk_summaries: list[dict[str, Any]] = []
    product_paths: list[str] = []
    attempt_paths: list[str] = []
    for chunk_index in range(total_chunks):
        selection = cfg.get('selected_batches')
        if selection is not None and chunk_index not in selection[segment_id, part_index]:
            continue
        start = chunk_index * chunk_rows
        stop = min(len(batch_all), start + chunk_rows)
        chunk_dir = part_dir / f'chunk_{chunk_index:05d}'
        chunk_dir.mkdir(parents=True, exist_ok=True)
        product_path = chunk_dir / 'candidate_products.parquet'
        attempt_path = chunk_dir / 'generation_attempts.parquet'
        chunk_status = chunk_dir / 'chunk_status.json'
        if skip_existing and product_path.exists() and attempt_path.exists():
            summary = {'segment_id': segment_id, 'pair_partition_index': part_index, 'chunk_index': chunk_index, 'status': 'skipped_existing', 'batch_rows': int(stop - start), 'product_rows': parquet_num_rows(product_path), 'attempt_rows': parquet_num_rows(attempt_path), 'product_path': str(product_path), 'attempt_path': str(attempt_path)}
            chunk_summaries.append(summary)
            product_paths.append(str(product_path))
            attempt_paths.append(str(attempt_path))
            continue
        write_json(chunk_status, {'status': 'started', 'started_at': now_iso(), 'row_start': start, 'row_stop_exclusive': stop})
        batch = batch_all.iloc[start:stop].copy()
        write_table(batch, chunk_dir / 'selected_pairs.parquet')
        write_sample(batch, chunk_dir / 'selected_pairs_sample.csv', sample_rows)
        if bool(cfg.get('run_esp_scoring', False)):
            scored, esp_report = score_batch_with_esp(batch, generator_paths, gen_cfg, chunk_dir)
            if not scored.empty:
                batch = batch.merge(scored.drop(columns=['substrate_ecfp_ok'], errors='ignore'), on='candidate_edge_id', how='left')
        else:
            esp_report = {'status': 'disabled_by_default_for_full_cohort_generation'}
        products, attempts, generation_report = generate_products(batch, generator_paths, gen_cfg)
        products = annotate_products(products)
        if not products.empty:
            products['layer_id'] = 'candidate_product'
            products['segment_id'] = segment_id
            products['pair_partition_index'] = part_index
            products['chunk_index'] = chunk_index
        write_table(products, product_path, ['candidate_edge_id', 'generated_product_side_smiles', 'generated_product_side_noncurrency'])
        write_table(attempts, attempt_path)
        write_sample(products, chunk_dir / 'candidate_products_sample.csv', sample_rows)
        write_sample(attempts, chunk_dir / 'generation_attempts_sample.csv', sample_rows)
        summary = {'segment_id': segment_id, 'source_namespace': task.get('source_namespace'), 'pair_partition_index': part_index, 'chunk_index': chunk_index, 'status': 'completed', 'completed_at': now_iso(), 'batch_rows': int(len(batch)), 'product_rows': int(len(products)), 'attempt_rows': int(len(attempts)), 'esp_report': esp_report, 'generation_report': generation_report, 'product_path': str(product_path), 'attempt_path': str(attempt_path)}
        write_json(chunk_status, summary)
        chunk_summaries.append(summary)
        product_paths.append(str(product_path))
        attempt_paths.append(str(attempt_path))
        pd.DataFrame(chunk_summaries).to_csv(part_dir / 'chunk_summary.csv', index=False)
    metrics = {'segment_id': segment_id, 'source_namespace': task.get('source_namespace'), 'pair_partition_index': part_index, 'pair_partition_rows_loaded': int(len(pair_df)), 'enzyme_reference_rows_loaded': int(len(enzyme_ref)), 'batch_rows': int(len(batch_all)), 'chunk_rows': chunk_rows, 'total_chunks': int(total_chunks), 'completed_or_skipped_chunks': int(len(chunk_summaries)), 'generated_product_rows': int(sum((int(x.get('product_rows', 0) or 0) for x in chunk_summaries))), 'generation_attempt_rows': int(sum((int(x.get('attempt_rows', 0) or 0) for x in chunk_summaries))), 'product_paths': product_paths, 'attempt_paths': attempt_paths}
    write_json(status_path, {'status': 'completed', 'completed_at': now_iso(), 'metrics': metrics, 'task': task})
    return {'status': 'completed', 'task': task, 'metrics': metrics, 'partition_dir': str(part_dir)}

def run_generation_collection(cfg: dict[str, Any] | None=None) -> dict[str, Any]:
    cfg = dict(cfg or {})
    repo_root = Path(cfg.get('repo_root') or REPO_ROOT).resolve()
    artifact_root = Path(cfg.get('artifact_root') or default_artifact_root(repo_root)).resolve()
    output_dir = Path(cfg.get('output_dir') or artifact_root / 'generation').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(repo_root, artifact_root, cfg)
    worklist_metrics = read_json(paths['worklist_metrics'])
    if 'metrics' in worklist_metrics and isinstance(worklist_metrics['metrics'], dict):
        worklist_metrics = worklist_metrics['metrics']
    estimate = estimate_runtime(worklist_metrics, cfg)
    write_json(output_dir / 'runtime_estimate.json', estimate)
    manifest = load_manifest(paths['worklist_manifest'], cfg)
    tasks = build_partition_tasks(manifest, cfg)
    planner_only = bool(cfg.get('planner_only', False))
    parallel_workers = max(1, int(cfg.get('parallel_workers', 1) or 1))
    parallel_workers = min(parallel_workers, max(1, len(tasks))) if tasks else 1
    write_json(output_dir / 'collection_manifest.json', {'status': 'planned' if planner_only else 'started', 'created_at': now_iso(), 'task_count': int(len(tasks)), 'parallel_workers': parallel_workers, 'runtime_estimate': estimate, 'worklist_manifest': str(paths['worklist_manifest']) if paths['worklist_manifest'] else None})
    if planner_only or not tasks:
        summary = {'stage': 'generation', 'status': 'planned_only' if planner_only else 'blocked', 'created_at': now_iso(), 'metrics': {'task_count': int(len(tasks)), 'candidate_pairs': int(worklist_metrics.get('candidate_edges', 0) or 0), 'parallel_workers': parallel_workers, 'runtime_estimate': estimate}, 'output_artifacts': {'output_dir': str(output_dir), 'runtime_estimate_json': str(output_dir / 'runtime_estimate.json'), 'summary_json': str(output_dir / 'generation_summary.json')}}
        write_json(Path(summary['output_artifacts']['summary_json']), summary)
        return summary
    base_cfg = dict(cfg)
    base_cfg['repo_root'] = str(repo_root)
    base_cfg['artifact_root'] = str(artifact_root)
    base_cfg['output_dir'] = str(output_dir)
    results: list[dict[str, Any]] = []
    if parallel_workers <= 1:
        for task in tasks:
            results.append(process_partition_task((base_cfg, task)))
    else:
        start_method = str(cfg.get('multiprocessing_start_method') or 'fork')
        try:
            context = mp.get_context(start_method)
        except ValueError:
            context = mp.get_context()
        with ProcessPoolExecutor(max_workers=parallel_workers, mp_context=context) as executor:
            future_to_task = {executor.submit(process_partition_task, (base_cfg, task)): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({'status': 'failed', 'task': task, 'error': repr(exc)})
    rows = []
    for item in results:
        metrics = item.get('metrics', {})
        task = item.get('task', {})
        rows.append({'segment_id': metrics.get('segment_id') or task.get('segment_id'), 'source_namespace': metrics.get('source_namespace') or task.get('source_namespace'), 'pair_partition_index': metrics.get('pair_partition_index') or task.get('pair_partition_index'), 'status': item.get('status'), 'batch_rows': metrics.get('batch_rows', 0), 'total_chunks': metrics.get('total_chunks', 0), 'completed_or_skipped_chunks': metrics.get('completed_or_skipped_chunks', 0), 'generated_product_rows': metrics.get('generated_product_rows', 0), 'generation_attempt_rows': metrics.get('generation_attempt_rows', 0), 'partition_dir': item.get('partition_dir', ''), 'error': item.get('error', '')})
    result_df = pd.DataFrame(rows)
    result_csv = output_dir / 'partition_results.csv'
    result_df.to_csv(result_csv, index=False)
    failed = result_df[result_df['status'].astype(str).ne('completed')] if not result_df.empty else pd.DataFrame()
    metrics = {'created_at': now_iso(), 'status': 'completed_with_failures' if len(failed) else 'completed', 'task_count': int(len(tasks)), 'parallel_workers': parallel_workers, 'completed_tasks': int(result_df['status'].astype(str).eq('completed').sum()) if not result_df.empty else 0, 'failed_tasks': int(len(failed)), 'total_batch_rows': int(pd.to_numeric(result_df.get('batch_rows', pd.Series(dtype='int64')), errors='coerce').fillna(0).sum()), 'total_generated_product_rows': int(pd.to_numeric(result_df.get('generated_product_rows', pd.Series(dtype='int64')), errors='coerce').fillna(0).sum()), 'total_generation_attempt_rows': int(pd.to_numeric(result_df.get('generation_attempt_rows', pd.Series(dtype='int64')), errors='coerce').fillna(0).sum()), 'runtime_estimate': estimate}
    metrics_json = output_dir / 'collection_metrics.json'
    summary_json = output_dir / 'generation_summary.json'
    write_json(metrics_json, metrics)
    summary = {'stage': 'generation', 'status': metrics['status'], 'created_at': metrics['created_at'], 'config': {key: value for key, value in cfg.items() if key != 'selected_batches'}, 'inputs': {name: str(path) if path else None for name, path in paths.items()}, 'metrics': metrics, 'output_artifacts': {'output_dir': str(output_dir), 'runtime_estimate_json': str(output_dir / 'runtime_estimate.json'), 'partition_result_summary_csv': str(result_csv), 'collection_metrics_json': str(metrics_json), 'summary_json': str(summary_json)}}
    write_json(summary_json, summary)
    return summary
