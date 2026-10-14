from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from pandas.errors import EmptyDataError
REPO_ROOT = Path(__file__).resolve().parents[2]

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')

def default_artifact_root(repo_root: Path) -> Path:
    env = os.environ.get('BIORXNGEN_ARTIFACT_ROOT')
    if env:
        return Path(env).expanduser().resolve()
    return repo_root / 'data' / 'full_artifacts'

def first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if path.exists() and (path.is_dir() or path.stat().st_size > 0):
            return path
    return None

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

def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(path)

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

def as_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 't', 'yes', 'y'}
    return bool(value)

def normalize_ec(value: Any) -> str:
    text = as_text(value)
    if text.upper().startswith('EC:'):
        text = text[3:].strip()
    return text

def is_full_ec4(value: Any) -> bool:
    text = normalize_ec(value)
    parts = text.split('.')
    return len(parts) == 4 and all((part.isdigit() for part in parts))

def sha1_short(text: str, length: int=16) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:length]

def artifact_paths(artifact_root: Path, cfg: dict[str, Any]) -> dict[str, Path | None]:
    return {'substrate_universe': first_existing([Path(cfg['substrate_universe_path']) if cfg.get('substrate_universe_path') else None, artifact_root / 'reaction_inputs/substrate_universe/full_cohort_substrate_universe.parquet']), 'substrate_scope_fallback': first_existing([artifact_root / 'reaction_inputs/substrate_universe/full_cohort_substrate_universe.parquet']), 'external_sequence_scope': first_existing([Path(cfg['external_sequence_scope_path']) if cfg.get('external_sequence_scope_path') else None, artifact_root / 'reaction_inputs/sequence_scope/external_sequences.parquet']), 'internal_sequence_chunks': first_existing([Path(cfg['internal_sequence_chunks_dir']) if cfg.get('internal_sequence_chunks_dir') else None, artifact_root / 'reaction_inputs/sequence_scope/internal_sequence_links']), 'sequence_scope_metrics': first_existing([artifact_root / 'reaction_inputs/sequence_scope/metrics.json']), 'substrate_scope_metrics': first_existing([artifact_root / 'reaction_inputs/substrate_universe/metrics.json'])}

def load_substrates(path: Path | None, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = read_table(path)
    if df.empty:
        return (pd.DataFrame(), {'status': 'missing_or_empty', 'path': str(path) if path else None})
    work = df.copy()
    if 'substrate_smiles' not in work.columns and 'canonical_smiles' in work.columns:
        work['substrate_smiles'] = work['canonical_smiles'].astype(str)
    if 'substrate_scope_source' not in work.columns and 'scope_source' in work.columns:
        work['substrate_scope_source'] = work['scope_source'].astype(str)
    if 'substrate_scope_confidence' not in work.columns and 'scope_confidence' in work.columns:
        work['substrate_scope_confidence'] = work['scope_confidence'].astype(str)
    if 'substrate_source_databases' not in work.columns and 'source_databases' in work.columns:
        work['substrate_source_databases'] = work['source_databases'].astype(str)
    if 'substrate_source_reaction_ids' not in work.columns and 'source_reaction_ids' in work.columns:
        work['substrate_source_reaction_ids'] = work['source_reaction_ids'].astype(str)
    if 'substrate_in_agora2_metabolite_set' not in work.columns and 'in_agora2_metabolite_set' in work.columns:
        work['substrate_in_agora2_metabolite_set'] = work['in_agora2_metabolite_set'].map(as_bool)
    if 'substrate_evidence_tier' not in work.columns:
        scope = work.get('substrate_scope_source', pd.Series('', index=work.index)).astype(str).str.lower()
        work['substrate_evidence_tier'] = 'S4_miscellaneous_context_source'
        work.loc[scope.str.contains('rhea', na=False), 'substrate_evidence_tier'] = 'S1_external_reaction_db_ec_supported'
        work.loc[scope.str.contains('agora', na=False), 'substrate_evidence_tier'] = 'S2_agora_native_known_positive'
    if 'substrate_uid' not in work.columns:
        work['substrate_uid'] = [f'substrate:{i:08d}' for i in range(len(work))]
    required = ['substrate_smiles', 'ec_number', 'substrate_uid']
    if any((col not in work.columns for col in required)):
        return (pd.DataFrame(), {'status': 'missing_required_columns', 'columns': list(work.columns)})
    work['ec_number'] = work['ec_number'].map(normalize_ec)
    work = work[work['ec_number'].ne('') & work['substrate_smiles'].notna()].copy()
    partial_ec_rows = int((~work['ec_number'].map(is_full_ec4)).sum())
    if bool(cfg.get('exact_ec4_only', True)):
        work = work[work['ec_number'].map(is_full_ec4)].copy()
    allowed_tiers = set(cfg.get('allowed_substrate_tiers') or ['S1_external_reaction_db_ec_supported', 'S2_agora_native_known_positive'])
    if allowed_tiers:
        work = work[work['substrate_evidence_tier'].isin(allowed_tiers)].copy()
    keep_cols = ['substrate_uid', 'substrate_smiles', 'substrate_heavy_atoms', 'ec_number', 'substrate_scope_source', 'substrate_scope_confidence', 'substrate_source_databases', 'substrate_source_reaction_ids', 'substrate_in_agora2_metabolite_set', 'substrate_evidence_tier']
    for col in keep_cols:
        if col not in work.columns:
            work[col] = ''
    work = work[keep_cols].drop_duplicates(['ec_number', 'substrate_smiles']).reset_index(drop=True)
    return (work, {'status': 'ok', 'path': str(path) if path else None, 'rows': int(len(work)), 'unique_substrates': int(work['substrate_smiles'].nunique()), 'unique_ec_numbers': int(work['ec_number'].nunique()), 'partial_ec_rows_seen_before_filter': partial_ec_rows, 'allowed_substrate_tiers': sorted(allowed_tiers)})

def prepare_internal_chunk(path: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ['model_id', 'model_name', 'reaction_id', 'ec_number', 'gene_product_id', 'internal_sequence_uid', 'sequence', 'query_type', 'needs_review', 'sequence_length']
    df = read_table(path, columns=columns)
    if df.empty:
        return (pd.DataFrame(), {'status': 'empty', 'path': str(path)})
    required = ['model_id', 'ec_number', 'sequence']
    if any((col not in df.columns for col in required)):
        return (pd.DataFrame(), {'status': 'missing_required_columns', 'path': str(path), 'columns': list(df.columns)})
    work = df[df['model_id'].notna() & df['ec_number'].notna() & df['sequence'].notna()].copy()
    if 'needs_review' in work.columns:
        work = work[~work['needs_review'].map(as_bool)].copy()
    work['ec_number'] = work['ec_number'].map(normalize_ec)
    partial_ec_rows = int((~work['ec_number'].map(is_full_ec4)).sum())
    if bool(cfg.get('exact_ec4_only', True)):
        work = work[work['ec_number'].map(is_full_ec4)].copy()
    work['protein_sequence'] = work['sequence'].astype(str)
    work = work[work['protein_sequence'].str.len().ge(int(cfg.get('min_protein_length', 30)))].copy()
    if work.empty:
        return (pd.DataFrame(), {'status': 'no_valid_rows', 'path': str(path), 'input_rows': int(len(df))})
    work['model_id'] = work['model_id'].astype(str)
    work['model_name'] = work['model_name'].astype(str) if 'model_name' in work.columns else ''
    work['sequence_uid'] = work['internal_sequence_uid'].astype(str) if 'internal_sequence_uid' in work.columns else ''
    work['sequence_uid'] = work['sequence_uid'].where(work['sequence_uid'].ne(''), work['gene_product_id'].astype(str) if 'gene_product_id' in work.columns else '')
    work['sequence_hash'] = work['protein_sequence'].map(sha1_short)
    work['enzyme_source_namespace'] = 'internal_agora_gene_annotation_sequence'
    work['protein_ref_db'] = 'agora_gene_annotation'
    work['protein_accession'] = work['gene_product_id'].astype(str) if 'gene_product_id' in work.columns else work['sequence_uid']
    work['reaction_id'] = work['reaction_id'].astype(str) if 'reaction_id' in work.columns else ''
    work['protein_sequence_source'] = 'agora_gene_annotation_sequence'
    work['protein_acceptance_tier'] = 'internal_primary_sequence'
    work['protein_confidence'] = 1.0
    work['sequence_length'] = pd.to_numeric(work['sequence_length'], errors='coerce') if 'sequence_length' in work.columns else work['protein_sequence'].str.len()
    work['enzyme_key'] = work['enzyme_source_namespace'] + ':' + work['model_id'].astype(str) + ':' + work['ec_number'].astype(str) + ':' + work['sequence_hash'].astype(str)
    out = work.sort_values(['ec_number', 'model_id', 'sequence_length'], ascending=[True, True, False]).drop_duplicates(['enzyme_key'], keep='first').reset_index(drop=True)
    return (out[enzyme_reference_columns()], {'status': 'ok', 'path': str(path), 'input_rows': int(len(df)), 'prepared_rows': int(len(out)), 'unique_models': int(out['model_id'].nunique()), 'unique_ec_numbers': int(out['ec_number'].nunique()), 'partial_ec_rows_seen_before_filter': partial_ec_rows})

def prepare_external_sequences(path: Path | None, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = read_table(path)
    if df.empty:
        return (pd.DataFrame(), {'status': 'missing_or_empty', 'path': str(path) if path else None})
    seq_col = 'sequence' if 'sequence' in df.columns else 'amino_acid_sequence' if 'amino_acid_sequence' in df.columns else ''
    accession_col = 'external_accession' if 'external_accession' in df.columns else 'accession' if 'accession' in df.columns else 'protein_ref'
    source_col = 'source_db' if 'source_db' in df.columns else 'source' if 'source' in df.columns else ''
    stage_col = 'protein_sequence_source' if 'protein_sequence_source' in df.columns else 'protein_sequence_source' if 'protein_sequence_source' in df.columns else ''
    tier_col = 'primary_acceptance_tier' if 'primary_acceptance_tier' in df.columns else 'protein_acceptance_tier' if 'protein_acceptance_tier' in df.columns else ''
    confidence_col = 'confidence' if 'confidence' in df.columns else 'protein_confidence' if 'protein_confidence' in df.columns else ''
    required = [seq_col, 'ec_number', 'model_id', accession_col]
    if any((not col or col not in df.columns for col in required)):
        return (pd.DataFrame(), {'status': 'missing_required_columns', 'columns': list(df.columns)})
    work = df[df[seq_col].notna() & df['ec_number'].notna() & df['model_id'].notna()].copy()
    work['ec_number'] = work['ec_number'].map(normalize_ec)
    partial_ec_rows = int((~work['ec_number'].map(is_full_ec4)).sum())
    if bool(cfg.get('exact_ec4_only', True)):
        work = work[work['ec_number'].map(is_full_ec4)].copy()
    work['protein_sequence'] = work[seq_col].astype(str)
    work = work[work['protein_sequence'].str.len().ge(int(cfg.get('min_protein_length', 30)))].copy()
    if work.empty:
        return (pd.DataFrame(), {'status': 'no_valid_rows', 'path': str(path) if path else None, 'input_rows': int(len(df))})
    work['model_id'] = work['model_id'].astype(str)
    work['model_name'] = work['model_name'].astype(str) if 'model_name' in work.columns else ''
    work['sequence_hash'] = work['protein_sequence'].map(sha1_short)
    work['sequence_uid'] = work['sequence_uid'].astype(str) if 'sequence_uid' in work.columns else work[accession_col].astype(str)
    work['enzyme_source_namespace'] = 'external_third_party_strain_ec_sequence'
    work['protein_ref_db'] = work[source_col].astype(str) if source_col else ''
    work['protein_accession'] = work[accession_col].astype(str)
    work['reaction_id'] = work['reaction_id'].astype(str) if 'reaction_id' in work.columns else ''
    work['protein_sequence_source'] = work[stage_col].astype(str) if stage_col else ''
    work['protein_acceptance_tier'] = work[tier_col].astype(str) if tier_col else ''
    work['protein_confidence'] = pd.to_numeric(work[confidence_col], errors='coerce') if confidence_col else pd.NA
    work['sequence_length'] = pd.to_numeric(work['sequence_length'], errors='coerce') if 'sequence_length' in work.columns else work['protein_sequence'].str.len()
    work['enzyme_key'] = work['enzyme_source_namespace'] + ':' + work['model_id'].astype(str) + ':' + work['ec_number'].astype(str) + ':' + work['sequence_hash'].astype(str)
    out = work.sort_values(['ec_number', 'model_id', 'protein_confidence', 'sequence_length'], ascending=[True, True, False, False]).drop_duplicates(['enzyme_key'], keep='first').reset_index(drop=True)
    return (out[enzyme_reference_columns()], {'status': 'ok', 'path': str(path) if path else None, 'input_rows': int(len(df)), 'prepared_rows': int(len(out)), 'unique_models': int(out['model_id'].nunique()), 'unique_ec_numbers': int(out['ec_number'].nunique()), 'partial_ec_rows_seen_before_filter': partial_ec_rows})

def enzyme_reference_columns() -> list[str]:
    return ['enzyme_key', 'model_id', 'model_name', 'ec_number', 'sequence_uid', 'sequence_hash', 'protein_accession', 'protein_ref_db', 'protein_sequence', 'sequence_length', 'reaction_id', 'protein_sequence_source', 'protein_acceptance_tier', 'protein_confidence', 'enzyme_source_namespace']

def pair_columns() -> list[str]:
    return ['candidate_pair_id', 'round_id', 'model_id', 'model_name', 'ec_number', 'enzyme_key', 'sequence_hash', 'protein_ref_db', 'protein_accession', 'enzyme_source_namespace', 'substrate_uid', 'substrate_smiles', 'substrate_heavy_atoms', 'substrate_evidence_tier', 'substrate_scope_source', 'substrate_scope_confidence', 'substrate_source_databases', 'substrate_source_reaction_ids', 'substrate_in_agora2_metabolite_set', 'edge_build_status', 'needs_generator_product', 'needs_esp_scoring', 'needs_external_reaction_validation', 'needs_full_agora2_overlap_audit']

def clean_segment_outputs(pair_dir: Path, enzyme_dir: Path, segment_id: str) -> None:
    for path in pair_dir.glob(f'{segment_id}_part_*.parquet'):
        path.unlink(missing_ok=True)
    for path in enzyme_dir.glob(f'{segment_id}_enzyme_reference.parquet'):
        path.unlink(missing_ok=True)

def write_segment_pairs(segment_id: str, enzymes: pd.DataFrame, substrates_by_ec: dict[str, pd.DataFrame], pair_dir: Path, enzyme_dir: Path, cfg: dict[str, Any], sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_dir.mkdir(parents=True, exist_ok=True)
    enzyme_dir.mkdir(parents=True, exist_ok=True)
    partition_rows = int(cfg.get('partition_rows', 250000))
    sample_limit = int(cfg.get('sample_rows', 200))
    max_pairs_per_segment = int(cfg.get('max_pairs_per_segment', 0))
    buffer: list[dict[str, Any]] = []
    part_idx = 0
    written_files: list[str] = []
    candidate_edges = 0
    models: set[str] = set()
    ecs: set[str] = set()
    enzyme_keys_with_substrates: set[str] = set()
    substrates_seen: set[str] = set()

    def flush() -> None:
        nonlocal part_idx, buffer
        if not buffer:
            return
        part_path = pair_dir / f'{segment_id}_part_{part_idx:05d}.parquet'
        pd.DataFrame(buffer, columns=pair_columns()).to_parquet(part_path, index=False)
        written_files.append(str(part_path))
        part_idx += 1
        buffer = []
    for enzyme in enzymes.to_dict('records'):
        ec = as_text(enzyme.get('ec_number'))
        sub_df = substrates_by_ec.get(ec)
        if sub_df is None or sub_df.empty:
            continue
        models.add(as_text(enzyme.get('model_id')))
        ecs.add(ec)
        enzyme_keys_with_substrates.add(as_text(enzyme.get('enzyme_key')))
        for sub in sub_df.to_dict('records'):
            sub_uid = as_text(sub.get('substrate_uid')) or sha1_short(as_text(sub.get('substrate_smiles')))
            pair_id = 'pair:' + sha1_short('|'.join([as_text(enzyme.get('enzyme_key')), ec, sub_uid, as_text(sub.get('substrate_smiles'))]), length=24)
            row = {'candidate_pair_id': pair_id, 'round_id': 1, 'model_id': enzyme.get('model_id'), 'model_name': enzyme.get('model_name'), 'ec_number': ec, 'enzyme_key': enzyme.get('enzyme_key'), 'sequence_hash': enzyme.get('sequence_hash'), 'protein_ref_db': enzyme.get('protein_ref_db'), 'protein_accession': enzyme.get('protein_accession'), 'enzyme_source_namespace': enzyme.get('enzyme_source_namespace'), 'substrate_uid': sub_uid, 'substrate_smiles': sub.get('substrate_smiles'), 'substrate_heavy_atoms': sub.get('substrate_heavy_atoms'), 'substrate_evidence_tier': sub.get('substrate_evidence_tier'), 'substrate_scope_source': sub.get('substrate_scope_source'), 'substrate_scope_confidence': sub.get('substrate_scope_confidence'), 'substrate_source_databases': sub.get('substrate_source_databases'), 'substrate_source_reaction_ids': sub.get('substrate_source_reaction_ids'), 'substrate_in_agora2_metabolite_set': sub.get('substrate_in_agora2_metabolite_set'), 'edge_build_status': 'pending_generator_and_esp', 'needs_generator_product': True, 'needs_esp_scoring': True, 'needs_external_reaction_validation': True, 'needs_full_agora2_overlap_audit': True}
            buffer.append(row)
            if len(sample_rows) < sample_limit:
                sample_rows.append(row)
            candidate_edges += 1
            substrates_seen.add(as_text(sub.get('substrate_smiles')))
            if len(buffer) >= partition_rows:
                flush()
            if max_pairs_per_segment > 0 and candidate_edges >= max_pairs_per_segment:
                break
        if max_pairs_per_segment > 0 and candidate_edges >= max_pairs_per_segment:
            break
    flush()
    eligible_enzymes = enzymes[enzymes['enzyme_key'].isin(enzyme_keys_with_substrates)].copy()
    enzyme_ref_path = enzyme_dir / f'{segment_id}_enzyme_reference.parquet'
    if eligible_enzymes.empty:
        pd.DataFrame(columns=enzyme_reference_columns()).to_parquet(enzyme_ref_path, index=False)
    else:
        eligible_enzymes.to_parquet(enzyme_ref_path, index=False)
    return {'segment_id': segment_id, 'candidate_edges': int(candidate_edges), 'pair_partitions': int(len(written_files)), 'pair_partition_files': '|'.join(written_files), 'enzyme_reference_file': str(enzyme_ref_path), 'eligible_enzyme_rows': int(len(eligible_enzymes)), 'eligible_models': int(len(models)), 'eligible_ec_numbers': int(len(ecs)), 'eligible_enzyme_keys': int(len(enzyme_keys_with_substrates)), 'substrates': int(len(substrates_seen))}

def build_ec_summary_from_segments(segment_summaries: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    rows = []
    for rec in segment_summaries:
        rows.append({'segment_id': rec.get('segment_id'), 'source_namespace': rec.get('source_namespace'), 'candidate_edges': rec.get('candidate_edges', 0), 'eligible_enzyme_rows': rec.get('eligible_enzyme_rows', 0), 'eligible_models': rec.get('eligible_models', 0), 'eligible_ec_numbers': rec.get('eligible_ec_numbers', 0), 'pair_partitions': rec.get('pair_partitions', 0)})
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=['segment_id', 'source_namespace', 'candidate_edges', 'eligible_enzyme_rows', 'eligible_models', 'eligible_ec_numbers', 'pair_partitions'])
    out.to_csv(output_dir / 'segment_summary.csv', index=False)
    return out

def build_pair_worklist(cfg: dict[str, Any] | None=None) -> dict[str, Any]:
    cfg = dict(cfg or {})
    repo_root = Path(cfg.get('repo_root') or REPO_ROOT).resolve()
    artifact_root = Path(cfg.get('artifact_root') or default_artifact_root(repo_root)).resolve()
    output_dir = Path(cfg.get('output_dir') or artifact_root / 'reaction_inputs/enzyme_substrate_worklist').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_dir = output_dir / 'pairs'
    enzyme_dir = output_dir / 'enzyme_references'
    pair_dir.mkdir(parents=True, exist_ok=True)
    enzyme_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(artifact_root, cfg)
    substrates, substrate_metrics = load_substrates(paths['substrate_universe'], cfg)
    if substrates.empty and paths.get('substrate_scope_fallback') != paths.get('substrate_universe'):
        fallback_substrates, fallback_metrics = load_substrates(paths['substrate_scope_fallback'], cfg)
        if not fallback_substrates.empty:
            substrates = fallback_substrates
            substrate_metrics = {**fallback_metrics, 'fallback_used_after_empty_substrate_universe': True, 'first_attempt': substrate_metrics}
    if substrates.empty:
        summary = {'stage': 'reaction_inputs/enzyme_substrate_worklist', 'created_at': now_iso(), 'status': 'blocked', 'reason': 'Missing EC-conditioned substrate universe. Supply the deposited reaction inputs.', 'inputs': {name: str(path) if path else None for name, path in paths.items()}, 'metrics': {'candidate_edges': 0, 'pair_partitions': 0}, 'output_artifacts': {}}
        write_json(output_dir / 'worklist_summary.json', summary)
        return summary
    substrates_by_ec = {ec: part.reset_index(drop=True) for ec, part in substrates.groupby('ec_number', sort=True)}
    completed_segments: set[str] = set()
    progress_path = output_dir / 'worklist_progress.json'
    manifest_path = output_dir / 'worklist_manifest.csv'
    resume = bool(cfg.get('resume', True))
    manifest_rows: list[dict[str, Any]] = []
    if resume and progress_path.exists():
        progress = read_json(progress_path)
        completed_segments = set(progress.get('completed_segments', []))
    if resume and manifest_path.exists():
        existing_manifest = read_table(manifest_path)
        if not existing_manifest.empty:
            manifest_rows = existing_manifest.to_dict('records')
    sample_rows: list[dict[str, Any]] = []
    segment_summaries: list[dict[str, Any]] = []
    prep_summaries: list[dict[str, Any]] = []

    def register_segment(segment_id: str, source_namespace: str, source_path: str, enzymes: pd.DataFrame, prep: dict[str, Any]) -> None:
        nonlocal manifest_rows
        prep_summaries.append({'segment_id': segment_id, 'source_namespace': source_namespace, **prep})
        if segment_id in completed_segments:
            return
        manifest_rows = [row for row in manifest_rows if row.get('segment_id') != segment_id]
        clean_segment_outputs(pair_dir, enzyme_dir, segment_id)
        result = write_segment_pairs(segment_id, enzymes, substrates_by_ec, pair_dir, enzyme_dir, cfg, sample_rows)
        result.update({'source_namespace': source_namespace, 'source_path': source_path, 'status': 'complete', 'completed_at': now_iso()})
        manifest_rows.append(result)
        segment_summaries.append(result)
        completed_segments.add(segment_id)
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        write_json(progress_path, {'updated_at': now_iso(), 'completed_segments': sorted(completed_segments), 'manifest_csv': str(manifest_path)})
    external_df, external_prep = prepare_external_sequences(paths['external_sequence_scope'], cfg)
    if not external_df.empty:
        register_segment('external_primary', 'external_third_party_strain_ec_sequence', str(paths['external_sequence_scope']), external_df, external_prep)
    else:
        prep_summaries.append({'segment_id': 'external_primary', 'source_namespace': 'external_third_party_strain_ec_sequence', **external_prep})
    chunk_dir = paths['internal_sequence_chunks']
    chunk_paths = sorted(Path(chunk_dir).glob('*.parquet')) if chunk_dir and Path(chunk_dir).exists() else []
    max_internal_chunks = int(cfg.get('max_internal_chunks', 0))
    if max_internal_chunks > 0:
        chunk_paths = chunk_paths[:max_internal_chunks]
    for idx, chunk_path in enumerate(chunk_paths):
        segment_id = f'internal_chunk_{idx:05d}'
        internal_df, internal_prep = prepare_internal_chunk(chunk_path, cfg)
        if internal_df.empty:
            prep_summaries.append({'segment_id': segment_id, 'source_namespace': 'internal_agora_gene_annotation_sequence', **internal_prep})
            continue
        register_segment(segment_id, 'internal_agora_gene_annotation_sequence', str(chunk_path), internal_df, internal_prep)
    manifest_df = pd.DataFrame(manifest_rows)
    if manifest_df.empty:
        manifest_df = pd.DataFrame(columns=['segment_id', 'source_namespace', 'source_path', 'candidate_edges', 'pair_partitions', 'eligible_enzyme_rows', 'eligible_models', 'eligible_ec_numbers', 'eligible_enzyme_keys', 'substrates', 'status', 'completed_at'])
    for numeric_col in ['candidate_edges', 'pair_partitions', 'eligible_enzyme_rows', 'eligible_models', 'eligible_ec_numbers', 'eligible_enzyme_keys', 'substrates']:
        if numeric_col in manifest_df.columns:
            manifest_df[numeric_col] = pd.to_numeric(manifest_df[numeric_col], errors='coerce').fillna(0).astype('int64')
    manifest_df.to_csv(manifest_path, index=False)
    segment_summary = build_ec_summary_from_segments(manifest_rows, output_dir)
    sample_csv = output_dir / 'worklist_sample.csv'
    pd.DataFrame(sample_rows, columns=pair_columns()).to_csv(sample_csv, index=False)
    prep_summary_csv = output_dir / 'input_preparation_summary.csv'
    pd.DataFrame(prep_summaries).to_csv(prep_summary_csv, index=False)
    namespace_summary = manifest_df.groupby('source_namespace', as_index=False).agg(candidate_edges=('candidate_edges', 'sum'), pair_partitions=('pair_partitions', 'sum'), eligible_enzyme_rows=('eligible_enzyme_rows', 'sum'), eligible_models_segment_sum=('eligible_models', 'sum'), eligible_ec_numbers_segment_sum=('eligible_ec_numbers', 'sum')).sort_values('candidate_edges', ascending=False) if not manifest_df.empty and 'source_namespace' in manifest_df.columns else pd.DataFrame(columns=['source_namespace', 'candidate_edges', 'pair_partitions', 'eligible_enzyme_rows'])
    namespace_summary_csv = output_dir / 'namespace_summary.csv'
    namespace_summary.to_csv(namespace_summary_csv, index=False)
    metrics = {'created_at': now_iso(), 'status': 'completed', 'substrate_rows': int(len(substrates)), 'substrate_unique_ec_numbers': int(substrates['ec_number'].nunique()), 'substrate_unique_smiles': int(substrates['substrate_smiles'].nunique()), 'input_internal_chunk_files': int(len(chunk_paths)), 'completed_segments': int(len(completed_segments)), 'candidate_edges': int(pd.to_numeric(manifest_df.get('candidate_edges', pd.Series(dtype='int64')), errors='coerce').fillna(0).sum()), 'pair_partitions': int(pd.to_numeric(manifest_df.get('pair_partitions', pd.Series(dtype='int64')), errors='coerce').fillna(0).sum()), 'eligible_enzyme_rows': int(pd.to_numeric(manifest_df.get('eligible_enzyme_rows', pd.Series(dtype='int64')), errors='coerce').fillna(0).sum()), 'resume_enabled': resume, 'normalized_design': 'pair partitions reference enzyme_key; protein sequences are stored once in enzyme-reference partitions', 'substrate_preparation': substrate_metrics}
    metrics_json = output_dir / 'worklist_metrics.json'
    summary_json = output_dir / 'worklist_summary.json'
    write_json(metrics_json, metrics)
    summary = {'stage': 'reaction_inputs/enzyme_substrate_worklist', 'created_at': metrics['created_at'], 'config': {'artifact_root': str(artifact_root), 'output_dir': str(output_dir), 'exact_ec4_only': bool(cfg.get('exact_ec4_only', True)), 'partition_rows': int(cfg.get('partition_rows', 250000)), 'allowed_substrate_tiers': cfg.get('allowed_substrate_tiers') or ['S1_external_reaction_db_ec_supported', 'S2_agora_native_known_positive']}, 'inputs': {name: str(path) if path else None for name, path in paths.items()}, 'metrics': metrics, 'decision': {'can_enter_generation_and_compatibility_scoring': metrics['candidate_edges'] > 0, 'reason': 'Partitioned enzyme-substrate pairs and normalized enzyme references.' if metrics['candidate_edges'] > 0 else 'No enzyme-substrate pairs were produced; inspect substrate and enzyme EC overlap.'}, 'output_artifacts': {'output_dir': str(output_dir), 'pair_partition_dir': str(pair_dir), 'enzyme_reference_partition_dir': str(enzyme_dir), 'manifest_csv': str(manifest_path), 'segment_summary_csv': str(output_dir / 'segment_summary.csv'), 'namespace_summary_csv': str(namespace_summary_csv), 'input_preparation_summary_csv': str(prep_summary_csv), 'sample_csv': str(sample_csv), 'metrics_json': str(metrics_json), 'summary_json': str(summary_json)}}
    write_json(summary_json, summary)
    return summary
