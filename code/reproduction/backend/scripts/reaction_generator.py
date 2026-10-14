from __future__ import annotations

import argparse

import json

import math

import os

import random

import sys

from collections import Counter

from collections.abc import Iterable

from datetime import datetime, timezone

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except Exception:  # pragma: no cover - server preflight reports this.
    Chem = None
    HAS_RDKIT = False

REPO_ROOT = Path(__file__).resolve().parents[2]

HELPERS_DIR = REPO_ROOT / "backend" / "helpers"

if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))

from generator_preparation import (  # noqa: E402
    _ec_level_dict_from_text,
    _ecfp_bitstring_for_smiles,
    _load_rule_template_prior_and_eval_records,
    _resolve_rule_template_candidates,
    _run_rule_smarts_on_reactant_side,
    _rule_source_depth,
    _strip_rule_template_spectators,
    _try_compute_transformer_protein_embeddings,
    _try_official_xgboost_esp_scores,
)

def default_artifact_root(repo_root: Path) -> Path:
    env = os.environ.get("BIORXNGEN_ARTIFACT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root / "data" / "full_artifacts"

def default_esp_root(repo_root: Path) -> Path:
    env = os.environ.get("BIORXNGEN_ESP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root / "codebases" / "ESP"

def default_esm1b_model_path(repo_root: Path) -> Path:
    env = os.environ.get("BIORXNGEN_ESM1B_MODEL")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root / "models" / "esm1b_t33_650M_UR50S"

def first_existing(paths: Iterable[Path | None]) -> Path | None:
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if path.exists() and (path.is_dir() or path.stat().st_size > 0):
            return path
    return None

def as_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)

def normalize_ec(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.upper().startswith("EC:"):
        text = text[3:].strip()
    return text

def canonical_smiles(value: Any, *, isomeric: bool = False) -> tuple[str, int, str]:
    text = "" if value is None else str(value).strip()
    if not text:
        return "", 0, "empty"
    if not HAS_RDKIT:
        return text, 0, "rdkit_unavailable"
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return "", 0, "parse_failed"
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric), int(mol.GetNumHeavyAtoms()), ""
    except Exception as exc:
        return "", 0, f"{type(exc).__name__}"

def canonical_side(value: Any, *, drop_currency: bool = False) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    currency = {"O", "[H+]", "[H]", "O=O"}
    pieces: list[str] = []
    for frag in text.split("."):
        clean, heavy, err = canonical_smiles(frag)
        if err or not clean or heavy <= 0:
            continue
        if drop_currency and clean in currency:
            continue
        pieces.append(clean)
    return ".".join(sorted(set(pieces)))

def artifact_paths(repo_root: Path, artifact_root: Path, cfg: dict[str, Any]) -> dict[str, Path | None]:
    del repo_root, artifact_root
    return {
        "annotated_substrate_pool": Path(cfg["annotated_substrate_pool_path"]) if cfg.get("annotated_substrate_pool_path") else None,
        "rule_chunks_dir": Path(cfg["rule_chunks_dir"]) if cfg.get("rule_chunks_dir") else None,
        "split_assignments": Path(cfg["split_assignments_path"]) if cfg.get("split_assignments_path") else None,
        "esp_xgboost_model": Path(cfg["esp_xgboost_model_path"]) if cfg.get("esp_xgboost_model_path") else None,
    }

def load_cofactor_panel(path: Path | None) -> dict[str, str]:
    panel = {
        "h2o": "O",
        "oxygen": "O=O",
        "ammonia": "N",
        "phosphate": "O=P([O-])([O-])O",
    }
    if path is None or not path.exists():
        return panel
    try:
        df = pd.read_parquet(path, columns=["substrate_base_metabolite_id", "substrate_smiles"])
    except Exception:
        return panel
    wanted = {
        "M_nad": "nad",
        "M_nadp": "nadp",
        "M_nadh": "nadh",
        "M_nadph": "nadph",
        "M_atp": "atp",
        "M_adp": "adp",
        "M_glu_L": "glu_l",
        "M_pyr": "pyruvate",
        "M_co2": "co2",
        "M_coa": "coa",
    }
    for rec in df[df["substrate_base_metabolite_id"].isin(wanted)].drop_duplicates("substrate_base_metabolite_id").to_dict("records"):
        clean, _heavy, err = canonical_smiles(rec.get("substrate_smiles"))
        if not err and clean:
            panel[wanted[str(rec["substrate_base_metabolite_id"])]] = clean
    return panel

def context_specs_for_ec(substrate: str, ec: str, panel: dict[str, str], max_contexts: int) -> list[dict[str, Any]]:
    prefix = str(ec).split(".", 1)[0]
    specs: list[dict[str, Any]] = [{"context_name": "substrate_only", "reactant_side": substrate, "cofactor_names": ""}]
    if prefix == "1":
        groups = [["nad"], ["nadp"], ["oxygen"], ["h2o"]]
    elif prefix == "2":
        groups = [["atp"], ["coa"], ["glu_l"], ["pyruvate"]]
    elif prefix == "3":
        groups = [["h2o"]]
    elif prefix == "4":
        groups = [["h2o"], ["phosphate"]]
    elif prefix == "5":
        groups = [[]]
    elif prefix == "6":
        groups = [["atp"], ["atp", "h2o"]]
    else:
        groups = [["h2o"]]
    for names in groups:
        if not names:
            continue
        if all(name in panel for name in names):
            specs.append(
                {
                    "context_name": "substrate_plus_" + "_".join(names),
                    "reactant_side": ".".join([substrate] + [panel[name] for name in names]),
                    "cofactor_names": "|".join(names),
                }
            )
    return specs[: max(1, int(max_contexts))]

def build_esp_pair_frame(batch: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_bits = int(cfg.get("substrate_ecfp_bits", 1024))
    max_pairs = int(cfg.get("max_esp_pairs", min(1500, len(batch))))
    rows: list[dict[str, Any]] = []
    for rec in batch.head(max_pairs).to_dict("records"):
        bitstring, ok = _ecfp_bitstring_for_smiles(rec.get("substrate_smiles"), n_bits=n_bits)
        rows.append(
            {
                "candidate_edge_id": rec.get("candidate_edge_id"),
                "esp_pair_id": rec.get("candidate_edge_id"),
                "protein_ref": rec.get("enzyme_sequence_uid") or rec.get("protein_accession"),
                "protein_accession": rec.get("protein_accession"),
                "protein_sequence": rec.get("protein_sequence"),
                "substrate_smiles": rec.get("substrate_smiles"),
                "substrate_ecfp_bitstring": bitstring,
                "substrate_ecfp_ok": bool(ok),
                "model_id": rec.get("model_id"),
                "ec_number": rec.get("ec_number"),
            }
        )
    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df, {"status": "empty_batch"}
    pair_df = pair_df[pair_df["substrate_ecfp_ok"] & pair_df["protein_ref"].notna() & pair_df["protein_sequence"].notna()].copy()
    return pair_df.reset_index(drop=True), {
        "status": "ok",
        "max_esp_pairs": max_pairs,
        "ecfp_sequence_pair_rows": int(len(pair_df)),
        "unique_protein_refs": int(pair_df["protein_ref"].nunique()) if not pair_df.empty else 0,
    }

def score_batch_with_esp(batch: pd.DataFrame, paths: dict[str, Path | None], cfg: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "candidate_edge_id",
        "substrate_ecfp_ok",
        "official_esp_xgboost_score",
        "esp_pair_score_source",
        "passes_esp_relaxed_gate",
        "passes_esp_main_gate",
    ]
    if not bool(cfg.get("run_esp_scoring", True)):
        return pd.DataFrame(columns=columns), {"status": "disabled_by_config"}
    pair_df, pair_report = build_esp_pair_frame(batch, cfg)
    if pair_df.empty:
        return pd.DataFrame(columns=columns), {"status": "empty_pair_df", "pair_report": pair_report}

    protein_df = pair_df[["protein_ref", "protein_sequence"]].drop_duplicates("protein_ref").copy()
    embedding_dir = output_dir / "protein_embedding_cache"
    embedding_df, embedding_report, embedding_path = _try_compute_transformer_protein_embeddings(
        protein_df=protein_df,
        model_path=Path(cfg.get("esm1b_model_path") or default_esm1b_model_path(REPO_ROOT)),
        output_dir=embedding_dir,
        cfg=cfg,
    )
    score_cfg = dict(cfg)
    if paths.get("esp_xgboost_model") and not score_cfg.get("esp_xgboost_model_path"):
        score_cfg["esp_xgboost_model_path"] = str(paths["esp_xgboost_model"])
    pred, esp_report = _try_official_xgboost_esp_scores(
        pair_df=pair_df,
        embedding_df=embedding_df,
        embedding_path=embedding_path,
        esp_root=Path(cfg.get("esp_root") or default_esp_root(REPO_ROOT)),
        cfg=score_cfg,
    )
    scored = pair_df[["candidate_edge_id", "substrate_ecfp_ok"]].copy()
    if pred is None:
        scored["official_esp_xgboost_score"] = np.nan
        scored["esp_pair_score_source"] = f"official_esp_unavailable:{esp_report.get('status')}"
    else:
        scored["official_esp_xgboost_score"] = pred
        scored["esp_pair_score_source"] = "official_esp_xgboost"
    main_threshold = float(cfg.get("esp_main_threshold", 0.63))
    relaxed_threshold = float(cfg.get("esp_relaxed_threshold", 0.525))
    score = pd.to_numeric(scored["official_esp_xgboost_score"], errors="coerce")
    scored["passes_esp_relaxed_gate"] = score.ge(relaxed_threshold).fillna(False).astype(bool)
    scored["passes_esp_main_gate"] = score.ge(main_threshold).fillna(False).astype(bool)
    scored = scored.drop_duplicates("candidate_edge_id", keep="first").reset_index(drop=True)
    report = {
        "status": str(esp_report.get("status") or "unknown"),
        "pair_report": pair_report,
        "embedding_report": embedding_report,
        "official_esp_report": esp_report,
        "predicted_pair_rows": int(score.notna().sum()),
        "passes_esp_relaxed_gate_rows": int(scored["passes_esp_relaxed_gate"].sum()),
        "passes_esp_main_gate_rows": int(scored["passes_esp_main_gate"].sum()),
        "esp_main_threshold": main_threshold,
        "esp_relaxed_threshold": relaxed_threshold,
    }
    return scored, report

def load_rule_context(paths: dict[str, Path | None], cfg: dict[str, Any]) -> tuple[dict[tuple[str, str], Counter], dict[str, Counter]]:
    rule_chunks_dir = paths.get("rule_chunks_dir")
    split_path = paths.get("split_assignments")
    if rule_chunks_dir is None or not Path(rule_chunks_dir).exists():
        raise FileNotFoundError(f"Missing rule chunks dir: {rule_chunks_dir}")
    if split_path is None or not Path(split_path).exists():
        raise FileNotFoundError(f"Missing split assignments: {split_path}")
    seed = int(cfg.get("random_seed", 20260518))
    rule_prior, rule_smarts_by_id, _eval_unused, _ctx = _load_rule_template_prior_and_eval_records(
        rule_chunks_dir=Path(rule_chunks_dir),
        split_assignments_path=Path(split_path),
        eval_splits=["valid"],
        max_rows_per_split=1,
        seed=seed,
    )
    return rule_prior, rule_smarts_by_id

def generate_products(batch: pd.DataFrame, paths: dict[str, Path | None], cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if batch.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "empty_batch", "product_candidate_rows": 0}
    if not bool(cfg.get("run_product_generation", True)):
        return pd.DataFrame(), pd.DataFrame(), {"status": "disabled_by_config", "product_candidate_rows": 0}
    try:
        random.seed(int(cfg.get("random_seed", 20260518)))
        preloaded_rule_context = cfg.get("_preloaded_rule_context")
        if preloaded_rule_context is not None:
            rule_prior, rule_smarts_by_id = preloaded_rule_context
        else:
            rule_prior, rule_smarts_by_id = load_rule_context(paths, cfg)
    except Exception as exc:
        audit = batch.copy()
        audit["generation_status"] = f"rule_context_unavailable:{type(exc).__name__}"
        return pd.DataFrame(), audit, {"status": f"rule_context_unavailable:{type(exc).__name__}", "error": str(exc), "product_candidate_rows": 0}

    panel = cfg.get("_preloaded_cofactor_panel")
    if panel is None:
        panel = load_cofactor_panel(paths.get("annotated_substrate_pool"))
    max_cases = int(cfg.get("max_generation_pairs", min(800, len(batch))))
    max_cases_per_stratum = int(cfg.get("max_generation_pairs_per_stratum", 300))
    rule_topk = int(cfg.get("rule_topk_per_ec", 20))
    max_products_per_rule = int(cfg.get("max_products_per_rule", 8))
    max_reactant_permutations = int(cfg.get("max_reactant_permutations", 96))
    max_product_rows = int(cfg.get("max_product_rows", 20000))
    max_contexts_per_candidate = int(cfg.get("max_contexts_per_candidate", 3))

    work = batch.copy()
    if "official_esp_xgboost_score" in work.columns:
        work["_esp_sort"] = pd.to_numeric(work["official_esp_xgboost_score"], errors="coerce").fillna(-1)
    else:
        work["_esp_sort"] = -1.0
    if "passes_esp_main_gate" not in work.columns:
        work["passes_esp_main_gate"] = False
    if "passes_esp_relaxed_gate" not in work.columns:
        work["passes_esp_relaxed_gate"] = False
    work = work.sort_values(
        ["passes_esp_main_gate", "passes_esp_relaxed_gate", "_esp_sort", "pair_priority_score"],
        ascending=[False, False, False, False],
    )
    if max_cases_per_stratum > 0 and "sampling_stratum" in work.columns:
        work = work.groupby("sampling_stratum", group_keys=False).head(max_cases_per_stratum)
        work = work.sort_values(
            ["passes_esp_main_gate", "passes_esp_relaxed_gate", "_esp_sort", "pair_priority_score"],
            ascending=[False, False, False, False],
        )
    if max_cases > 0:
        work = work.head(max_cases)

    product_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    status_counts: Counter = Counter()
    rxn_cache: dict[str, Any] = {}
    for rec in work.to_dict("records"):
        ec = normalize_ec(rec.get("ec_number"))
        meta = {"ec_num": ec, **_ec_level_dict_from_text(ec)}
        rule_candidates = _resolve_rule_template_candidates(rule_prior, rule_smarts_by_id, meta, topk=rule_topk)
        if not rule_candidates:
            status_counts["no_rule_candidates"] += 1
            attempt_rows.append({**rec, "n_rule_candidates": 0, "n_product_rows": 0, "generation_status": "no_rule_candidates"})
            continue
        contexts = context_specs_for_ec(str(rec.get("substrate_smiles") or ""), ec, panel, max_contexts=max_contexts_per_candidate)
        n_products_for_case = 0
        for context in contexts:
            reactant_side = canonical_side(context.get("reactant_side"), drop_currency=False)
            if not reactant_side:
                continue
            for cand in rule_candidates:
                products, template_stats = _run_rule_smarts_on_reactant_side(
                    rule_smarts=str(cand.get("rule") or ""),
                    reactant_side=reactant_side,
                    rxn_cache=rxn_cache,
                    max_products=max_products_per_rule,
                    max_reactant_permutations=max_reactant_permutations,
                )
                if not products:
                    continue
                rule_depth = int(_rule_source_depth(cand.get("rule_source_key")))
                for product_idx, product in enumerate(products, start=1):
                    product_side = str(product.get("full_nonisomeric") or "")
                    product_no_pw = _strip_rule_template_spectators(product_side)
                    product_clean = canonical_side(product_no_pw or product_side, drop_currency=False)
                    product_noncurrency = canonical_side(product_clean, drop_currency=True)
                    substrate_clean = canonical_side(rec.get("substrate_smiles"), drop_currency=True)
                    if not product_clean:
                        continue
                    row = {
                        **rec,
                        "reactant_context_name": context.get("context_name"),
                        "reactant_cofactor_names": context.get("cofactor_names"),
                        "reactant_side_smiles": reactant_side,
                        "generated_product_side_smiles": product_clean,
                        "generated_product_side_noncurrency": product_noncurrency,
                        "generated_product_no_proton_water_smiles": product_no_pw,
                        "candidate_reaction_smiles": f"{reactant_side}>>{product_clean}",
                        "candidate_rule_directional_id": str(cand.get("rule_directional_id") or ""),
                        "rule_rank": int(cand.get("rule_rank") or 0),
                        "rule_source_key": str(cand.get("rule_source_key") or ""),
                        "rule_source_depth": rule_depth,
                        "rule_train_count": int(cand.get("rule_train_count") or 0),
                        "rule_smarts_count": int(cand.get("rule_smarts_count") or 0),
                        "product_rank_within_rule": int(product_idx),
                        "required_reactants": int(template_stats.get("required_reactants") or 0),
                        "available_reactants": int(template_stats.get("available_reactants") or 0),
                        "permutations_tried": int(template_stats.get("permutations_tried") or 0),
                        "product_sets_returned": int(template_stats.get("product_sets_returned") or 0),
                        "template_product_cap_hit": int(template_stats.get("product_cap_hit") or 0),
                        "generated_product_equals_substrate_noncurrency": bool(product_noncurrency == substrate_clean and product_noncurrency != ""),
                        "external_validation_level": "not_evaluated",
                        "agora_overlap_status": "not_evaluated",
                        "edge_build_status": "pending_validation",
                    }
                    esp_main = 1.0 if as_bool(row.get("passes_esp_main_gate")) else 0.0
                    esp_relaxed = 1.0 if as_bool(row.get("passes_esp_relaxed_gate")) else 0.0
                    rule_term = min(math.log1p(float(row["rule_train_count"])), 8.0) / 4.0
                    row["product_priority_score"] = float(
                        row.get("pair_priority_score", 0.0)
                        + 2.0 * esp_main
                        + 0.8 * esp_relaxed
                        + rule_term
                        + min(rule_depth, 4) / 4.0
                        - (2.0 if row["generated_product_equals_substrate_noncurrency"] else 0.0)
                    )
                    product_rows.append(row)
                    n_products_for_case += 1
                    if max_product_rows > 0 and len(product_rows) >= max_product_rows:
                        break
                if max_product_rows > 0 and len(product_rows) >= max_product_rows:
                    break
            if max_product_rows > 0 and len(product_rows) >= max_product_rows:
                break
        status = "ok" if n_products_for_case else "no_template_product"
        status_counts[status] += 1
        attempt_rows.append({**rec, "n_rule_candidates": len(rule_candidates), "n_product_rows": n_products_for_case, "generation_status": status})
        if max_product_rows > 0 and len(product_rows) >= max_product_rows:
            status_counts["max_product_rows_reached"] += 1
            break

    product_df = pd.DataFrame(product_rows)
    attempt_df = pd.DataFrame(attempt_rows)
    if not product_df.empty:
        product_df = product_df.drop_duplicates(
            ["candidate_edge_id", "reactant_context_name", "candidate_rule_directional_id", "generated_product_side_smiles"],
            keep="first",
        )
        product_df = product_df.sort_values("product_priority_score", ascending=False).reset_index(drop=True)
    metrics = {
        "status": "ok",
        "attempted_generation_cases": int(len(attempt_df)),
        "product_candidate_rows": int(len(product_df)),
        "cases_with_products": int(attempt_df["generation_status"].eq("ok").sum()) if not attempt_df.empty else 0,
        "generation_attempt_stratum_counts": attempt_df["sampling_stratum"].value_counts(dropna=False).astype(int).to_dict()
        if not attempt_df.empty and "sampling_stratum" in attempt_df.columns
        else {},
        "unique_ec_with_products": int(product_df["ec_number"].nunique()) if not product_df.empty else 0,
        "unique_models_with_products": int(product_df["model_id"].nunique()) if not product_df.empty else 0,
        "unique_products": int(product_df["generated_product_side_noncurrency"].nunique()) if not product_df.empty else 0,
        "same_as_substrate_rows": int(product_df["generated_product_equals_substrate_noncurrency"].sum()) if not product_df.empty else 0,
        "generation_status_counts": {str(k): int(v) for k, v in status_counts.items()},
    }
    return product_df, attempt_df, metrics
