from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

def stable_uid(prefix: str, *parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"

def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)

def annotate_products(products: pd.DataFrame) -> pd.DataFrame:
    if products.empty:
        return products
    out = products.copy()
    out["layer_id"] = "candidate_product"
    out["reaction_id"] = [
        stable_uid("reaction", rec.get("candidate_edge_id"), rec.get("reactant_side_smiles"), rec.get("generated_product_side_smiles"))
        for rec in out.to_dict("records")
    ]
    out["merge_into_reliable_graph"] = False
    out["claim_boundary"] = "exploratory_generated_product_pending_validation"
    return out
