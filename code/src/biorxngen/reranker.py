"""Train and apply the candidate-product reranker used for the internal benchmark."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "rule_rank", "rule_source_depth", "rule_train_count_log",
    "rule_smarts_count_log", "product_rank_within_rule", "n_product_fragments",
    "product_smiles_len", "product_no_pw_smiles_len", "required_reactants",
    "available_reactants", "permutations_tried_log", "product_sets_returned_log",
    "template_product_cap_hit", "true_rule_hit_topk",
]
CATEGORICAL_COLUMNS = [
    "ec1", "ec2", "ec3", "ec4", "rule_directional_id",
    "rule_family_id", "candidate_rule_directional_id", "rule_source_key",
]
LABEL_COLUMN = "exact_agora_norm_no_proton_water"


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install the model extra: pip install -e './code[model]'.") from exc
    return torch


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for source, target in (
        ("rule_train_count", "rule_train_count_log"),
        ("rule_smarts_count", "rule_smarts_count_log"),
        ("permutations_tried", "permutations_tried_log"),
        ("product_sets_returned", "product_sets_returned_log"),
    ):
        values = work[source] if source in work else pd.Series(0, index=work.index)
        work[target] = np.log1p(pd.to_numeric(values, errors="coerce").fillna(0.0))
    for name in ("true_rule_hit_topk", "template_product_cap_hit"):
        values = work[name] if name in work else pd.Series(0, index=work.index)
        work[name] = pd.to_numeric(values, errors="coerce").fillna(0).astype(int)
    return work


def _fingerprint(smiles: str, bits: int) -> np.ndarray:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    result = np.zeros(bits, dtype=np.float32)
    try:
        for part in str(smiles).split("."):
            if not part.strip():
                continue
            mol = Chem.MolFromSmiles(part)
            if mol is None:
                mol = Chem.MolFromSmiles(part, sanitize=False)
                if mol is None:
                    raise ValueError("Invalid product SMILES")
                Chem.SanitizeMol(mol, catchErrors=True)
            fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits).GetFingerprint(mol)
            values = np.zeros(bits, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, values)
            result = np.maximum(result, values.astype(np.float32))
    except Exception:
        return np.zeros(bits, dtype=np.float32)
    return result


def feature_arrays(
    frame: pd.DataFrame,
    config: dict,
    *,
    fit: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build the numerical, categorical and Morgan-fingerprint model inputs."""
    work = _feature_frame(frame)
    numeric_cols = config.get("numeric_cols", NUMERIC_COLUMNS)
    categorical_cols = config.get("categorical_cols", CATEGORICAL_COLUMNS)
    hash_size = int(config.get("hash_size", 8192))
    fp_bits = int(config.get("fp_bits", 256))
    numeric = work.reindex(columns=numeric_cols).apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0).to_numpy(dtype=np.float32)
    mean = numeric.mean(axis=0) if fit else np.asarray(config["numeric_mean"], dtype=np.float32)
    std = numeric.std(axis=0) if fit else np.asarray(config["numeric_std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    numeric = ((numeric - mean) / std).astype(np.float32)

    categories = []
    for column in categorical_cols:
        values = work[column].fillna("NA").astype(str) if column in work else pd.Series("NA", index=work.index)
        categories.append(np.asarray([
            int(hashlib.sha1(f"{column}={value}".encode("utf-8", errors="ignore")).hexdigest()[:16], 16)
            % max(2, hash_size)
            for value in values
        ], dtype=np.int64))
    categorical = np.stack(categories, axis=1) if categories else np.empty((len(work), 0), dtype=np.int64)
    smiles = work.get("product_full_nonisomeric", pd.Series("", index=work.index)).fillna("").astype(str)
    cache: dict[str, np.ndarray] = {}
    fingerprints = []
    for value in smiles:
        if value not in cache:
            cache[value] = _fingerprint(value, fp_bits)
        fingerprints.append(cache[value])
    fingerprint = np.stack(fingerprints) if fingerprints else np.empty((0, fp_bits), dtype=np.float32)
    label_col = config.get("label_col", LABEL_COLUMN)
    labels = work.get(label_col, pd.Series(0, index=work.index)).fillna(False).astype(bool).astype(np.float32).to_numpy()
    fitted = {
        "numeric_cols": list(numeric_cols), "categorical_cols": list(categorical_cols),
        "numeric_mean": mean.tolist(), "numeric_std": std.tolist(),
        "hash_size": hash_size, "embed_dim": int(config.get("embed_dim", 16)),
        "fp_bits": fp_bits, "label_col": label_col,
    }
    return numeric, categorical, fingerprint, labels, fitted


def build_model(config: dict):
    torch = _torch()

    class ProductReranker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cat_embedding = torch.nn.Embedding(int(config["hash_size"]), int(config["embed_dim"]))
            width = len(config["numeric_cols"]) + int(config["fp_bits"]) + len(config["categorical_cols"]) * int(config["embed_dim"])
            hidden = int(config.get("hidden_dim", 256))
            dropout = float(config.get("dropout", 0.15))
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(width, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, 1),
            )

        def forward(self, numeric, categorical, fingerprint):
            if categorical.numel():
                embedded = self.cat_embedding(categorical.long()).flatten(start_dim=1)
                x = torch.cat([numeric.float(), fingerprint.float(), embedded], dim=1)
            else:
                x = torch.cat([numeric.float(), fingerprint.float()], dim=1)
            return self.mlp(x).squeeze(-1)

    return ProductReranker()


def load_model(checkpoint: str | Path, device: str = "cpu"):
    torch = _torch()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    config = state["feature_config"]
    model = build_model(config).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, config


def score_candidates(frame: pd.DataFrame, model, config: dict, *, device: str = "cpu", batch_size: int = 65536) -> np.ndarray:
    torch = _torch()
    numeric, categorical, fingerprint, _, _ = feature_arrays(frame, config)
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            end = min(len(frame), start + batch_size)
            logits = model(
                torch.from_numpy(numeric[start:end]).to(device),
                torch.from_numpy(categorical[start:end]).to(device),
                torch.from_numpy(fingerprint[start:end]).to(device),
            )
            values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def train_reranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    checkpoint: str | Path,
    *,
    seed: int = 20260512,
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 8192,
    device: str = "cpu",
) -> dict:
    """Train with the manuscript architecture and validation-AUPRC selection."""
    torch = _torch()
    from sklearn.metrics import average_precision_score
    from torch.utils.data import DataLoader, TensorDataset

    if train.empty or validation.empty:
        raise ValueError("Training and validation sets must be non-empty")
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = {"numeric_cols": NUMERIC_COLUMNS, "categorical_cols": CATEGORICAL_COLUMNS,
            "hash_size": 8192, "embed_dim": 16, "fp_bits": 256, "label_col": LABEL_COLUMN}
    x, c, f, y, config = feature_arrays(train, base, fit=True)
    xv, cv, fv, yv, _ = feature_arrays(validation, config)
    if y.sum() == 0 or yv.sum() == 0:
        raise ValueError("Both partitions require positive examples")
    model = build_model(config).to(device)
    pos_weight = torch.tensor([(len(y) - float(y.sum())) / max(1.0, float(y.sum()))], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    loader = DataLoader(TensorDataset(*[torch.from_numpy(a.copy()) for a in (x, c, f, y)]), batch_size=batch_size, shuffle=True, num_workers=0)
    best = -1.0
    best_epoch = 0
    no_improvement = 0
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, cb, fb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device), cb.to(device), fb.to(device)), yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_scores = torch.sigmoid(model(
                torch.from_numpy(xv).to(device), torch.from_numpy(cv).to(device),
                torch.from_numpy(fv).to(device),
            )).cpu().numpy()
        auprc = float(average_precision_score(yv, valid_scores))
        if auprc > best:
            best, best_epoch, no_improvement = auprc, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "feature_config": config, "valid_selection_score": auprc}, checkpoint)
        else:
            no_improvement += 1
        if no_improvement >= patience:
            break
    return {"best_epoch": best_epoch, "validation_auprc": best,
            "train_rows": len(train), "validation_rows": len(validation),
            "checkpoint": str(checkpoint)}
