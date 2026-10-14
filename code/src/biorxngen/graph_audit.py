"""Signature-level classification and full-union candidate-map statistics."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
from pathlib import Path

import pandas as pd
from rdkit import Chem


NATIVE = "native_substrate_associated"
DATABASE = "database_matched_non_agora"
OVERLAP = "database_matched_agora_overlap"
EXPLORATORY = "exploratory"
IDENTITY = "identity_or_non_insertion"
NUMERIC_PROFILE = ["product_rows", "nonself_rows", "native_nonself_rows", "external_nonself_rows", "self_rows"]


@lru_cache(maxsize=200_000)
def generation_side(value: str) -> str:
    pieces = set()
    for fragment in str(value).split("."):
        mol = Chem.MolFromSmiles(fragment)
        if mol is None:
            raise ValueError(f"Invalid archived molecular structure: {fragment}")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.SanitizeMol(mol)
        clean = Chem.MolToSmiles(mol, isomericSmiles=False)
        if mol.GetNumHeavyAtoms() > 0 and clean not in {"O", "[H+]", "[H]", "O=O"}:
            pieces.add(clean)
    return ".".join(sorted(pieces))


def signature_id(ec: str, reaction: str) -> str:
    digest = hashlib.sha1(f"{str(ec).strip()}||{str(reaction).strip()}".encode("utf-8")).hexdigest()[:24]
    return "signature:" + digest


def ec_class(value: str) -> str:
    return str(value).removeprefix("EC:").strip().split(".")[0]


def module_label(value: str) -> str:
    mapping = {
        "5.4.99.3": "chorismate / aromatic precursor metabolism",
        "3.5.3.12": "polyamine / agmatine metabolism",
        "5.4.2.7": "nucleotide / phosphate metabolism",
        "5.4.2.2": "nucleotide / phosphate metabolism",
        "5.4.2.5": "nucleotide / phosphate metabolism",
        "3.3.2.1": "chorismate / aromatic precursor metabolism",
        "1.4.1.1": "amino-acid / redox metabolism",
        "2.6.1.18": "amino-acid metabolism",
        "2.6.1.21": "amino-acid metabolism",
        "2.6.1.58": "amino-acid metabolism",
        "3.5.3.4": "polyamine / arginine metabolism",
    }
    return mapping.get(str(value), "additional enzyme chemistry")


def prepare_products(products: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    required = {"pair_id", "model_id", "ec_number", "substrate_smiles", "product_smiles", "reaction_smiles", "enzyme_source_namespace"}
    if not required.issubset(products.columns):
        raise ValueError(f"Missing product columns: {sorted(required - set(products.columns))}")
    metadata = pairs[["candidate_pair_id", "enzyme_key", "substrate_evidence_tier"]].rename(columns={"candidate_pair_id": "pair_id"})
    if metadata.pair_id.duplicated().any():
        raise ValueError("Pair identifiers are not unique")
    out = products.merge(metadata, on="pair_id", how="left", validate="many_to_one")
    if out.substrate_evidence_tier.isna().any():
        raise ValueError("A product has no matching pair context")
    basis = out.ec_number.astype(str).str.strip() + "||" + out.reaction_smiles.astype(str).str.strip()
    lookup = {key: "signature:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:24] for key in basis.unique()}
    out["signature_id"] = basis.map(lookup)
    out["noncurrency_product_smiles"] = out.product_smiles.map(generation_side)
    substrate = out.substrate_smiles.map(generation_side)
    out["self_flag"] = out.noncurrency_product_smiles.eq(substrate) & out.noncurrency_product_smiles.ne("")
    out["nonself_flag"] = ~out.self_flag
    out["native_nonself_flag"] = out.substrate_evidence_tier.str.startswith("S2_agora_native") & out.nonself_flag
    out["external_nonself_flag"] = out.substrate_evidence_tier.str.startswith("S1_external") & out.nonself_flag
    out["ec_class"] = out.ec_number.map(ec_class)
    out["module_label"] = out.ec_number.map(module_label)
    return out


def profile_products(products: pd.DataFrame) -> pd.DataFrame:
    return products.groupby("signature_id", as_index=False).agg(
        product_rows=("signature_id", "size"), nonself_rows=("nonself_flag", "sum"),
        native_nonself_rows=("native_nonself_flag", "sum"), external_nonself_rows=("external_nonself_flag", "sum"),
        self_rows=("self_flag", "sum"), ec_number=("ec_number", "first"),
    )


def combine_profiles(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["signature_id", *NUMERIC_PROFILE, "ec_number"])
    frame = pd.concat(parts, ignore_index=True)
    if frame.groupby("signature_id").ec_number.nunique().gt(1).any():
        raise ValueError("A signature is assigned conflicting EC annotations")
    return frame.groupby("signature_id", as_index=False).agg({**{column: "sum" for column in NUMERIC_PROFILE}, "ec_number": "first"})


def classify_signatures(profile: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    if validation.signature_id.duplicated().any():
        raise ValueError("Duplicate signature-validation identifiers")
    out = profile.merge(validation[["signature_id", "validation_tier"]], on="signature_id", how="left", validate="one_to_one")
    if out.validation_tier.isna().any():
        raise ValueError("Signature validation is incomplete")
    out["candidate_class"] = IDENTITY
    out.loc[out.nonself_rows.gt(0), "candidate_class"] = EXPLORATORY
    out.loc[out.native_nonself_rows.gt(0), "candidate_class"] = NATIVE
    out.loc[out.validation_tier.eq("agora_overlap"), "candidate_class"] = OVERLAP
    # Preserve the archived classification priority, including identity rows.
    out.loc[out.validation_tier.eq("rhea_exact_secondary_non_agora"), "candidate_class"] = DATABASE
    out["display_group"] = out.candidate_class.map({NATIVE: NATIVE, EXPLORATORY: EXPLORATORY, DATABASE: "boundary_audit", OVERLAP: "boundary_audit", IDENTITY: "boundary_audit"})
    out["ec_class"] = out.ec_number.map(ec_class)
    out["module_label"] = out.ec_number.map(module_label)
    out["nonidentity_database_candidate"] = out.validation_tier.isin(["rhea_exact_secondary_non_agora", "rhea_exact_non_agora"]) & out.nonself_rows.gt(0)
    return out


class UnionStats:
    def __init__(self) -> None:
        self.rows = self.nonself_rows = self.native_rows = 0
        self.signatures: set[str] = set()
        self.models: set[str] = set()
        self.products: set[str] = set()
        self.ec: set[str] = set()
        self.enzymes: set[str] = set()
        self.model_signatures: set[int] = set()
        self.internal_models: set[str] = set()
        self.external_models: set[str] = set()
        self.internal_signatures: set[str] = set()
        self.external_signatures: set[str] = set()

    def update(self, rows: pd.DataFrame) -> None:
        self.rows += len(rows)
        self.nonself_rows += int(rows.nonself_flag.sum())
        self.native_rows += int(rows.native_nonself_flag.sum())
        self.signatures.update(rows.signature_id.unique())
        self.models.update(rows.model_id.unique())
        self.products.update(rows.noncurrency_product_smiles.unique())
        self.ec.update(rows.ec_number.unique())
        self.enzymes.update(rows.enzyme_key.unique())
        # Preserve the original hash input despite the public identifier prefix.
        keys = rows.model_id.astype(str) + "||RXN_" + rows.signature_id.str.removeprefix("signature:")
        self.model_signatures.update(pd.util.hash_pandas_object(keys, index=False).astype("uint64").unique().tolist())
        for namespace, models, signatures in (
            ("internal_agora_gene_annotation_sequence", self.internal_models, self.internal_signatures),
            ("external_third_party_strain_ec_sequence", self.external_models, self.external_signatures),
        ):
            selected = rows.loc[rows.enzyme_source_namespace.eq(namespace)]
            models.update(selected.model_id.unique())
            signatures.update(selected.signature_id.unique())

    def record(self) -> dict:
        return {
            "reaction_signatures": len(self.signatures), "model_specific_signatures": len(self.model_signatures),
            "candidate_rows": self.rows, "nonself_rows": self.nonself_rows, "native_substrate_nonself_rows": self.native_rows,
            "unique_models": len(self.models), "unique_ec": len(self.ec), "unique_products": len(self.products),
            "unique_enzymes": len(self.enzymes), "internal_models": len(self.internal_models),
            "external_models": len(self.external_models), "both_source_models": len(self.internal_models & self.external_models),
            "internal_signatures": len(self.internal_signatures), "external_signatures": len(self.external_signatures),
        }


def product_inputs(root: Path):
    manifest = pd.read_csv(root / "metadata/candidate_product_manifest.tsv", sep="\t")
    manifest = manifest.loc[manifest.scope.eq("standard")]
    worklist = pd.read_csv(root / "worklist_manifest.csv").set_index("segment_id")
    last_segment = None
    pairs = None
    for row in manifest.itertuples(index=False):
        segment = row.batch_id.split("__")[0]
        if segment != last_segment:
            paths = worklist.loc[segment, "pair_partition_files"].split("|")
            pairs = pd.concat([pd.read_parquet(root / path, columns=["candidate_pair_id", "enzyme_key", "substrate_evidence_tier"]) for path in paths], ignore_index=True)
            last_segment = segment
        columns = ["pair_id", "model_id", "ec_number", "substrate_smiles", "product_smiles", "reaction_smiles", "enzyme_source_namespace"]
        products = pd.read_parquet(root / row.file, columns=columns)
        yield row.batch_id, prepare_products(products, pairs)


def summary_table(stats: dict, dimension: str) -> pd.DataFrame:
    summary = pd.DataFrame([{dimension: value, **value_stats.record()} for (dim, value), value_stats in stats.items() if dim == dimension])
    if dimension == "enzyme_source_namespace":
        internal = stats[(dimension, "internal_agora_gene_annotation_sequence")].models if (dimension, "internal_agora_gene_annotation_sequence") in stats else set()
        external = stats[(dimension, "external_third_party_strain_ec_sequence")].models if (dimension, "external_third_party_strain_ec_sequence") in stats else set()
        # The source table reports global overlap, not within-source overlap.
        summary["both_source_models"] = len(internal & external)
    return summary


def audit_map(root: Path, output: Path, check: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    parts = []
    batches = 0
    for batch_id, products in product_inputs(root):
        parts.append(profile_products(products))
        batches += 1
        if batches % 100 == 0:
            print(f"Profiled {batches} standard batches", flush=True)
    profile = combine_profiles(parts)
    validation = pd.read_parquet(root / "graph_analysis/signature_validation.parquet")
    classes = classify_signatures(profile, validation)
    if check:
        reference = pd.read_parquet(root / "graph_analysis/signature_classification.parquet")
        compare = ["signature_id", *NUMERIC_PROFILE, "ec_number", "validation_tier", "candidate_class", "display_group", "ec_class", "module_label"]
        pd.testing.assert_frame_equal(classes[compare].sort_values("signature_id").reset_index(drop=True), reference[compare].sort_values("signature_id").reset_index(drop=True), check_dtype=False)
    classes.to_parquet(output / "signature_classification.parquet", index=False)
    class_lookup = classes.set_index("signature_id")[["candidate_class", "display_group"]]
    stats = defaultdict(UnionStats)
    native_models = set()
    rows = 0
    for number, (batch_id, products) in enumerate(product_inputs(root), 1):
        products = products.join(class_lookup, on="signature_id", validate="many_to_one")
        if products.candidate_class.isna().any():
            raise ValueError("Missing signature classification")
        rows += len(products)
        native_models.update(products.loc[products.native_nonself_flag, "model_id"].unique())
        for dimension in ("candidate_class", "display_group", "ec_class", "module_label", "enzyme_source_namespace"):
            for value, group in products.groupby(dimension):
                stats[(dimension, str(value))].update(group)
        if number % 100 == 0:
            print(f"Summarized {number} standard batches", flush=True)
    for dimension in ("candidate_class", "display_group", "ec_class", "module_label", "enzyme_source_namespace"):
        summary = summary_table(stats, dimension)
        if check:
            reference_path = root / f"graph_analysis/{dimension}_summary.tsv"
            if dimension == "candidate_class" and not reference_path.exists():
                reference_path = Path(__file__).resolve().parents[3] / "data/map_results/candidate_class_summary.tsv"
            reference = pd.read_csv(reference_path, sep="\t", dtype={dimension: str})
            columns = [dimension, *[column for column in UnionStats().record() if column in reference]]
            pd.testing.assert_frame_equal(summary[columns].sort_values(dimension).reset_index(drop=True), reference[columns].sort_values(dimension).reset_index(drop=True), check_dtype=False)
        summary.to_csv(output / f"{dimension}_summary.tsv", sep="\t", index=False)
    report = {"standard_batches": batches, "candidate_rows": rows, "reaction_signatures": len(classes), "native_substrate_models": len(native_models), "nonidentity_database_candidates": int(classes.nonidentity_database_candidate.sum()), "reference_checks": "passed" if check else "not_requested"}
    return report
