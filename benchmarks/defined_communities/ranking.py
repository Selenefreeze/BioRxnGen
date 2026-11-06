"""Score and rank products within each strain–EC–substrate input."""
from biorxngen.reranker import score_candidates

IDENTITY = [
    "model_id", "candidate_edge_id", "reactant_context_name", "reactant_side_smiles",
    "generated_product_side_smiles", "candidate_rule_directional_id",
]
GENERATION_FIELDS = [
    *IDENTITY, "ec_number", "substrate_smiles", "generated_product_no_proton_water_smiles",
    "product_full_nonisomeric", "n_product_fragments", "rule_rank", "rule_source_key",
    "rule_source_depth", "rule_train_count", "rule_smarts_count", "product_rank_within_rule",
    "required_reactants", "available_reactants", "permutations_tried", "product_sets_returned",
    "template_product_cap_hit",
]
TIE_COLUMNS = [
    "reactant_side_smiles", "reactant_context_name", "candidate_rule_directional_id",
    "product_full_nonisomeric", "generated_product_side_smiles",
]


def infer(rows, model, config):
    features = rows.copy()
    levels = features.ec_number.str.split(".")
    for level in range(1, 5):
        features[f"ec{level}"] = levels.str[:level].str.join(".")
    features["rule_directional_id"] = features.candidate_rule_directional_id
    features["rule_family_id"] = features.rule_directional_id.str.split("__", n=1).str[0]
    features["product_smiles_len"] = features.product_full_nonisomeric.str.len()
    features["product_no_pw_smiles_len"] = features.generated_product_no_proton_water_smiles.str.len()
    features["true_rule_hit_topk"] = 0
    if features[GENERATION_FIELDS].isna().any().any() or features.product_full_nonisomeric.eq("").any():
        raise ValueError("Missing reranker inputs")
    return score_candidates(features, model, config)


def assign_ranks(rows):
    for rank, keys in {
        "candidate_rank": ["model_id", "candidate_edge_id"],
        "reactant_context_rank": ["model_id", "candidate_edge_id", "reactant_side_smiles"],
    }.items():
        columns = [*keys, "reranker_score", "rule_rank", "product_rank_within_rule"]
        columns += [c for c in TIE_COLUMNS if c not in keys]
        ascending = [name != "reranker_score" for name in columns]
        rows[rank] = rows.sort_values(columns, ascending=ascending).groupby(keys, sort=False).cumcount() + 1
    return rows
