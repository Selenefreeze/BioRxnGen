"""Match measured structures to ranked products and native AGORA2 outputs."""
from collections import defaultdict
import json

import pandas as pd

from structures import aliases, composition, side_keys, structure_key

TOP_K = (1, 5, 10, 20, 30)


def assay_targets(data):
    conditions = pd.read_csv(data / "conditions.tsv", sep="\t", keep_default_na=False)
    measured = pd.read_csv(data / "compound_observations.tsv", sep="\t", keep_default_na=False)
    measured = measured.loc[measured.connectivity_smiles.ne("") & measured.detected.eq(1)].merge(
        conditions[["dataset", "condition", "cohort", "scope"]], on=["dataset", "condition"],
        validate="many_to_one")
    measured["structure_key"] = measured.connectivity_smiles.map(structure_key)
    adjusted = measured.scope.ne("omm12")
    measured.loc[adjusted, "structure_key"] = measured.loc[adjusted, "structure_key"].map(
        lambda key: aliases().get(key, key))
    measured["above_control"] = pd.to_numeric(measured.above_control, errors="coerce").eq(1)
    measured["repeated"] = pd.to_numeric(measured.positive_signal_n, errors="coerce").ge(2)
    if measured.structure_key.eq("").any():
        raise ValueError("Unresolved assay structure")
    targets = []
    for scope, frame in measured.groupby("scope", sort=True):
        separator = " | " if scope == "omm12" else "|"
        targets.append(frame.groupby(["dataset", "condition", "cohort", "structure_key"], sort=True).agg(
            names=("source_label", lambda labels: " | ".join(sorted(set(labels)))),
            source_feature=("source_feature", lambda values: separator.join(sorted(set(values)))),
            above_control=("above_control", "max"),
            repeated=("repeated", "max"),
        ).reset_index())
    return conditions, pd.concat(targets, ignore_index=True)


def candidate_index(rows, targets, scope):
    ranks, contexts, balanced = (defaultdict(dict) for _ in range(3))
    fields = ["model_id", "candidate_rank", "reactant_context_rank", "reactant_side_smiles",
              "generated_product_side_smiles", "product_full_nonisomeric", "candidate_edge_id",
              "candidate_rule_directional_id"]
    for model, rank, context, left, right, full, edge, rule in rows[fields].itertuples(index=False, name=None):
        match = (side_keys(right, scope) - side_keys(left, scope)) & targets
        if not match:
            continue
        a, b = composition(left), composition(full)
        for key in match:
            evidence = (rank, model, edge, rule)
            ranks[model][key] = min(evidence, ranks[model].get(key, evidence))
            contexts[model][key] = min(context, contexts[model].get(key, context))
            if a is not None and a == b:
                balanced[model][key] = min(rank, balanced[model].get(key, rank))
    return ranks, contexts, balanced


def gem_index(folder, scope):
    native = pd.read_parquet(folder / "native_products.parquet")
    exchange = pd.read_csv(folder / "native_exchanges.tsv", sep="\t", keep_default_na=False)
    growth = pd.read_csv(folder / "growth.tsv", sep="\t")
    selected = growth.loc[growth.scope.eq(scope)]
    if len(selected) != 1 or selected.growth_status.iloc[0] != "optimal" or selected.growth.iloc[0] <= 1e-6:
        raise ValueError(f"No evaluable growth state for {folder.name}")
    flux = pd.read_csv(folder / "maximum_net_export.tsv", sep="\t")
    co_growth = "community_co_growth" if scope == "community" else "individual"
    return {
        "native_graph": set(native.structure_key),
        "native_graph_annotated": set(native.loc[native.annotated, "structure_key"]),
        "native_exchange_mapped": set(exchange.loc[exchange.key.ne(""), "key"]),
        "native_gem": set(flux.loc[flux.scope.eq(scope) & flux.hit, "structure_key"]),
        "native_gem_co_growth": set(flux.loc[flux.scope.eq(co_growth) & flux.hit, "structure_key"]),
    }


def match_compounds(rows, gem, conditions, targets):
    metadata = conditions.set_index(["dataset", "condition"])
    indexes = {}
    for scope, group in conditions.groupby("scope"):
        models = set(group.agora2_model_ids.str.split("|").explode())
        keys = set(targets.loc[targets.dataset.isin(group.dataset), "structure_key"])
        indexes[scope] = candidate_index(rows[rows.model_id.isin(models)], keys, scope)
    native = {
        row.cohort: gem_index(gem / row.cohort, "community" if row.strains > 1 else "individual")
        for row in conditions.drop_duplicates("cohort").itertuples(index=False)
    }
    results = []
    for row in targets.itertuples(index=False):
        condition = metadata.loc[(row.dataset, row.condition)]
        models = condition.agora2_model_ids.split("|")
        ranks, contexts, balanced = indexes[condition.scope]
        key = row.structure_key
        best = min((ranks[m][key] for m in models if key in ranks[m]), default=(None, "", "", ""))
        results.append({
            **row._asdict(),
            **{name: key in keys for name, keys in native[row.cohort].items()},
            "best_rank": best[0], "best_model": best[1], "best_candidate_edge": best[2], "best_rule": best[3],
            "best_context_rank": min((contexts[m][key] for m in models if key in contexts[m]), default=None),
            "best_balanced_rank": min((balanced[m][key] for m in models if key in balanced[m]), default=None),
        })
    return pd.DataFrame(results).sort_values(["dataset", "condition", "structure_key"])


def coverage(frame, dataset, panel, *, unique=False, condition=None):
    def count(mask):
        return frame.loc[mask, "structure_key"].nunique() if unique else int(mask.sum())

    distinct = frame.structure_key.nunique()
    n = distinct if unique else len(frame)
    group = {"level": "dataset"} if condition is None else {"condition": condition}
    result = dict(dataset=dataset, panel=panel, **group, observations=len(frame),
                  conditions=frame.condition.nunique(), distinct_structures=distinct, gem_evaluable=n,
                  native_gem_hits=count(frame.native_gem), native_gem_co_growth_hits=count(frame.native_gem_co_growth),
                  native_graph_hits=count(frame.native_graph), native_graph_annotated_hits=count(frame.native_graph_annotated),
                  native_exchange_mapped=count(frame.native_exchange_mapped))
    for k in (*TOP_K, None):
        label = f"top{k}" if k else "full"
        matched = frame.best_rank.le(k) if k else frame.best_rank.notna()
        balanced = frame.best_balanced_rank.le(k) if k else frame.best_balanced_rank.notna()
        hits = count(matched)
        result.update({
            f"bio_{label}_hits": hits,
            f"bio_{label}_balanced_hits": count(balanced),
            f"bio_{label}_plus_native_gem_hits": count(matched | frame.native_gem),
            f"bio_{label}_outside_native_graph": count(matched & ~frame.native_graph),
            f"bio_{label}_fraction": hits / n,
        })
    result["native_gem_fraction"] = result["native_gem_hits"] / n
    return result


def sensitivity(frame, dataset, panel):
    for label, target in (("all", frame), ("mapped_to_native_exchanges", frame[frame.native_exchange_mapped])):
        distinct = target.groupby("structure_key").agg(
            gem=("native_gem", "any"), graph=("native_graph", "any"),
            top5=("best_rank", "min"), balanced=("best_balanced_rank", "min"))
        top5 = distinct.top5.le(5)
        balanced = distinct.balanced.le(5)
        yield dict(dataset=dataset, panel=panel, target_subset=label, distinct_structures=len(distinct),
                   native_gem_hits=int(distinct.gem.sum()), bio_top5_hits=int(top5.sum()),
                   balanced_top5_hits=int(balanced.sum()), native_graph_hits=int(distinct.graph.sum()),
                   native_graph_plus_balanced_top5_hits=int((distinct.graph | balanced).sum()))


def build_results(rows, gem, data):
    conditions, targets = assay_targets(data)
    matches = match_compounds(rows, gem, conditions, targets)
    panels = json.loads((data / "panels.json").read_text(encoding="utf-8"))
    by_record, by_structure, by_condition, sensitivity_rows = [], [], [], []
    for study in panels:
        dataset = study["dataset"]
        frame = matches[matches.dataset.eq(dataset)]
        subsets = {"detected": frame, "repeated_detection": frame[frame.repeated],
                   "above_control": frame[frame.above_control],
                   "above_control_exchange_mapped": frame[frame.above_control & frame.native_exchange_mapped]}
        if dataset == "Farag2020_SIHUMIx":
            subsets["untreated_detected"] = frame[frame.condition.str.startswith("Blank ")]
        for label, subset in subsets.items():
            if subset.empty:
                continue
            by_record.append(coverage(subset, dataset, label))
            by_structure.append(coverage(subset, dataset, label, unique=True))
            for condition, group in subset.groupby("condition"):
                by_condition.append(coverage(group, dataset, label, condition=condition))
        sensitivity_rows.extend(sensitivity(subsets[study["criterion"]], dataset, study["criterion"]))
    budgets = [
        dict(model_id=model, candidate_rows=len(frame),
             **{f"top{k}_rows": int(frame.candidate_rank.le(k).sum()) for k in TOP_K})
        for model, frame in rows.groupby("model_id", sort=True)
    ]
    return {
        "per_metabolite.tsv": matches,
        "coverage_condition_records.tsv": pd.DataFrame(by_record),
        "coverage_unique_metabolites.tsv": pd.DataFrame(by_structure),
        "coverage_by_condition.tsv": pd.DataFrame(by_condition),
        "coverage_sensitivity.tsv": pd.DataFrame(sensitivity_rows),
        "candidate_budgets.tsv": pd.DataFrame(budgets),
    }
