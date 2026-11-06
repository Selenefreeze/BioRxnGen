"""Evaluate native AGORA2 chemistry and maximum extracellular net secretion."""
import re

import cobra
import pandas as pd
from cobra.util.solver import linear_reaction_coefficients
from cobra.io import sbml

EPS = 1e-6
GROWTH_FRACTION = 0.10
OXYGEN = {"o2", "oxygen"}


def exchange_base(met):
    compartment = re.escape(met.compartment)
    return re.sub(rf"(?:\[{compartment}\]|_{compartment}_?)$", "", met.id)


def pars_id(met):
    return f"M_{exchange_base(met)}__91__{met.compartment}__93__"


def potential_bounds(reaction):
    met, = reaction.metabolites
    coefficient = reaction.metabolites[met]
    return sorted(-coefficient * bound for bound in reaction.bounds)


def graph_inventory(models, hints, parsed_rxns):

    decode = sbml.F_REPLACE[sbml.F_REACTION]
    by_model = {(r.model_id, decode(r.reaction_id)): r for r in parsed_rxns.itertuples(index=False)}
    evidence = []
    for model in models:
        for reaction in model.reactions:
            record = by_model[(model.id, reaction.id)]
            if record is None or record.reaction_type != "enzymatic" or reaction.compartments != {"c"}:
                continue
            for direction, allowed in [(1, reaction.upper_bound > EPS), (-1, reaction.lower_bound < -EPS)]:
                if not allowed:
                    continue
                left, right = set(), []
                for met, coefficient in reaction.metabolites.items():
                    key = hints.get((model.id, pars_id(met)), "")
                    if not key:
                        continue
                    if coefficient * direction < 0:
                        left.add(key)
                    else:
                        right.append((key, met))
                for key, met in right:
                    if key not in left:
                        evidence.append(dict(model=model.id, reaction=reaction.id, direction=direction,
                                             product_id=met.id, structure_key=key,
                                             ec_numbers=record.ec_list_json, has_gpr=record.has_gpr,
                                             annotated=bool(record.has_gpr or record.ec_list_json not in {"","[]"}),
                                             equation=reaction.reaction))
    return pd.DataFrame(evidence)


def collect_exchanges(model, hints):
    rows = []
    for r in model.exchanges:
        if len(r.metabolites) != 1:
            raise ValueError(f"Non-singleton exchange {model.id}:{r.id}")
        met, = r.metabolites
        if met.compartment != "e":
            continue
        base = exchange_base(met)
        rows.append(dict(model=model.id, reaction_id=r.id, metabolite_id=met.id, base=base,
                         name=met.name, formula=met.formula, charge=met.charge,
                         key=hints.get((model.id, pars_id(met)), ""), 
                         coefficient=r.metabolites[met], min_net=potential_bounds(r)[0], max_net=potential_bounds(r)[1]))
    return pd.DataFrame(rows)


def native_community(models, exchanges):
    community = cobra.Model("OMM12_native_shared_medium")
    community.solver = "glpk"
    shared = {}
    boundaries = {}
    member_biomass = []
    evidence = []
    for model in models:
        biomass = linear_reaction_coefficients(model)
        if len(biomass) != 1 or next(iter(biomass.values())) != 1:
            raise ValueError(f"Unexpected objective for {model.id}: {biomass}")
        prefix = model.id + "__"
        ex = {r.reaction_id: r for r in exchanges[exchanges.model.eq(model.id)].itertuples(index=False)}
        for metabolite in model.metabolites:
            metabolite.id = prefix + metabolite.id
            metabolite.compartment = prefix + metabolite.compartment
        for reaction in model.reactions:
            rid = reaction.id
            reaction.id = prefix + rid
            if rid not in ex:
                continue
            row = ex[rid]
            if row.base not in shared:
                shared[row.base] = cobra.Metabolite("shared_" + row.base, compartment="shared_e")
            reaction.add_metabolites({shared[row.base]: -row.coefficient})
            evidence.append((model.id, rid, reaction.id, row.base))
        community.add_reactions(list(model.reactions))
        member_biomass.append(next(iter(biomass)).id)
    for base, metabolite in shared.items():
        reaction = cobra.Reaction("environment_" + base)
        reaction.add_metabolites({metabolite: -1})
        limits = exchanges[exchanges.base.eq(base)]
        reaction.bounds = (min(0, limits.min_net.min()), max(0, limits.max_net.sum()))
        community.add_reactions([reaction])
        boundaries[base] = reaction.id
    community.objective = {community.reactions.get_by_id(r): 1.0/len(models) for r in member_biomass}
    return community, boundaries, member_biomass, evidence


def evaluate(model, scope, profile, boundaries, target_keys, min_co_growth=()):
    if profile != "rich_anaerobic":
        raise ValueError(f"Unsupported medium profile: {profile}")
    rows = []
    with model:
        for reaction_id, base, coefficient, keys in boundaries:
            reaction = model.reactions.get_by_id(reaction_id)
            if base in OXYGEN:
                if coefficient < 0:
                    reaction.lower_bound = max(0.0, reaction.lower_bound)
                else:
                    reaction.upper_bound = min(0.0, reaction.upper_bound)
        biomass_objective = model.objective
        growth = model.slim_optimize(error_value=float("nan"))
        status = model.solver.status
        growth_floor = None
        if pd.notna(growth) and growth > EPS and status == "optimal":
            growth_floor = GROWTH_FRACTION * growth
            biomass_constraint = model.problem.Constraint(biomass_objective.expression, lb=growth_floor, name="growth_floor")
            model.add_cons_vars(biomass_constraint)
            if min_co_growth:
                for rid in min_co_growth:
                    reaction = model.reactions.get_by_id(rid)
                    if reaction.lower_bound < 1e-5:
                        reaction.lower_bound = 1e-5
                check = model.slim_optimize(error_value=float("nan"))
                if not pd.notna(check) or model.solver.status != "optimal":
                    status = f"co_growth_{model.solver.status}"
        for reaction_id, base, coefficient, keys in boundaries:
            for key in sorted(set(keys) & target_keys):
                export = None
                outcome = status
                if growth_floor is not None and status == "optimal":
                    reaction = model.reactions.get_by_id(reaction_id)
                    model.objective = {reaction: -coefficient}
                    export = model.slim_optimize(error_value=float("nan"))
                    outcome = model.solver.status
                rows.append(dict(scope=scope, model=model.id, profile=profile, structure_key=key, metabolite_base=base,
                                 exchange=reaction_id, solver_status=outcome, maximum_net_export=export,
                                 hit=bool(outcome=="optimal" and export is not None and export>EPS)))
    return dict(scope=scope, model=model.id, profile=profile, growth=growth, growth_status=status,
                growth_floor=growth_floor, member_growth_constraint=bool(min_co_growth)), rows
