"""Replay or audit the defined-community metabolite benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd
import pyarrow.parquet as pq

from structures import side_keys

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
CODE = HERE.parents[1] / "code"
GROUPS = json.loads((DATA / "communities.json").read_text())
OMM = "omm12"
OTHER = "asf_and_rios_garza"
LABELS = (OMM, OTHER)


def read_table(archive, name):
    raw = BytesIO(archive.read(name))
    return pd.read_parquet(raw) if name.endswith(".parquet") else pd.read_csv(raw, sep="\t")


def check_archive(archive):
    manifest = json.loads(archive.read("manifest.json"))
    if set(archive.namelist()) != set(manifest) | {"manifest.json"}:
        raise ValueError("Archive contents differ from manifest")
    for member, record in manifest.items():
        data = archive.read(member)
        if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch for {member}")


def product_index(archive):
    index, counts = {}, {"ec": 0, "sequence": 0}
    for name in archive.namelist():
        if not (name.endswith("/candidate_products.parquet") or "_products_" in name):
            continue
        mode, label, _ = name.split("/", 2)
        rows = read_table(archive, name)
        counts[mode] += len(rows)
        for row in rows.itertuples(index=False):
            produced = side_keys(row.generated_product_side_smiles, label)
            consumed = side_keys(row.reactant_side_smiles, label)
            index.setdefault((mode, row.model_id), set()).update(produced - consumed)
    return index, counts


def require_equal(frame, observed, field):
    if not frame[field].eq(observed).all():
        raise ValueError(f"Recomputed {field} does not match the saved scoring table")


def verify():
    with ZipFile(DATA / "generation_inputs.zip") as archive:
        check_archive(archive)
        ec_inputs = sum(pq.ParquetFile(BytesIO(archive.read(n))).metadata.num_rows
                        for n in archive.namelist() if n.startswith("ec/") and "_pairs_" in n)
    with ZipFile(RESULTS / "generation_records.zip") as archive:
        check_archive(archive)
        generated, counts = product_index(archive)
        attempts = sum(pq.ParquetFile(BytesIO(archive.read(n))).metadata.num_rows
                       for n in archive.namelist() if n.startswith("ec/") and "_attempts_" in n)
    if ec_inputs != 10422 or attempts != ec_inputs or counts != {"ec": 350265, "sequence": 67780}:
        raise ValueError(f"Unexpected generation counts {ec_inputs}, {attempts}, {counts}")
    saved = pd.read_csv(RESULTS / "per_analyte.tsv", sep="\t")
    targets = pd.read_csv(DATA / "assay_targets.tsv", sep="\t")
    identity = ["cohort", "analyte"]
    for column in ("valid", "enriched", "structure_key"):
        require_equal(saved.set_index(identity), targets.set_index(identity)[column], column)

    saved_groups = pd.read_csv(RESULTS / "per_cohort.tsv", sep="\t").query("panel == 'enriched'").set_index("cohort")
    group_counts = (saved[saved.valid & saved.enriched].groupby("cohort")
                    .agg(total=("analyte", "size"), bio_ec=("bio_ec", "sum"),
                         gem_community=("gem_community", "sum"), native_graph=("native_graph", "sum")))
    for field in group_counts.columns:
        require_equal(saved_groups, group_counts[field].reindex(saved_groups.index), field)
    positive, ec_hits, gem_hits = 0, 0, 0
    with ZipFile(RESULTS / "native_gem_records.zip") as archive:
        check_archive(archive)
        for group in GROUPS:
            cohort = group["cohort"]
            label = OMM if cohort.startswith("OMM12_") else OTHER
            folder = OMM if label == OMM else cohort
            frame = saved[saved.cohort.eq(cohort)].copy()
            keys = frame.structure_key.str.split("|")
            supported = {mode: set().union(*(generated.get((mode, m), set()) for m in group["models"]))
                         for mode in ("ec", "sequence")}
            for mode in ("ec", "sequence"):
                require_equal(frame, keys.map(lambda items: bool(set(items) & supported[mode])), f"bio_{mode}")
            flux = read_table(archive, folder + "/maximum_net_export.tsv")
            active = flux[flux.profile.eq("rich_anaerobic") & flux.scope.eq("community")]
            feasible = active.solver_status.eq("optimal") & active.maximum_net_export.gt(1e-6)
            require_equal(active, feasible, "hit")
            hits = set(active.loc[feasible, "structure_key"])
            require_equal(frame, keys.map(lambda items: bool(set(items) & hits)), "gem_community")
            graph = read_table(archive, label + "/native_products.parquet")
            graph_hits = set(graph.loc[graph.model.isin(group["models"]), "structure_key"])
            require_equal(frame, keys.map(lambda items: bool(set(items) & graph_hits)), "native_graph")
            frame = frame[frame.valid & frame.enriched]
            positive += len(frame)
            ec_hits += int(frame.bio_ec.sum())
            gem_hits += int(frame.gem_community.sum())
    if (positive, ec_hits, gem_hits) != (114, 83, 70):
        raise ValueError(f"Unexpected coverage {(positive, ec_hits, gem_hits)}")
    print(f"{len(GROUPS)} groups, {ec_inputs:,} EC contexts, {counts['ec']:,} product rows.")
    print(f"Recovered {ec_hits}/{positive} by EC-constrained generation and {gem_hits}/{positive} by native GEM.")


def fetch_sources(out):
    from download_inputs import fetch_assay_sources

    out.mkdir(parents=True, exist_ok=True)
    fetch_assay_sources(out / "sources")
    (out / "inputs").mkdir(exist_ok=True)
    for name in ("omm12", "asf", "garza"):
        subprocess.run([sys.executable, str(HERE / "assays" / f"prepare_{name}.py")], cwd=out, check=True)
    omm = pd.read_csv(out / "inputs/assay_targets.tsv", sep="\t")
    omm["cohort"] = "OMM12_" + omm.medium
    omm["valid"] &= omm.single_structure
    omm = omm.rename(columns={"structure_keys": "structure_key"})
    datasets = [omm, *(pd.read_csv(out / "cohorts" / d / "targets.tsv", sep="\t") for d in ("asf", "garza"))]
    scored = pd.concat(datasets).set_index(["cohort", "analyte"])
    expected = pd.read_csv(DATA / "assay_targets.tsv", sep="\t").set_index(["cohort", "analyte"])
    for field in ("valid", "enriched", "structure_key"):
        require_equal(expected, scored[field].reindex(expected.index), field)
    print(f"Reproduced {len(expected)} assay records.")


def generation_batch(task):
    file, destination, paths, settings = task
    sys.path.insert(0, str(CODE / "reproduction/backend/scripts"))
    from reaction_generator import generate_products

    products, attempts, report = generate_products(pd.read_parquet(file), paths, settings)
    if report["status"] != "ok":
        raise RuntimeError(f"Generation failed for {file.name}: {report['status']}")
    products.to_parquet(destination / file.name.replace("_pairs_", "_products_"), index=False)
    attempts.to_parquet(destination / file.name.replace("_pairs_", "_attempts_"), index=False)
    audit = destination / file.name.replace("_pairs_", "_audit_").replace(".parquet", ".json")
    audit.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(file.stem, len(products), flush=True)


def replay_generation(reference, out, workers):
    with reference.open("rb") as handle:
        digest = file_digest(handle, "sha256").hexdigest()
    if digest != "991507e6f7d21f9abcdc141e26591ceb4cee20b618187fec68f48d3a4080b33c":
        raise ValueError("The reference archive does not match the v1.0.0 release.")
    with TemporaryDirectory(prefix="biorxngen-community-") as directory:
        work = Path(directory)
        with ZipFile(DATA / "generation_inputs.zip") as archive:
            archive.extractall(work)
        with ZipFile(reference) as archive:
            rules = "reaction_inputs/directional_rules/"
            for name in archive.namelist():
                if name.startswith(rules) and name.endswith(".parquet"):
                    member = work / "rules/reaction_labels" / Path(name).name
                    member.parent.mkdir(parents=True, exist_ok=True)
                    member.write_bytes(archive.read(name))
            split = archive.extract("reaction_inputs/reaction_split/split_assignments.parquet", work)
            cofactors = archive.extract("reaction_inputs/substrate_context/ec_annotated_substrate_pool.parquet", work)
        paths = {"rule_chunks_dir": work / "rules", "split_assignments": Path(split),
                 "annotated_substrate_pool": Path(cofactors)}
        for label in LABELS:
            src = work / "ec" / label
            destination = out / "ec" / label
            destination.mkdir(parents=True, exist_ok=True)
            cfg = json.loads((src / "settings.json").read_text())
            tasks = [(file, destination, paths, cfg) for file in sorted(src.glob("*_pairs_*.parquet"))]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(generation_batch, tasks))
            subprocess.run([
                sys.executable, str(CODE / "scripts/generation.py"), "run", "--artifact-root", str(work),
                "--worklist-manifest", str(work / "sequence" / label / "worklist_manifest.csv"),
                "--rule-chunks", str(work / "rules"), "--split-assignments", split, "--cofactor-pool", cofactors,
                "--output", str(out / "sequence" / label), "--workers", str(workers), "--profile", "standard", "--run",
            ], check=True)


def run_gem(task):
    from native_gem import collect_exchanges, evaluate, graph_inventory, native_community
    from structures import native_structure_keys
    import cobra

    models_dir, out, label, cohort, model_ids, keys, profiles = task
    cobra.Configuration().solver = "glpk"
    models = [cobra.io.read_sbml_model(str(models_dir / (m.removeprefix("M_") + ".xml"))) for m in model_ids]
    with ZipFile(DATA / "generation_inputs.zip") as archive:
        hints = read_table(archive, f"native/{label}/metabolite_structure_hints.parquet")
        reactions = read_table(archive, f"native/{label}/reactions.parquet")
    hints["key"] = native_structure_keys(hints, label)
    index = {(r.model_id, r.metabolite_id): r.key for r in hints.itertuples(index=False)}
    folder = out / cohort
    folder.mkdir(parents=True, exist_ok=True)
    graph_inventory(models, index, reactions).to_parquet(folder / "native_products.parquet", index=False)
    exchanges = pd.concat([collect_exchanges(m, index) for m in models], ignore_index=True)
    exchanges.to_csv(folder / "native_exchanges.tsv", sep="\t", index=False)
    states, outputs = [], []
    for model in models:
        subset = exchanges[exchanges.model.eq(model.id)]
        boundaries = [(r.reaction_id, r.base, r.coefficient, [r.key]) for r in subset.itertuples(index=False)]
        for profile in profiles:
            state, result = evaluate(model, "individual", profile, boundaries, keys)
            states.append(state)
            outputs.extend(result)
    community, exchange_ids, biomass, transfers = native_community(models, exchanges)
    if label != OMM:
        community.id = cohort
    by_base = exchanges.groupby("base").key.agg(lambda values: sorted({x for x in values if x})).to_dict()
    if any(len(keys) > 1 for keys in by_base.values()):
        raise ValueError("Conflicting structure annotations for a shared exchange")
    boundaries = [(r, base, -1.0, by_base[base]) for base, r in exchange_ids.items()]
    for profile in profiles:
        for scope, minimum in (("community", ()), ("community_co_growth", biomass)):
            state, result = evaluate(community, scope, profile, boundaries, keys, minimum)
            states.append(state)
            outputs.extend(result)
    pd.DataFrame(states).to_csv(folder / "growth.tsv", sep="\t", index=False)
    pd.DataFrame(outputs).to_csv(folder / "maximum_net_export.tsv", sep="\t", index=False)
    pd.DataFrame(transfers, columns=["model", "native_exchange", "community_transfer", "base"]).to_csv(
        folder / "community_transfers.tsv", sep="\t", index=False)
    print(cohort, "complete", flush=True)


def replay_gem(models_dir, out, workers, cohort):
    from download_inputs import verify_models
    from native_gem import PROFILES

    verify_models(models_dir)
    targets = pd.read_csv(DATA / "assay_targets.tsv", sep="\t")
    tasks = []
    for group in GROUPS:
        name = group["cohort"]
        if cohort and cohort != name:
            continue
        label = OMM if name.startswith("OMM12_") else OTHER
        if label == OMM and name == "OMM12_APF" and not cohort:
            continue
        selected = targets.cohort.str.startswith("OMM12_") if label == OMM else targets.cohort.eq(name)
        keys = {key for values in targets.loc[selected, "structure_key"] for key in values.split("|")}
        tasks.append((models_dir, out, label, OMM if label == OMM else name, group["models"], keys,
                      tuple(PROFILES) if label == OMM else ("rich_anaerobic",)))
    if not tasks:
        raise ValueError(f"Unknown cohort {cohort}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_gem, tasks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    for name in ("assays", "fetch-models", "generate", "gem"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "generate":
            command.add_argument("--reference-archive", type=Path, required=True)
            command.add_argument("--workers", type=int, default=4)
        if name == "gem":
            command.add_argument("--models", type=Path, required=True)
            command.add_argument("--workers", type=int, default=4)
            command.add_argument("--cohort")
    args = parser.parse_args()
    if args.command == "verify":
        verify()
    elif args.command == "assays":
        fetch_sources(args.output.resolve())
    elif args.command == "fetch-models":
        from download_inputs import fetch_models
        fetch_models(args.output.resolve())
    elif args.command == "generate":
        replay_generation(args.reference_archive.resolve(), args.output.resolve(), args.workers)
    elif args.command == "gem":
        replay_gem(args.models.resolve(), args.output.resolve(), args.workers, args.cohort)


if __name__ == "__main__":
    main()
