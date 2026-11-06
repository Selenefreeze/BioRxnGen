"""Replay the metabolite benchmark and verify the archived scoring inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest, sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
CODE = HERE.parents[1] / "code"
CHECKPOINT = CODE / "models/product_reranker.pt"
REFERENCE_SHA256 = "991507e6f7d21f9abcdc141e26591ceb4cee20b618187fec68f48d3a4080b33c"


def read_table(archive, name):
    data = BytesIO(archive.read(name))
    return pd.read_parquet(data) if name.endswith(".parquet") else pd.read_csv(data, sep="\t")


def check_archive(archive):
    manifest = json.loads(archive.read("manifest.json"))
    if set(archive.namelist()) != set(manifest) | {"manifest.json"}:
        raise ValueError("Archive contents differ from the manifest")
    for name, record in manifest.items():
        data = archive.read(name)
        if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}")


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


def generate(reference, out, workers):
    with reference.open("rb") as handle:
        if file_digest(handle, "sha256").hexdigest() != REFERENCE_SHA256:
            raise ValueError("Unexpected model_and_generation_inputs.zip checksum")
    with TemporaryDirectory(prefix="biorxngen-metabolome-") as directory:
        work = Path(directory)
        with ZipFile(DATA / "generation_inputs.zip") as archive:
            check_archive(archive)
            archive.extractall(work)
        with ZipFile(reference) as archive:
            for name in archive.namelist():
                if name.startswith("reaction_inputs/directional_rules/") and name.endswith(".parquet"):
                    member = work / "rules/reaction_labels" / Path(name).name
                    member.parent.mkdir(parents=True, exist_ok=True)
                    member.write_bytes(archive.read(name))
            split = archive.extract("reaction_inputs/reaction_split/split_assignments.parquet", work)
            pool = archive.extract("reaction_inputs/substrate_context/ec_annotated_substrate_pool.parquet", work)
        paths = {"rule_chunks_dir": work / "rules", "split_assignments": Path(split),
                 "annotated_substrate_pool": Path(pool)}
        tasks = []
        for scope in sorted((work / "ec").iterdir()):
            destination = out / scope.name
            destination.mkdir(parents=True, exist_ok=True)
            settings = json.loads((scope / "settings.json").read_text(encoding="utf-8"))
            for file in sorted(scope.glob("*_pairs_*.parquet")):
                tasks.append((file, destination, paths, {**settings, "export_reranker_features": True}))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(generation_batch, tasks))


def run_gem(task):
    import cobra
    from native_gem import collect_exchanges, evaluate, graph_inventory, native_community
    from structures import native_structure_keys

    models_dir, out, scope, cohort, members, keys = task
    cobra.Configuration().solver = "glpk"
    models = [cobra.io.read_sbml_model(str(models_dir / (m.removeprefix("M_") + ".xml"))) for m in members]
    if [model.id for model in models] != members:
        raise ValueError(f"Unexpected SBML model IDs for {cohort}")
    with ZipFile(DATA / "generation_inputs.zip") as archive:
        hints = read_table(archive, f"native/{scope}/metabolite_structure_hints.parquet")
        reactions = read_table(archive, f"native/{scope}/reactions.parquet")
    hints["key"] = native_structure_keys(hints, scope)
    index = {(row.model_id, row.metabolite_id): row.key for row in hints.itertuples(index=False)}
    folder = out / cohort
    folder.mkdir(parents=True, exist_ok=True)
    graph_inventory(models, index, reactions).to_parquet(folder / "native_products.parquet", index=False)
    exchanges = pd.concat([collect_exchanges(model, index) for model in models], ignore_index=True)
    exchanges.to_csv(folder / "native_exchanges.tsv", sep="\t", index=False)
    states, outputs = [], []
    profile = "rich_anaerobic"
    for model in models:
        subset = exchanges[exchanges.model.eq(model.id)]
        boundaries = [(r.reaction_id, r.base, r.coefficient, [r.key]) for r in subset.itertuples(index=False)]
        state, flux = evaluate(model, "individual", profile, boundaries, keys)
        states.append(state)
        outputs.extend(flux)
    if len(models) > 1:
        community, exchange_ids, biomass, transfers = native_community(models, exchanges)
        community.id = cohort
        by_base = exchanges.groupby("base").key.agg(lambda values: sorted({v for v in values if v}))
        if by_base.map(len).gt(1).any():
            raise ValueError(f"Conflicting structure annotations for a shared exchange in {cohort}")
        boundaries = [(r, base, -1.0, by_base[base]) for base, r in exchange_ids.items()]
        for mode, floor in (("community", ()), ("community_co_growth", biomass)):
            state, flux = evaluate(community, mode, profile, boundaries, keys, floor)
            states.append(state)
            outputs.extend(flux)
        pd.DataFrame(transfers, columns=["model", "native_exchange", "community_transfer", "base"]).to_csv(
            folder / "community_transfers.tsv", sep="\t", index=False)
    pd.DataFrame(states).to_csv(folder / "growth.tsv", sep="\t", index=False)
    pd.DataFrame(outputs).to_csv(folder / "maximum_net_export.tsv", sep="\t", index=False)


def replay_gem(models, out, workers):
    from download_inputs import verify_models
    from metabolite_coverage import assay_targets

    verify_models(models)
    conditions, targets = assay_targets(DATA)
    tasks = []
    for cohort, frame in conditions.groupby("cohort", sort=True):
        if frame.agora2_model_ids.nunique() != 1 or frame.scope.nunique() != 1:
            raise ValueError(f"Inconsistent strain mapping for {cohort}")
        members = frame.agora2_model_ids.iloc[0].split("|")
        keys = set(targets.loc[targets.cohort.eq(cohort), "structure_key"])
        tasks.append((models, out, frame.scope.iloc[0], cohort, members, keys))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_gem, tasks))


def score(frames):
    import torch
    from biorxngen.reranker import load_model
    from ranking import GENERATION_FIELDS, IDENTITY, assign_ranks, infer

    torch.set_num_threads(2)
    model, config = load_model(CHECKPOINT)
    results = []
    for raw in frames:
        frame = raw[GENERATION_FIELDS].copy()
        frame["reranker_score"] = infer(frame, model, config)
        results.append(frame)
    rows = pd.concat(results, ignore_index=True)
    if rows.duplicated(IDENTITY).any() or not np.isfinite(rows.reranker_score).all():
        raise ValueError("Duplicate candidate identities or non-finite scores")
    return assign_ranks(rows)


def verify():
    from ranking import GENERATION_FIELDS, IDENTITY

    with ZipFile(DATA / "generation_inputs.zip") as inputs, ZipFile(RESULTS / "generation_records.zip") as records:
        check_archive(inputs)
        check_archive(records)
        pairs = {n.removeprefix("ec/").replace("_pairs_", "_attempts_"): len(read_table(inputs, n))
                 for n in inputs.namelist() if "_pairs_" in n}
        attempts = {n: len(read_table(records, n)) for n in records.namelist() if "_attempts_" in n}
        if pairs != attempts or len(pairs) != 25:
            raise ValueError("Generation inputs and attempts do not agree")
        rescored = score(read_table(records, n) for n in sorted(records.namelist()) if "_products_" in n)
    with ZipFile(RESULTS / "ranked_candidates.zip") as archive:
        check_archive(archive)
        saved = read_table(archive, "scored_candidates.parquet")
    if len(saved) != 504532:
        raise ValueError("Unexpected archived candidate count")
    order = [*GENERATION_FIELDS, "candidate_rank", "reactant_context_rank"]
    left, right = (df.sort_values(IDENTITY).reset_index(drop=True) for df in (rescored, saved))
    pd.testing.assert_frame_equal(left[order], right[order], check_dtype=False)
    np.testing.assert_allclose(left.reranker_score, right.reranker_score, rtol=0, atol=1e-6)
    print(f"Verified scores and ranks for {len(saved):,} candidate records.")
    summarize(None, None, None)


def summarize(ranked, gem, output):
    from metabolite_coverage import build_results

    if ranked is None:
        with ZipFile(RESULTS / "ranked_candidates.zip") as archive:
            rows = read_table(archive, "scored_candidates.parquet")
    else:
        rows = pd.read_parquet(ranked)
    if gem is None:
        with TemporaryDirectory(prefix="biorxngen-gem-") as directory:
            with ZipFile(RESULTS / "native_gem_records.zip") as archive:
                check_archive(archive)
                archive.extractall(directory)
            tables = build_results(rows, Path(directory), DATA)
    else:
        tables = build_results(rows, gem, DATA)
    for filename, table in tables.items():
        if output is None:
            expected = pd.read_csv(RESULTS / filename, sep="\t", keep_default_na=False)
            actual = pd.read_csv(StringIO(table.to_csv(sep="\t", index=False)), sep="\t", keep_default_na=False)
            order = list(actual.columns)
            columns = list(expected.columns)
            if set(order) != set(columns):
                raise ValueError(f"Unexpected columns in {filename}")
            pd.testing.assert_frame_equal(
                actual.sort_values(order, kind="stable").reset_index(drop=True),
                expected[order].sort_values(order, kind="stable").reset_index(drop=True),
                check_dtype=False, atol=1e-12)
        else:
            output.mkdir(parents=True, exist_ok=True)
            table.to_csv(output / filename, sep="\t", index=False, lineterminator="\n")
    print("Verified benchmark tables." if output is None else f"Wrote benchmark tables to {output}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    fetch = sub.add_parser("fetch-models")
    run = sub.add_parser("generate")
    ranking = sub.add_parser("score")
    gem = sub.add_parser("gem")
    summary = sub.add_parser("summarize")
    for command in (fetch, run, ranking, gem, summary):
        command.add_argument("--output", type=Path, required=True)
    for command in (run, gem):
        command.add_argument("--workers", type=int, default=4)
    run.add_argument("--reference-archive", type=Path, required=True)
    ranking.add_argument("--generation", type=Path, required=True)
    gem.add_argument("--models", type=Path, required=True)
    summary.add_argument("--ranked", type=Path)
    summary.add_argument("--gem", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        verify()
    elif args.command == "fetch-models":
        from download_inputs import fetch_models
        fetch_models(args.output)
    elif args.command == "generate":
        generate(args.reference_archive, args.output, args.workers)
    elif args.command == "score":
        rows = score(pd.read_parquet(f) for f in sorted(args.generation.glob("*/*_products_*.parquet")))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(args.output, index=False)
    elif args.command == "gem":
        replay_gem(args.models, args.output, args.workers)
    else:
        summarize(args.ranked, args.gem, args.output)


if __name__ == "__main__":
    main()
