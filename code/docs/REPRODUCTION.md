# Reproduction guide

Run from the repository root after [installation](../../README.md#install-and-run). Extract the required [archives](../../README.md#data-downloads) into `data/full_artifacts/`; write outputs elsewhere. Use `--help` for options.

## Inference samples

```bash
python code/scripts/reranker.py score --output output/scored_sample.parquet
python code/scripts/compatibility.py score --output output/compatibility_sample.tsv --check
python code/examples/inference.py external
python code/examples/inference.py esp
python code/examples/inference.py graph
```

These use study records and require no downloads.

## Inputs

| Task | Archives |
|---|---|
| Reranker training and scoring | `reranker_inputs` |
| Worklist construction and generation | `model_and_generation_inputs` |
| Standard-map analysis and database matching | `model_and_generation_inputs`, `candidate_reactions_standard`, `reaction_database_matching` |
| Product audit | `model_and_generation_inputs`, `candidate_reactions_standard`, `candidate_reactions_regenerated` |
| Regenerated EC-class counts | `model_and_generation_inputs`, `candidate_reactions_regenerated` |
| External benchmark | `model_and_generation_inputs`, `external_benchmark` |
| ESP inference | `protein_substrate_compatibility` |
| Case inspection | `case_verification` |

## Reranker

```bash
python code/scripts/reranker.py train --candidates data/full_artifacts/benchmark_inputs/reranker_candidates.parquet --split data/benchmark_results/internal_split_manifest.csv --checkpoint output/retrained_reranker.pt --epochs 10
python code/scripts/reranker.py score --candidates data/full_artifacts/benchmark_inputs/reranker_candidates.parquet --checkpoint code/models/product_reranker.pt --output output/full_candidate_scores.parquet
python code/scripts/benchmark.py internal --reranker-scores output/full_candidate_scores.parquet --output output/fig2_metrics.tsv --check
```

Evaluation uses all 540 test units, including those without candidates. It combines fresh reranker scores with saved comparison scores; omit `--check` when evaluating a retrained model.

## Generation

Use the supplied worklist, or rebuild a bounded sample:

```bash
python code/scripts/generation.py worklist --data-root data/full_artifacts --output output/worklist --max-internal-chunks 1 --max-pairs-per-segment 100
python code/scripts/generation.py run --artifact-root data/full_artifacts --worklist-manifest data/full_artifacts/worklist_manifest.csv --rule-chunks data/full_artifacts/reaction_inputs/directional_rules --split-assignments data/full_artifacts/reaction_inputs/reaction_split/split_assignments.parquet --cofactor-pool data/full_artifacts/reaction_inputs/substrate_context/ec_annotated_substrate_pool.parquet --output output/generation --profile standard --max-tasks 1 --max-pairs-per-task 100
```

Generation writes a plan by default; add `--run` to execute. Remove sampling bounds for the full cohort. For all 460 recorded output-limited batches, omit both bounds and replace `--profile standard` with `--profile regenerated --batch-manifest data/full_artifacts/metadata/candidate_product_manifest.tsv`, using a separate output directory.

The [profiles](../configs/generation_profiles.json) differ only in the batch output cap. Regenerated records replace the corresponding standard records; do not add the totals.

## External benchmark

```bash
python code/scripts/benchmark.py external --data-root data/full_artifacts --output output/external_selection --operation selection --check
python code/scripts/benchmark.py external --data-root data/full_artifacts --output output/external_generation --operation generation --check
python code/scripts/benchmark.py external --data-root data/full_artifacts --output output/external_inference --operation inference --check
python code/scripts/benchmark.py recovery --data-root data/full_artifacts --output output/external_recovery.tsv --check
```

Generation and inference default to the original cohort. Add `--profile residual` to either operation for the supplied residual cohort.

## ESP inference

```bash
python code/scripts/compatibility.py score --pairs data/full_artifacts/protein_substrate_compatibility/pair_inputs.parquet --embedding-map data/full_artifacts/protein_substrate_compatibility/protein_embedding_map.parquet --embeddings data/full_artifacts/protein_substrate_compatibility/protein_embeddings.npy --checkpoint code/models/esp_compatibility_classifier.json --reference-scores data/full_artifacts/protein_substrate_compatibility/compatibility_scores.parquet --output output/compatibility.parquet --check
```

ESP uses cached ESM1b embeddings, not new sequence extraction. In samples, `embedding_index` addresses the sample array; `cohort_embedding_index` addresses the full array. Read `substrate_ecfp_bitstring` as text to preserve leading zeros.

## Result checks

```bash
python code/scripts/validation.py files
python code/scripts/validation.py inputs --data-root data/full_artifacts --output output/generation_inputs.json
python code/scripts/validation.py products --data-root data/full_artifacts --check-hashes
python code/scripts/resource.py regenerated --data-root data/full_artifacts --reference data/map_results/regenerated_ec_class_summary.tsv --output output/regenerated_ec_summary.tsv
python code/scripts/resource.py classify --data-root data/full_artifacts --output output/graph_audit --check
python code/scripts/resource.py coverage --data-root data/full_artifacts --output output/model_coverage
python code/scripts/resource.py database --data-root data/full_artifacts --output output/database_matching --check
```

Graph and database checks use standard-generation records; regenerated EC counts use the selected product subset. [Source tables](../../data/results) are indexed by workbook and sheet in [table_index.tsv](../../data/results/table_index.tsv); case-specific scripts are not included. Cobalamin modes are `strict_strong` (direct support), `claimable_strong` (also annotation-matched support) and `diagnostic` (also low-confidence matches).
