# Defined-community metabolite benchmark

We evaluated ranked BioRxnGen candidates against extracellular metabolites enriched in OMM12, ASF and Rios Garza cultures. The primary run used strain-linked EC annotations and native substrates for 19 strains. The frozen Fig. 2 reranker scored the generated products without ESP. Ranking was performed within each strain–EC–substrate input.

The top five candidates per input retained 71 of 114 enriched condition–metabolite observations using 23,629 of 350,265 candidate rows (6.7%). The full candidate set matched 83 observations. Native AGORA2 supported feasible net secretion for 70 observations under anaerobic rich-medium bounds at 10% of maximum community growth. Its intracellular reaction-product inventory matched 104 observations.

| Location | Contents |
| --- | --- |
| [data](data) | 19 model identifiers, 13 community definitions, the assay panel, structure mappings and the exact generation worklists |
| [generation records](results/generation_records.zip) | Original candidate and attempt Parquet files. The EC and sequence-linked runs are stored separately |
| [ranked candidates](results/ranked_candidates.zip) | Full EC candidate features, frozen scores and per-input ranks |
| [ranking results](results/ranked_coverage.tsv) | Top-k and full-pool coverage by study and community, with exchange and ranking-unit sensitivity analyses |
| [analyte ranks](results/ranked_analytes.tsv) | Best rank and supporting strain models for each measured analyte |
| [native GEM records](results/native_gem_records.zip) | Original solver output, growth, exchange mappings and native intracellular products |
| [full-pool coverage](results/study_coverage.tsv) | Full-pool counts and equal-weight means across communities within each study. Includes D’hoe et al. |

Each archive includes a SHA-256 for every member. The [candidate budgets](results/ranking_budgets.tsv) count retained rows across all generation inputs.

## Reproduce

Run from the repository root after the [repository installation](../../README.md#install-and-run). The verification commands check archived candidate matches, solver outputs and frozen model scores.

```bash
python -m pip install -r benchmarks/defined_communities/requirements.txt
python benchmarks/defined_communities/benchmark.py verify
python benchmarks/defined_communities/ranking.py verify
```

For a rerun, the assay command fetches the original source files and rebuilds the assay panel. The model command extracts the required AGORA2 2.01 SBML members from the VMH archive. The generation command uses the [released input archive](https://github.com/Selenefreeze/BioRxnGen/releases/download/v1.0.0/model_and_generation_inputs.zip), including its directional rules and split assignments. Pass the archive path to the command below.

```bash
python benchmarks/defined_communities/benchmark.py assays --output output/community/assays
python benchmarks/defined_communities/benchmark.py fetch-models --output output/community/models
python benchmarks/defined_communities/benchmark.py generate --reference-archive model_and_generation_inputs.zip --output output/community/generation
python benchmarks/defined_communities/ranking.py score --generation output/community/generation --output output/community/ranking
python benchmarks/defined_communities/benchmark.py gem --models output/community/models --output output/community/gem
```

The [source manifest](data/source_data.json) records the assay files and checksums. The [model manifest](data/models.json) identifies the original SBML members and records the display-name encoding repair for VPI-5482. Raw study files and SBML are fetched from their sources.

## References

- Weiss, A. S. et al. *Nature Communications* **14**, 4780 (2023). [OMM12 data](https://doi.org/10.1038/s41467-023-40372-0)
- Medlock, G. L. et al. *Cell Systems* **7**, 245–257.e7 (2018). [ASF data](https://doi.org/10.1016/j.cels.2018.08.003)
- Rios Garza, D. et al. *Nature Communications* **17**, 326 (2026). [Batch culture data](https://doi.org/10.1038/s41467-025-67036-5)
- Heinken, A. et al. *Nature Biotechnology* **41**, 1320–1331 (2023). [AGORA2](https://doi.org/10.1038/s41587-022-01628-0)
- D’hoe, K. et al. *eLife* **7**, e37090 (2018). [Study scope sensitivity](https://doi.org/10.7554/eLife.37090)
