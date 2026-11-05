# Defined-community metabolite benchmark

This benchmark compares BioRxnGen candidate-product recovery with feasible net secretion from the corresponding native AGORA2 models. It covers OMM12 in AF and APF, seven ASF co-cultures and four Rios Garza communities. The primary generation run uses strain-linked EC annotations and native substrates. ESP and the learned reranker are disabled. The GEM comparison uses an anaerobic rich-medium profile and requires at least 10% of maximum community growth.

| Location | Contents |
| --- | --- |
| [data](data) | 19 model identifiers, 13 community definitions, the assay panel, structure mappings and the exact generation worklists |
| [generation records](results/generation_records.zip) | Original candidate and attempt Parquet files. The EC and sequence-linked runs are stored separately |
| [native GEM records](results/native_gem_records.zip) | Original solver output, growth, exchange mappings and native intracellular products |
| [per-analyte scores](results/per_analyte.tsv) | Assay eligibility, enrichment and method-specific matches |
| [community scores](results/per_cohort.tsv) | Counts by community and condition |
| [study coverage](results/study_coverage.tsv) | Study-level results including the fourth study in the scope sensitivity analysis |

Each archive has a manifest with a SHA-256 for every member. The EC run generated 350,265 product rows from 10,422 contexts. The sequence-linked sensitivity run generated 67,780 rows from 833 inputs. The selected studies yielded 83 of 114 enriched observations for EC-constrained generation and 70 of 114 for native GEM. These denominators count condition–metabolite observations. D’hoe et al. are represented in the study-level sensitivity table.

## Reproduce

Run from the repository root. Install the requirements, then verify the archived results by recomputing matches from the original candidate products and solver output.

```bash
python -m pip install -r benchmarks/defined_communities/requirements.txt
python benchmarks/defined_communities/benchmark.py verify
```

For a rerun, the assay command fetches the original source files and rebuilds the assay panel. The model command extracts the required AGORA2 2.01 SBML members from the VMH archive. The generation command uses the [released input archive](https://github.com/Selenefreeze/BioRxnGen/releases/download/v1.0.0/model_and_generation_inputs.zip), including its directional rules and split assignments. Pass the archive path to the command below.

```bash
python benchmarks/defined_communities/benchmark.py assays --output output/community/assays
python benchmarks/defined_communities/benchmark.py fetch-models --output output/community/models
python benchmarks/defined_communities/benchmark.py generate --reference-archive model_and_generation_inputs.zip --output output/community/generation
python benchmarks/defined_communities/benchmark.py gem --models output/community/models --output output/community/gem
```

The [source manifest](data/source_data.json) records the assay files and checksums. The [model manifest](data/models.json) identifies the original SBML members and records the display-name encoding repair for VPI-5482. Raw study files and SBML are fetched from their sources.

## References

- Weiss, A. S. et al. *Nature Communications* **14**, 4780 (2023). [OMM12 data](https://doi.org/10.1038/s41467-023-40372-0)
- Medlock, G. L. et al. *Cell Systems* **7**, 245–257.e7 (2018). [ASF data](https://doi.org/10.1016/j.cels.2018.08.003)
- Rios Garza, D. et al. *Nature Communications* **17**, 326 (2026). [Batch culture data](https://doi.org/10.1038/s41467-025-67036-5)
- Heinken, A. et al. *Nature Biotechnology* **41**, 1320–1331 (2023). [AGORA2](https://doi.org/10.1038/s41587-022-01628-0)
- D’hoe, K. et al. *eLife* **7**, e37090 (2018). [Study scope sensitivity](https://doi.org/10.7554/eLife.37090)
