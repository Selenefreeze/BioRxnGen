# Defined-community metabolite benchmark

The benchmark contains four selected panels from three studies. Candidates were generated from native strain substrates and enzyme classifications, then ranked within each strain–EC–substrate input with the frozen Fig. 2 reranker. Native AGORA2 coverage is the fraction of measured structures with feasible net secretion under anaerobic rich-medium bounds. A model retained at least 10% of its maximum biomass growth for each secretion calculation.

| Panel | Target criterion | Distinct compounds |
| --- | --- | ---: |
| Gut bacterial monocultures | Above matched sterile-medium controls | 208 |
| Defined communities in gnotobiotic mice | Above matched germ-free controls | 181 |
| SIHUMIx consortium cultures | Detected in cultures | 99 |
| OMM¹² member monocultures | Detected in culture supernatants | 43 |

The two Han panels share one source study. A structure counts once per panel if it matches in an eligible condition. BioRxnGen coverage uses structural matches to generated products, and the native baseline uses feasible net secretion. The sensitivity table also compares the native intracellular product inventory and candidates that are element- and charge-balanced at their original ranks.

| Location | Contents |
| --- | --- |
| [data](data) | Assay observations and identities, condition and model mappings, exact generation worklists |
| [distinct compounds](results/coverage_unique_metabolites.tsv) | Coverage by panel, including top-1, 5, 10, 20, 30 and the full candidate set |
| [condition records](results/coverage_condition_records.tsv) | Condition–compound totals, with a [per-condition breakdown](results/coverage_by_condition.tsv) |
| [exchange and balance](results/coverage_sensitivity.tsv) | Exchange-mapped subsets, native products and balanced top-five candidates |
| [compound matches](results/per_metabolite.tsv) | Condition-level matches, ranks, candidate identifiers and native GEM hits |
| [original run records](results) | Candidate generation, frozen scores and original GEM solver outputs in checksummed archives |

## Reproduce

From the repository root after [installation](../../README.md#install-and-run):

```bash
python benchmarks/defined_communities/benchmark.py verify
```

This checks the archives, rescores 504,532 candidate records and rebuilds the coverage tables from the archived ranks. Candidate features must match exactly and rerun scores must agree within an absolute tolerance of 10⁻⁶. Small numerical differences between platforms can change the order of nearly tied candidates. The full rerun uses the archived generation worklists, the [release rule and context archive](https://github.com/Selenefreeze/BioRxnGen/releases/download/v1.0.0/model_and_generation_inputs.zip), and the original AGORA2 SBML files:

```bash
python -m pip install -r benchmarks/defined_communities/requirements.txt
python benchmarks/defined_communities/benchmark.py fetch-models --output output/community/models
python benchmarks/defined_communities/benchmark.py generate --reference-archive model_and_generation_inputs.zip --output output/community/generation
python benchmarks/defined_communities/benchmark.py score --generation output/community/generation --output output/community/ranked.parquet
python benchmarks/defined_communities/benchmark.py gem --models output/community/models --output output/community/gem
python benchmarks/defined_communities/benchmark.py summarize --ranked output/community/ranked.parquet --gem output/community/gem --output output/community/results
```

To rebuild assay inputs, obtain the files listed in the [source manifest](data/source_data.json) and save them under the indicated filenames in a local directory. The script checks file hashes and writes the processed observations, identities and condition definitions.

```bash
python benchmarks/defined_communities/prepare_assays.py --source path/to/source_files --output output/community/assays
```

## References

- Han, S., Van Treuren, W. et al. *Nature* **595**, 415–420 (2021). [Paper and source tables](https://doi.org/10.1038/s41586-021-03707-9)
- Farag, M. A. et al. *Journal of Advanced Research* **23**, 47–59 (2020). [Paper and supplementary data](https://doi.org/10.1016/j.jare.2020.01.001)
- Weiss, A. S. et al. *The ISME Journal* **16**, 1095–1109 (2022). [Paper](https://doi.org/10.1038/s41396-021-01153-z) · [Peak data](https://zenodo.org/records/6560725)
- Heinken, A. et al. *Nature Biotechnology* **41**, 1320–1331 (2023). [AGORA2](https://doi.org/10.1038/s41587-022-01628-0)
