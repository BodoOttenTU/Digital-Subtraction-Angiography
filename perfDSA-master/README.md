# AIF variability in perfDSA

BEP project (TU/e, 2026) on the arterial input function (AIF) in automatic DSA perfusion imaging, built on
[perfDSA](https://github.com/RuishengSu/perfDSA) (Su et al., IJCARS 2025). The project asks how much the AIF
changes with the sampling segment (ICA-top vs ICA-full, SQ1), the projection view (frontal vs lateral, SQ2),
how much of its variance is patient vs acquisition (SQ3), and how much that matters for the perfusion maps (SQ4).

Data: PRAISE biplane DSA (batch 1, 30 patients, and batch 2, 38 patients; Siemens, AIS/EVT). Patient data,
model weights and all outputs are **not** in the repository (see `.gitignore`).

## Installation

Tested with Python 3.14 on Windows (conda env), CUDA GPU for the perfDSA step only.

```bash
conda create -n BEP python=3.14
conda activate BEP
pip install -r requirements.txt
```

Put the two segmentation models from perfDSA in `models/`:

```text
models/best_model_ica_top.pt   ICA-top segmentation (full torch model)
models/best_model_ica.pt       full-ICA segmentation (state_dict, only for --ica-full model)
```

Put the DICOM data next to this folder (paths are set in `run_all.py`):

```text
Digital-Subtraction-Angiography/
├── PRAISE/            batch 1, one folder per patient with the DSA runs
├── PRAISE_BATCH_2/    batch 2
└── perfDSA-master/    this code
```

## Reproducing the results

```bash
python run_all.py                    # from postprocess on (minutes); re-uses the saved perfDSA runs
python run_all.py --from raw_model   # everything from the DICOMs (GPU, a few hours)
python run_all.py --full             # also recompute the truncation simulation and AIF swap
python run_all.py --only sq2 sq2_slides
```

`--from raw_model` is needed on a fresh checkout without `results/data/raw_model` and `raw_grown`.
Every script can also be run on its own (`python scripts/sq2_plane.py`); folder locations are in
`scripts/paths.py`. Steps in order:

| Stage | Step | Script | Output |
|---|---|---|---|
| Data -> maps | `raw_model`, `raw_grown` | `run_dataset.py` (perfDSA per run, ICA-top + ICA-full AIF) | `results/data/raw_*` |
| Filtered data | `postprocess` | `scripts/postprocess_combined.py` (three AIF variants) | `results/data/npz`, `aif_curves.csv` |
| | `select` | `scripts/select_runs.py` (one run per patient) | `selection.csv` |
| | `build_features` | `scripts/build_features.py` | `features*.csv` |
| | `qc`, `taxonomy` | `scripts/aif_qc.py`, `scripts/aif_taxonomy.py` (trust flag) | `results/data/qc/` |
| | `truncation` | `scripts/aif_truncation.py` | `results/data/qc/truncation/` |
| | `summary` | `scripts/data_summary.py` (cohort table, flowchart) | `results/data/cohort/` |
| | `plane`, `features` | `scripts/plane_difference.py`, `scripts/aif_feature_set.py` | `sq2/plane_difference/`, `data/features/` |
| SQ1 segment | `sq1` | `scripts/sq1_full_vs_top.py` | `results/sq1/` |
| SQ2 view | `sq2` | `scripts/sq2_plane.py` | `results/sq2/` |
| SQ3 variance | `sq3`, `sq3_all_variants`, `sq3_component_ci` | `scripts/sq3_variance.py`, `scripts/sq3_component_ci.py` | `results/sq3/` |
| SQ4 AIF swap | `sq4` | `scripts/sq4_aif_swap.py` | `results/sq4/` |
| Figures | `sq1_figures`, `sq*_slides`, `sq3_figure` | `scripts/sq1_summary_figure.py`, `scripts/sq*_slides.py`, `scripts/sq3_*figure*.py` | `results/sq*/`, `slides/` |

Numbers in the report come from the csv files and the `table_*.tex` files in each output folder
(e.g. `results/sq1/table_sq1.tex`, `results/data/qc/table_i_qc/`). AIF variant names: `ica_top` = ICA-top,
`ica_full_grown` = ICA-full in the report; `ica_full_model` is only used in the SQ3 sensitivity run.

Figures for the report and the final presentation that are not part of `run_all.py` (run by hand):
`scripts/methods_figures.py` (report Methods figures -> `report/figures/`), `scripts/presentation_*.py`,
`scripts/pipeline_*.py`, `scripts/qc_examples.py`, `scripts/qc_synthetic_examples.py`, `scripts/outline_road.py`
(-> `results/presentation/`).

## Layout

```text
run_all.py                       runs the whole pipeline in order
run_dataset.py, DSASequence.py   perfDSA on the DICOMs (DSASequence from the original perfDSA code)
unet/                            U-Net used by the segmentation models
scripts/                         post-processing, QC, SQ1-SQ4, figures (paths.py: output folders)
models/                          segmentation weights (not in git)
results/                         pipeline output (not in git)
  data/                          perfDSA runs (raw_model, raw_grown), npz bundles, tables,
                                 cohort/ (Table I, Fig. 1), qc/ (QC, taxonomy, truncation), features/
  sq1/ sq2/ sq3/ sq4/            statistics (csv), LaTeX tables, figures and slides per sub-question
                                 (sq2/plane_difference; sq3/top_model_grown is the three-variant sensitivity run)
  presentation/                  figures and animations for the final presentation
report/                          report figures (not in git)
```

## Known issues

- `DSASequence` assumes a background grey value of 170, while PRAISE frames sit near 188, so tissue curves
  and AIFs are not fully baseline-subtracted. This mainly affects CBV/MTT in SQ4; not fixed in the pipeline.
- QC thresholds (`TH` in `scripts/aif_qc.py`, e.g. truncation level 0.20) were tuned on this cohort; the
  simulation in `aif_truncation.py` gives a literature-based alternative of 0.15 that was not adopted.

## Citation

```bibtex
@article{su2025perfdsa,
  title={perfDSA: Automatic perfusion imaging in cerebral digital subtraction angiography},
  author={Su, Ruisheng and van der Sluijs, P Matthijs and Marc, Flavius-Gabriel and Te Nijenhuis, Frank and Cornelissen, Sandra AP and Roozenbeek, Bob and van Zwam, Wim H and van der Lugt, Aad and Ruijters, Danny and Pluim, Josien and others},
  journal={International journal of computer assisted radiology and surgery},
  volume={20},
  number={6},
  pages={1195--1203},
  year={2025},
  publisher={Springer}
}
```
