# Santorini-Amorgos seismicity analysis

Code, earthquake catalogs, numerical results, and figures for **Cross-Catalog Evidence for a Consistent NE-SW Seismicity Orientation in the 2025 Santorini-Amorgos Unrest**.

The analysis compares spatial orientation, geometry, temporal behavior, and magnitude distributions in two earthquake catalogs. **SantoriniE** labels the Fountoulakis–Evangelidis catalog; **NLL** labels the NLL-SSST-coherence relocated catalog. Dataset citations and map sources are listed in [DATA_SOURCES.md](DATA_SOURCES.md).

Reference results are in `outputs/`, and the eight figures are in `figures/`. The workflows below write new results and figures to `generated/`.

## Setup

Use Python 3.12 and run commands from the project directory.

```bash
python3 -m pip install -r requirements.txt
python3 verify_package.py
```

`requirements.txt` specifies the dependency versions. `verify_package.py` checks the project files against `MANIFEST.sha256`; `checksums.sha256` contains the two source-catalog checksums. The manifest covers every release file except itself. On systems with `sha256sum`, the equivalent integrity check is:

```bash
sha256sum -c MANIFEST.sha256
```

## Render all eight figures from reference results

This workflow converts the SantoriniE catalog to CSV, recomputes Q1 geometry, and checks it against the reference values. It uses the stored confidence intervals, temporal summaries, and Omori parameters without repeating statistical resampling.

```bash
python3 data/prepare_santorini_e_catalog.py data/catalog.txt data/santorini_e_catalog.csv
python3 render_figures.py config.json \
  --results outputs/results.json \
  --figure-dir generated/figures \
  --verification-report generated/graphics_verification.json
```

The CSV conversion preserves the source columns and values. `data/santorini_e_catalog.csv` is an intermediate file created by the workflow.

The renderer requires the analysis configuration to match the configuration in the selected results file. It excludes `out_dir` from this comparison because that setting controls only where outputs are written.

## Run the full analysis

Run the stages in order. The supplementary stage reads and extends the primary results, and the rendering stage creates the complete figure set from those results.

```bash
python3 data/prepare_santorini_e_catalog.py data/catalog.txt data/santorini_e_catalog.csv
python3 analysis_pipeline.py config.json
python3 supplementary_analyses.py config.json
python3 render_figures.py config.json \
  --results generated/results.json \
  --figure-dir generated/figures \
  --verification-report generated/graphics_verification.json
```

The analysis writes `generated/results.json`, `generated/manuscript_values.csv`, `generated/summary.md`, and diagnostic figures. The rendering stage writes all eight figures to `generated/figures/`, including the quality-filter comparison. Statistical resampling controls the computation time; the configuration uses random seed `20260822`.

## Render residual figures

To render the residual maps and model profiles:

```bash
python3 data/prepare_santorini_e_catalog.py data/catalog.txt data/santorini_e_catalog.csv
python3 render_residuals.py config.json \
  --verify outputs/results.json \
  --figure-dir generated/figures \
  --verification-report generated/residual_verification.json
```

This produces `fig_residual_maps.png`, `fig_residuals_two_panel.png`, and `fig_supplementary_S1.png`. Maps show every Q1 event in a common local east–north frame, with equal map scales, common limits, and a shared full-range color scale. The signed residual is `t = (X - mu) dot (-v_y, v_x)` relative to the northeast-oriented PCA axis; positive residuals lie northwest of that axis.

Profiles compare the straight PCA axis, a quadratic, and the continuous one-break model `t = a + b*s + d*max(s-c, 0)`. The break minimizes AIC over 17 candidate quantiles from the 10th through 90th percentiles of along-axis position. Curves span the observed range.

## Analysis conventions

* The common UTC window is **\[2025-01-27T19:00:00, 2025-03-04T23:59:59)**: the start is included and the end is excluded. The local projection origin is 36.605514° N, 25.646090° E.
* Q0–Q3 quality thresholds, the primary Q1 selection, bootstrap settings, and input paths are defined in `config.json`.
* Temporal tests use daily medians from days containing at least 10 selected events. Primary Theil–Sen slope intervals use a residual moving-block bootstrap; Spearman associations use block permutation, with 10,000 resamples for each primary procedure. A block contains **five consecutive retained daily medians**, which need not be five consecutive calendar days. Slopes use actual elapsed calendar-day coordinates.
* Bonferroni and Benjamini–Hochberg adjustments cover four primary block-permutation p values. The primary trend criterion requires a block interval excluding zero and a Bonferroni-adjusted p value below 0.05.
* Event matching uses greedy one-to-one assignment within 1.0 s and examines three nearby time candidates. This restricted search can miss eligible matches in dense catalogs. Deming regressions use a catalog-level error-variance ratio from reported location errors.
* Paired-day orientation intervals resample the same calendar days for both catalogs, using 5,000 resamples. Three-dimensional line/plane intervals use 300 day-cluster resamples.
* Early-axis late-slope tests use the same UTC split for both catalogs. The early interval starts at `config.json`'s full window-start timestamp and ends after `early_axis_days` days: **\[2025-01-27T19:00:00, 2025-02-10T19:00:00)** with the supplied 14-day setting. The late interval includes events at or after **2025-02-10T19:00:00**, within the common analysis window.
* Supplementary analyses include completeness and fixed-magnitude thresholds, event-level orientation differences, early-axis late-slope tests, and sample-size sensitivity. Some supplementary resampling counts and thresholds are specified in `supplementary_analyses.py`.
* The overview map uses the configured `map_bbox` and a fixed sample of at most 6,000 events per catalog with display seed 1. Residual maps show all Q1 events. Contextual map layers do not enter the numerical seismicity analyses.
* Figures use a shared visual style and 300 dpi PNG output. SantoriniE is warm orange and NLL-SSST is blue wherever color identifies a catalog; residual maps use a common diverging scale for signed residuals.

## Project structure

| Path | Purpose |
| --- | --- |
| `config.json` | Input paths, quality filters, analysis settings, projection origin, and seed |
| `analysis_pipeline.py` | Primary analysis, statistical comparisons, and diagnostic figures |
| `analysis_stats.py` | Statistical and geometric functions |
| `supplementary_analyses.py` | Sensitivity analyses, value tables, and residual figures |
| `render_figures.py` | Complete eight-figure rendering workflow |
| `render_residuals.py` | Residual maps and model-comparison profiles |
| `study_figures.py` | Shared study-figure functions |
| `plot_style.py` | Shared figure colors and formatting |
| `verify_package.py` | File integrity verification |
| `data/catalog.txt` | SantoriniE source catalog |
| `data/Santorini_catalog.csv` | NLL-SSST-coherence source catalog |
| `data/prepare_santorini_e_catalog.py` | SantoriniE CSV conversion |
| `maps/` | Coastline, faults, Kolumbo marker, modeled-source traces, and bathymetric context |
| `outputs/results.json` | Reference numerical results and analysis configuration |
| `outputs/manuscript_values.csv` | Numerical value table |
| `figures/` | Eight analysis figures |
| `requirements.txt` | Python dependency versions |
| `checksums.sha256`, `MANIFEST.sha256` | Source-catalog and release-file checksums |
| `DATA_SOURCES.md`, `LICENSES.md`, `LICENSE` | Dataset citations, licensing details, and code license |

`generated/` and `data/santorini_e_catalog.csv` are created by the workflows above and excluded from Git tracking.

## Figure index

| Figure | Contents |
| --- | --- |
| [fig_supplementary_S1.png](figures/fig_supplementary_S1.png) | Common-frame signed-residual maps (A/B) and model-comparison profiles (C/D) for both Q1 catalogs |
| [fig_residual_maps.png](figures/fig_residual_maps.png) | Residual maps with PCA axes, common extent, and a full-range color scale |
| [fig_residuals_two_panel.png](figures/fig_residuals_two_panel.png) | Along-axis profiles with straight, quadratic, and continuous piecewise fits |
| [fig_map_overlay.png](figures/fig_map_overlay.png) | Catalog samples and PCA axes with coastline, fault, vent, bathymetric, and modeled-source context |
| [fig_filters.png](figures/fig_filters.png) | PCA azimuths and event-resampling intervals across Q0–Q3 |
| [fig_fmd.png](figures/fig_fmd.png) | Frequency–magnitude distributions and completeness diagnostics |
| [fig_time_stratified.png](figures/fig_time_stratified.png) | PCA azimuths in configured seven-day windows |
| [fig_rate_omori.png](figures/fig_rate_omori.png) | Daily rates and modified-Omori fits after the largest selected event |

## Licensing and attribution

The analysis code is licensed under [MIT](LICENSE). Earthquake catalogs and map layers have separate source attributions and terms, described in [LICENSES.md](LICENSES.md). The exact source and license of the coastline layer and the redistribution terms for the approximate modeled-source traces are not specified. Cite the dataset and publication records in [DATA_SOURCES.md](DATA_SOURCES.md) when reusing their data or derived figures.
