"""Render all eight study figures from the source catalogs and reference results.

Usage from the package directory:
    python3 data/prepare_santorini_e_catalog.py data/catalog.txt data/santorini_e_catalog.csv
    python3 render_figures.py config.json --results outputs/results.json \
        --figure-dir generated/figures

Uses reference confidence intervals and Omori parameters, with catalog selection
and geometry computed by the shared analysis helpers. The overview map uses a
fixed display sample. Use analysis_pipeline.py for statistical resampling.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import render_residuals as G
import analysis_stats as S
import study_figures
from plot_style import (
    COLORS, LABELS, MARKERS, NAMES, STYLE,
    figure_heading, save_figure, style_axes,
)


FIGURE_NAMES = (
    "fig_filters.png", "fig_fmd.png", "fig_map_overlay.png",
    "fig_rate_omori.png", "fig_residual_maps.png",
    "fig_residuals_two_panel.png", "fig_supplementary_S1.png",
    "fig_time_stratified.png",
)


def _resolve_inputs(config, base):
    config = copy.deepcopy(config)
    for spec in config["catalogs"].values():
        path = Path(spec["path"])
        if not path.is_absolute():
            spec["path"] = str(base / path)
    for key, value in config.get("maps", {}).items():
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                config["maps"][key] = str(base / path)
    return config


def write_filter_figure(reference, figure_dir):
    """Plot the reference Q0-Q3 azimuths and event-resampling intervals."""
    filters = ("Q0", "Q1", "Q2", "Q3")
    x = np.arange(len(filters))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 4.9))
        fig.subplots_adjust(left=0.105, right=0.97, bottom=0.15, top=0.775)
        figure_heading(
            fig, "Orientation across quality filters",
            "PCA azimuth · 95% event-resampling confidence intervals")
        extrema = [45.0]
        for name, offset in zip(NAMES, (-0.08, 0.08)):
            values = [reference["pca"][name][key] for key in filters]
            azimuths = np.array([value["az"] for value in values])
            lower = np.array([value["ci_simple"][0] for value in values])
            upper = np.array([value["ci_simple"][1] for value in values])
            extrema.extend([float(lower.min()), float(upper.max())])
            ax.errorbar(
                x + offset, azimuths, yerr=[azimuths - lower, upper - azimuths],
                fmt=MARKERS[name], color=COLORS[name], markersize=5.7,
                elinewidth=1.35, capsize=3.5, label=LABELS[name])
        ax.axhline(45, color="#83929C", ls=(0, (4, 3)), lw=1,
                   label="45° NE–SW reference")
        ax.set_xticks(x, ["Q0", "Q1", "Q2", "Q3"])
        ax.set(xlabel="Quality filter", ylabel="PCA azimuth (° clockwise from north)",
               ylim=(min(extrema) - 0.6, max(extrema) + 0.8))
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, borderaxespad=0)
        style_axes(ax, grid="y")
        save_figure(fig, Path(figure_dir) / "fig_filters.png")


def render_figures(config_path, results_path, figure_dir):
    """Read reference results, verify Q1 geometry, and render all eight figures."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    reference = json.loads(Path(results_path).read_text())
    # Output location does not affect the analysis; all other settings must match.
    current_settings = {key: value for key, value in config.items() if key != "out_dir"}
    reference_settings = {key: value for key, value in reference["config"].items()
                          if key != "out_dir"}
    if current_settings != reference_settings:
        raise ValueError("Configuration must match the reference results except for out_dir")
    if config.get("primary_filter", "Q1") != "Q1":
        raise ValueError("The residual figures require the Q1 selection")
    resolved = _resolve_inputs(config, config_path.parent)
    catalogs = G.load_q1(resolved)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report = G.write_residual_figures(catalogs, figure_dir, reference=reference)
    pca = {name: S.pca_azimuth(data["x"].values, data["y"].values)
           for name, data in catalogs.items()}
    study_figures.make_figures(
        resolved, figure_dir, catalogs["SantoriniE"], catalogs["NLL"],
        pca["SantoriniE"], pca["NLL"], reference)
    write_filter_figure(reference, figure_dir)
    report["residual_all_events_shown"] = report.pop("all_events_shown")
    report["residual_random_draws"] = report.pop("random_draws")
    report["figures"] = list(FIGURE_NAMES)
    report["confidence_intervals_from_reference"] = True
    report["omori_parameters_from_reference"] = True
    report["statistical_resampling_performed"] = False
    report["map_display_sample_max_per_catalog"] = 6000
    report["map_display_sample_seed"] = 1
    report["config_comparison_ignored_keys"] = ["out_dir"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--results", default="outputs/results.json")
    parser.add_argument("--figure-dir", default="generated/figures")
    parser.add_argument("--verification-report", help="Optional JSON path for geometry and reference comparisons")
    args = parser.parse_args()
    report = render_figures(args.config, args.results, args.figure_dir)
    if args.verification_report:
        Path(args.verification_report).write_text(json.dumps(report, indent=2) + "\n")
    for name, item in report["catalogs"].items():
        print(f"{name}: n={item['n']}, az={item['azimuth_deg']:.10f}, "
              f"reference max difference={item['reference_numeric_max_abs_difference']}")
    print(f"Eight figures written to {args.figure_dir}")


if __name__ == "__main__":
    main()
