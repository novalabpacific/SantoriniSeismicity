"""Q1 residual maps and model profiles for Supplementary Figure S1.

Uses the shared Q1 selection, projection, PCA, and geometry_models helpers.
The fits are deterministic; --verify checks them against reference results.
Run from the package directory:

    python3 data/prepare_santorini_e_catalog.py data/catalog.txt data/santorini_e_catalog.csv
    python3 render_residuals.py config.json --figure-dir generated/figures \
        --verify outputs/results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

import analysis_stats as S
from plot_style import (
    COLORS, LABELS, MODEL_COLORS, MUTED, NAMES, STYLE,
    figure_heading, save_figure, style_axes,
)


def load_q1(config):
    """Load both catalogs using the configured time window and Q1 filters."""
    import pandas as pd
    import analysis_pipeline as P
    t1, t2 = map(pd.Timestamp, config["window"])
    q = {}
    for name in NAMES:
        raw = P.load_catalog(config["catalogs"][name], config)
        window = raw[(raw["t"] >= t1) & (raw["t"] < t2)].reset_index(drop=True)
        projected = P.add_xy(window, config["origin"]["lat0"],
                             config["origin"]["lon0"])
        q[name] = P.apply_filter(projected, config["quality"][name]["Q1"])
    return q


def _numeric_differences(actual, expected, prefix=""):
    differences = {}
    for key, value in expected.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            differences.update(_numeric_differences(actual[key], value, path))
        elif isinstance(value, (int, float)):
            differences[path] = abs(float(actual[key]) - value)
    return differences


def fit_catalog(q):
    """Fit the straight, quadratic, and optimized continuous hinge models."""
    x, y = q["x"].to_numpy(float), q["y"].to_numpy(float)
    pca = S.pca_azimuth(x, y)
    s, t = S.along_perp(x, y, pca["mu"], pca["v"])
    models = S.geometry_models(s, t)
    n = len(s)
    X2 = np.column_stack([np.ones(n), s, s ** 2])
    b2 = np.linalg.lstsq(X2, t, rcond=None)[0]
    c = models["piecewise_break_s_km"]
    Xp = np.column_stack([np.ones(n), s, np.maximum(s - c, 0)])
    bp = np.linalg.lstsq(Xp, t, rcond=None)[0]
    intercept = float(t.mean())
    fitted_ic = {
        "line": S._gauss_ic(t - intercept, 2, n),
        "quadratic": S._gauss_ic(t - X2 @ b2, 4, n),
        "piecewise": S._gauss_ic(t - Xp @ bp, 5, n),
    }
    fit_diff = _numeric_differences(
        fitted_ic, {key: models[key] for key in fitted_ic})
    if max(fit_diff.values()) > 1e-9:
        raise ValueError("Displayed fits disagree with geometry_models statistics")
    return dict(x=x, y=y, s=s, t=t, n=n, pca=pca, models=models,
                line_intercept=intercept, quadratic_coefficients=b2,
                piecewise_coefficients=bp, break_s=c,
                displayed_fit_max_abs_difference=max(fit_diff.values()))


def _limits(fits):
    x = np.concatenate([f["x"] for f in fits.values()])
    y = np.concatenate([f["y"] for f in fits.values()])
    s = np.concatenate([f["s"] for f in fits.values()])
    t = np.concatenate([f["t"] for f in fits.values()])
    # Include the full coordinate range of both catalogs.
    def padded(a, step=5):
        return [float(np.floor(a.min() / step) * step - 2.5),
                float(np.ceil(a.max() / step) * step + 2.5)]
    xx, yy = padded(x), padded(y)
    # A square common map extent supports equal geographic scales in both panels.
    span = max(xx[1] - xx[0], yy[1] - yy[0])
    xx = [float(np.mean(xx) - span / 2), float(np.mean(xx) + span / 2)]
    yy = [float(np.mean(yy) - span / 2), float(np.mean(yy) + span / 2)]
    bound = float(max(5, np.ceil(np.max(np.abs(t)) / 5) * 5))
    return dict(map_xlim=xx, map_ylim=yy, profile_xlim=padded(s),
                profile_ylim=padded(t), color_limits=[-bound, bound])


def _plot_map(ax, f, name, letter, limits):
    vmin, vmax = limits["color_limits"]
    scatter = ax.scatter(f["x"], f["y"], c=f["t"], cmap="RdBu_r",
                         norm=Normalize(vmin, vmax), s=4.0, alpha=0.88,
                         linewidths=0, rasterized=True, zorder=2)
    v, mu = f["pca"]["v"], f["pca"]["mu"]
    reach = 2 * max(np.ptp(limits["map_xlim"]), np.ptp(limits["map_ylim"]))
    endpoints = mu + np.array([-reach, reach])[:, None] * v
    ax.plot(endpoints[:, 0], endpoints[:, 1], ls=(0, (5, 3)),
            color=MODEL_COLORS["line"], lw=1.2, zorder=3)
    ax.set(xlim=limits["map_xlim"], ylim=limits["map_ylim"],
           xlabel="East (km)", ylabel="North (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{letter}  {LABELS[name]}", loc="left", pad=9,
                 color=COLORS[name], fontweight="semibold")
    ax.text(0.98, 1.028, f"n = {f['n']:,}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=MUTED)
    ax.text(0.035, 0.045, f"PCA {f['pca']['az']:.1f}°", transform=ax.transAxes,
            fontsize=9, color=MODEL_COLORS["line"],
            bbox={"fc": "white", "ec": "none", "alpha": 0.9, "pad": 2})
    style_axes(ax, boxed=True)
    return scatter


def _plot_profile(ax, f, name, letter, limits):
    s, t = f["s"], f["t"]
    ax.scatter(s, t, s=3.5, alpha=0.32, color=COLORS[name], linewidths=0,
               rasterized=True, zorder=2)
    # Curves are displayed only over the observed along-axis range.
    grid = np.unique(np.r_[np.linspace(s.min(), s.max(), 600), f["break_s"]])
    b2, bp = f["quadratic_coefficients"], f["piecewise_coefficients"]
    ax.axhline(f["line_intercept"], color=MODEL_COLORS["line"],
               ls=(0, (5, 3)), lw=1.2, zorder=3)
    ax.plot(grid, b2[0] + b2[1] * grid + b2[2] * grid ** 2,
            color=MODEL_COLORS["quadratic"], lw=1.7, zorder=4)
    ax.plot(grid, bp[0] + bp[1] * grid + bp[2] * np.maximum(grid - f["break_s"], 0),
            color=MODEL_COLORS["piecewise"], lw=1.7, zorder=5)
    ax.set(xlim=limits["profile_xlim"], ylim=limits["profile_ylim"],
           xlabel="Along-axis distance, s (km)", ylabel="Signed residual, t (km)")
    ax.set_title(f"{letter}  {LABELS[name]}", loc="left", pad=9,
                 color=COLORS[name], fontweight="semibold")
    ax.text(0.98, 1.028, f"n = {f['n']:,}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=MUTED)
    d = f["models"]["delta_aic_vs_line"]
    ax.text(0.035, 0.96,
            f"ΔAIC: quadratic {d['quadratic']:,.0f}; piecewise {d['piecewise']:,.0f}",
            transform=ax.transAxes, va="top", fontsize=8.1, color=MUTED,
            bbox={"fc": "white", "ec": "none", "alpha": 0.9, "pad": 2})
    style_axes(ax)


def _legend(fig, y=0.025, x=0.51):
    handles = [Line2D([], [], color=MODEL_COLORS["line"], ls=(0, (5, 3)), lw=1.2),
               Line2D([], [], color=MODEL_COLORS["quadratic"], lw=1.7),
               Line2D([], [], color=MODEL_COLORS["piecewise"], lw=1.7)]
    fig.legend(handles, ["Straight PCA axis", "Quadratic", "Continuous piecewise"],
               loc="lower center", bbox_to_anchor=(x, y), ncol=3,
               frameon=False, columnspacing=2, handlelength=2.5, fontsize=9.3)


def _save(fig, path):
    save_figure(fig, path)


def write_residual_figures(q_catalogs, figure_dir, reference=None):
    """Write three graphics and return their geometry and verification results."""
    target = Path(figure_dir)
    target.mkdir(parents=True, exist_ok=True)
    fits = {name: fit_catalog(q_catalogs[name]) for name in NAMES}
    limits = _limits(fits)
    report = {"method": "Q1 PCA residuals; optimized continuous hinge from geometry_models",
              "all_events_shown": True, "random_draws": 0,
              "limits": limits, "catalogs": {}}
    for name, f in fits.items():
        diffs = {}
        if reference is not None:
            rp = reference["pca"][name]["Q1"]
            diffs.update(_numeric_differences(
                {"n": f["n"], "az": f["pca"]["az"], "ev1_frac": f["pca"]["ev1_frac"]},
                {key: rp[key] for key in ("n", "az", "ev1_frac")}, "pca"))
            diffs.update(_numeric_differences(f["models"],
                         reference["geometry"][name]["models"], "models"))
            if max(diffs.values()) > 1e-9:
                raise ValueError(f"{name}: computed geometry differs from reference results")
        report["catalogs"][name] = {
            "n": f["n"], "azimuth_deg": f["pca"]["az"],
            "ev1_frac": f["pca"]["ev1_frac"],
            "median_absolute_residual_km": float(np.median(np.abs(f["t"]))),
            "p90_absolute_residual_km": float(np.percentile(np.abs(f["t"]), 90)),
            "ranges_km": {k: [float(f[k].min()), float(f[k].max())] for k in ("x", "y", "s", "t")},
            "models": f["models"],
            "quadratic_coefficients": f["quadratic_coefficients"].tolist(),
            "piecewise_coefficients": f["piecewise_coefficients"].tolist(),
            "displayed_fit_max_abs_difference": f["displayed_fit_max_abs_difference"],
            "reference_numeric_max_abs_difference": max(diffs.values()) if diffs else None,
        }
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.85))
        fig.subplots_adjust(left=0.085, right=0.975, bottom=0.235, top=0.775, wspace=0.29)
        figure_heading(
            fig, "Along-axis residual structure",
            "All Q1 events · Deterministic model fits · ΔAIC relative to the straight PCA axis",
            left=0.085)
        for ax, name, letter in zip(axes, NAMES, "AB"):
            _plot_profile(ax, fits[name], name, letter, limits)
        _legend(fig, 0.045, x=0.53)
        _save(fig, target / "fig_residuals_two_panel.png")

        fig, axes = plt.subplots(1, 2, figsize=(9.3, 5.15))
        fig.subplots_adjust(left=0.085, right=0.825, bottom=0.18, top=0.79, wspace=0.28)
        figure_heading(
            fig, "Signed residuals from the PCA axes",
            "All Q1 events · Common east–north frame and full residual color scale", left=0.085)
        for ax, name, letter in zip(axes, NAMES, "AB"):
            scatter = _plot_map(ax, fits[name], name, letter, limits)
        cax = fig.add_axes([0.872, 0.24, 0.016, 0.48])
        fig.colorbar(scatter, cax=cax, label="Signed residual, t (km)",
                     ticks=np.linspace(*limits["color_limits"], 5))
        fig.text(0.085, 0.055,
                 "Positive residuals lie northwest of the northeast-oriented PCA axis.",
                 fontsize=8.6, color=MUTED)
        _save(fig, target / "fig_residual_maps.png")

        fig = plt.figure(figsize=(9.3, 9.0))
        figure_heading(
            fig, "Residual geometry and model comparison",
            "All Q1 events · Shared map extent and signed-residual scale", left=0.085)
        gs = fig.add_gridspec(2, 2, left=0.085, right=0.855, bottom=0.17,
                             top=0.85, hspace=0.28, wspace=0.29,
                             height_ratios=[1.30, 1.0])
        for col, name in enumerate(NAMES):
            scatter = _plot_map(fig.add_subplot(gs[0, col]), fits[name], name,
                                "AB"[col], limits)
            _plot_profile(fig.add_subplot(gs[1, col]), fits[name], name,
                          "CD"[col], limits)
        cax = fig.add_axes([0.895, 0.565, 0.014, 0.245])
        fig.colorbar(scatter, cax=cax, label="Signed residual, t (km)",
                     ticks=np.linspace(*limits["color_limits"], 5))
        _legend(fig, 0.085, x=0.485)
        fig.text(0.085, 0.04,
                 "ΔAIC is relative to the straight PCA axis; positive t lies northwest of that axis.",
                 fontsize=8.6, color=MUTED)
        _save(fig, target / "fig_supplementary_S1.png")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--figure-dir", default="generated/figures")
    parser.add_argument("--verify", help="Reference results.json for geometry verification")
    parser.add_argument("--verification-report", help="Optional JSON path for geometry and reference comparisons")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    # Resolve relative catalog paths from the configuration file's directory.
    for spec in config["catalogs"].values():
        path = Path(spec["path"])
        if not path.is_absolute():
            spec["path"] = str(config_path.parent / path)
    reference = json.loads(Path(args.verify).read_text()) if args.verify else None
    report = write_residual_figures(load_q1(config), args.figure_dir, reference)
    if args.verification_report:
        Path(args.verification_report).write_text(json.dumps(report, indent=2) + "\n")
    for name, item in report["catalogs"].items():
        print(f"{name}: n={item['n']}, az={item['azimuth_deg']:.10f}, "
              f"reference max difference={item['reference_numeric_max_abs_difference']}")
    print(f"Three residual figures written to {args.figure_dir}")


if __name__ == "__main__":
    main()
