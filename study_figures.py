"""Overview, orientation, magnitude, and daily-rate figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

import analysis_stats as S
from plot_style import (
    COLORS, INK, LABELS, MARKERS, MUTED, NAMES, STYLE,
    figure_heading, save_figure, style_axes,
)


def _map_segments(path, bbox, lat0, lon0):
    from analysis_pipeline import read_gmt_xy

    west, east, south, north = bbox
    segments = []
    for segment in read_gmt_xy(path):
        if (segment[:, 0].max() < west or segment[:, 0].min() > east
                or segment[:, 1].max() < south or segment[:, 1].min() > north):
            continue
        x, y = S.project_equirect(segment[:, 1], segment[:, 0], lat0, lon0)
        segments.append(np.column_stack([x, y]))
    return segments


def _overview(cfg, target, catalogs, pca):
    lat0, lon0 = cfg["origin"]["lat0"], cfg["origin"]["lon0"]
    bbox = cfg.get("map_bbox", [25.15, 26.10, 36.30, 36.85])
    west, east, south, north = bbox
    xx, yy = S.project_equirect(
        np.array([south, south, north, north]),
        np.array([west, east, east, west]), lat0, lon0)
    xlim, ylim = (float(xx.min()), float(xx.max())), (float(yy.min()), float(yy.max()))
    maps = cfg.get("maps", {})
    fig = plt.figure(figsize=(9.6, 6.0))
    ax = fig.add_axes([0.085, 0.155, 0.555, 0.675])
    figure_heading(
        fig, "Seismicity and regional context",
        "Q1 catalogs · PCA orientation axes · Approximate modeled-source traces", left=0.085)
    context_handles, context_labels = [], []
    layers = (
        ("bathy", "#C0CBD2", 0.65, 1, "200 m boundary, generalized\nNatural Earth 1:10M"),
        ("coast", "#647785", 0.85, 2, "Coastline"),
        ("faults", "#AA8D6E", 0.85, 3, "NOAFAULTS v6.0"),
        ("dike_traces", "#AD4F65", 1.65, 6,
         "Approximate modeled-source traces\nLomax et al. (2025)"),
    )
    for key, color, width, order, label in layers:
        if not maps.get(key):
            continue
        segments = _map_segments(maps[key], bbox, lat0, lon0)
        if segments:
            ax.add_collection(LineCollection(
                segments, colors=color, linewidths=width, zorder=order))
            context_handles.append(Line2D([], [], color=color, lw=width))
            context_labels.append(label)

    catalog_handles, catalog_labels = [], []
    for name in NAMES:
        data, orientation = catalogs[name], pca[name]
        sample = data.sample(min(6000, len(data)), random_state=1)
        ax.scatter(sample["x"], sample["y"], s=3.4, alpha=0.34, marker=MARKERS[name],
                   color=COLORS[name], linewidths=0, rasterized=True, zorder=4)
        along = np.array([-25, 25])
        axis = orientation["mu"][None, :] + along[:, None] * orientation["v"][None, :]
        linestyle = "-" if name == "SantoriniE" else (0, (5, 2))
        ax.plot(axis[:, 0], axis[:, 1], color="white", lw=3.1, zorder=7)
        ax.plot(axis[:, 0], axis[:, 1], color=COLORS[name],
                ls=linestyle, lw=1.8, zorder=8)
        catalog_handles.extend([
            Line2D([], [], ls="none", marker=MARKERS[name], ms=4.8,
                   color=COLORS[name], markeredgewidth=0),
            Line2D([], [], color=COLORS[name], ls=linestyle, lw=1.8),
        ])
        catalog_labels.extend([
            f"{LABELS[name]} events",
            f"{LABELS[name]} PC1 · {orientation['az']:.1f}°",
        ])

    if maps.get("kolumbo"):
        kx, ky = S.project_equirect(36.52304, 25.48632, lat0, lon0)
        ax.plot(kx, ky, marker="^", ms=6.5, color=INK, zorder=9)
        xspan, yspan = np.ptp(xlim), np.ptp(ylim)
        label_x = np.clip(kx - 0.19 * xspan, xlim[0] + 0.05 * xspan, xlim[1] - 0.1 * xspan)
        label_y = np.clip(ky - 0.23 * yspan, ylim[0] + 0.08 * yspan, ylim[1] - 0.08 * yspan)
        ax.annotate(
            "Kolumbo", (kx, ky), xytext=(label_x, label_y), fontsize=9,
            ha="center", va="center", color=INK, zorder=10,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.94},
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": INK,
                        "connectionstyle": "angle3,angleA=0,angleB=80"})
        context_handles.append(Line2D([], [], ls="none", marker="^", ms=5.5, color=INK))
        context_labels.append("Kolumbo")

    ax.set(xlim=xlim, ylim=ylim, xlabel="East (km)", ylabel="North (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    style_axes(ax, boxed=True)
    fig.legend(
        catalog_handles, catalog_labels, loc="upper left", bbox_to_anchor=(0.69, 0.817),
        borderaxespad=0, title="Catalogs & orientation", title_fontsize=10,
        fontsize=9, labelspacing=1.05)
    fig.legend(
        context_handles, context_labels, loc="upper left", bbox_to_anchor=(0.69, 0.535),
        borderaxespad=0, title="Map context", title_fontsize=10,
        fontsize=8.7, labelspacing=1.05)
    fig.text(
        0.085, 0.05,
        "Display sample: up to 6,000 events per catalog. Map layers provide spatial context.",
        fontsize=8.6, color=MUTED)
    save_figure(fig, target / "fig_map_overlay.png")


def _orientation(cfg, target, results):
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    fig.subplots_adjust(left=0.105, right=0.97, bottom=0.165, top=0.775)
    days = cfg.get("strat_days", 7)
    figure_heading(
        fig, "Orientation through time",
        f"{days}-day windows · PCA azimuth · 95% day-cluster confidence intervals")
    starts = set()
    for name in NAMES:
        windows = results["geometry"][name]["time_stratified"]
        if not windows:
            continue
        dates = pd.to_datetime([window["start"] for window in windows])
        starts.update(dates)
        offset = pd.Timedelta(days=-0.4 if name == "SantoriniE" else 0.4)
        azimuth = [window["az"] for window in windows]
        lower = [abs(S.axial_diff(window["az"], window["ci"][0])) for window in windows]
        upper = [abs(S.axial_diff(window["ci"][1], window["az"])) for window in windows]
        ax.errorbar(dates + offset, azimuth, yerr=[lower, upper],
                    fmt=MARKERS[name], color=COLORS[name], ms=5.7,
                    elinewidth=1.35, capsize=3.5, label=LABELS[name])
    ax.axhline(45, color="#83929C", ls=(0, (4, 3)), lw=1,
               label="45° NE–SW reference")
    dates = sorted(starts)
    if dates:
        ax.set_xticks(dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_xlim(dates[0] - pd.Timedelta(days=2.6), dates[-1] + pd.Timedelta(days=2.6))
    years = sorted({date.year for date in dates})
    year_label = f" ({years[0]})" if len(years) == 1 else ""
    ax.set(xlabel=f"Window start{year_label}", ylabel="PCA azimuth (° clockwise from north)")
    ax.margins(y=0.10)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, borderaxespad=0)
    style_axes(ax, grid="y")
    save_figure(fig, target / "fig_time_stratified.png")


def _magnitude(target, catalogs, results):
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    fig.subplots_adjust(left=0.105, right=0.97, bottom=0.15, top=0.775)
    figure_heading(
        fig, "Frequency–magnitude distributions",
        "Q1 events · Vertical guides mark Mc = MAXC + 0.2")
    max_count = 1
    for name in NAMES:
        data = catalogs[name]
        if not data["mag"].notna().any():
            continue
        magnitude = np.sort(data["mag"].dropna().values)
        cumulative = np.arange(len(magnitude), 0, -1)
        max_count = max(max_count, len(magnitude))
        ax.semilogy(magnitude, cumulative, ".", ms=2.4,
                    color=COLORS[name], alpha=0.82, label=LABELS[name])
        mc = results["magnitude_completeness"][name]["mc_maxc_plus02"]
        ax.axvline(mc, color=COLORS[name], ls=(0, (1, 2)), lw=1.25)
        ax.text(mc, 0.97, f"Mc {mc:.2f}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.6, color=COLORS[name],
                bbox={"fc": "white", "ec": "none", "alpha": 0.92, "pad": 1.8})
    ax.set(xlabel="Magnitude", ylabel="Cumulative events, N(≥ M)", ylim=(0.75, max_count * 2))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, borderaxespad=0,
              markerscale=2.5)
    style_axes(ax)
    save_figure(fig, target / "fig_fmd.png")


def _rate(cfg, target, catalogs, results):
    fig, ax = plt.subplots(figsize=(7.6, 5.05))
    fig.subplots_adjust(left=0.105, right=0.97, bottom=0.155, top=0.735)
    figure_heading(
        fig, "Daily activity and Omori decay",
        "Q1 events · Fits begin after each catalog's largest selected event")
    for name in NAMES:
        data = catalogs[name]
        counts = data.groupby("day").size()
        ax.plot(counts.index, counts.values, drawstyle="steps-mid",
                color=COLORS[name], lw=1.5, label=f"{LABELS[name]} · daily count")
        omori = results["omori"].get(name)
        if omori and omori["delta_aic_const_minus_omori"] is not None:
            t0 = pd.Timestamp(omori["t0"])
            elapsed = np.linspace(
                0.05, (data["t"].max() - t0).total_seconds() / 86400, 200)
            rate = omori["K"] / (elapsed + omori["c_days"]) ** omori["p"]
            ax.plot(
                t0 + pd.to_timedelta(elapsed, "D"), rate,
                ls=(0, (5, 2.5)), lw=1.65, color=COLORS[name],
                label=f"{LABELS[name]} · Omori p = {omori['p']:.2f}")
    start, end = map(pd.Timestamp, cfg["window"])
    ax.set_xlim(start.normalize() - pd.Timedelta(days=0.8), end + pd.Timedelta(days=0.8))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    year_label = f"{start.year}, UTC" if start.year == end.year else "UTC"
    ax.set(yscale="log", xlabel=f"Date ({year_label})", ylabel="Events per day")
    ax.margins(y=0.10)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.055), ncol=2,
              borderaxespad=0, fontsize=8.8, columnspacing=2.4, labelspacing=0.7)
    style_axes(ax, grid="y")
    save_figure(fig, target / "fig_rate_omori.png")


def make_figures(cfg, out_dir, A, N, pzA, pzN, res):
    """Draw four figures using selected catalogs and supplied numerical results."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    catalogs = {"SantoriniE": A, "NLL": N}
    pca = {"SantoriniE": pzA, "NLL": pzN}
    with plt.rc_context(STYLE):
        _overview(cfg, target, catalogs, pca)
        _orientation(cfg, target, res)
        _magnitude(target, catalogs, res)
        _rate(cfg, target, catalogs, res)
