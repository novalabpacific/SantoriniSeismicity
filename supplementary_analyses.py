"""Supplementary analyses of the seismicity catalogs.

Usage: python3 supplementary_analyses.py config.json

Run from the package directory after analysis_pipeline.py. The configured
random seed controls resampling. Results, manuscript values, an analysis
summary, and residual figures are written to the configured output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_pipeline as P
import analysis_stats as S
from render_residuals import write_residual_figures


NAMES = ("SantoriniE", "NLL")
PAIRED_DAY_RESAMPLES = 5000
GEOMETRY_RESAMPLES = 300
EVENT_RESAMPLES = 2000
DOWNSAMPLE_RESAMPLES = 200
MAGNITUDE_BOOTSTRAPS = 1000
MAGNITUDE_PERMUTATIONS = 5000
MAGNITUDE_CUTOFFS = (1.0, 1.3, 1.5, 2.0)


def _load_catalogs(config):
    """Read the common time window and Q1 selection for both catalogs."""
    start, stop = map(pd.Timestamp, config["window"])
    catalogs = {}
    for name in NAMES:
        raw = P.load_catalog(config["catalogs"][name], config)
        window = raw[(raw["t"] >= start) & (raw["t"] < stop)].copy()
        selected = P.apply_filter(window, config["quality"][name]["Q1"])
        catalogs[name] = P.add_xy(
            selected, config["origin"]["lat0"], config["origin"]["lon0"]
        )
    return catalogs


def _azimuth(points):
    """Apply the shared PCA estimator to an East/North coordinate array."""
    return S.pca_azimuth(points[:, 0], points[:, 1])["az"]


def _signed_axial_interval(draws, center, alpha=0.05):
    """Express an axial interval as signed differences around its estimate."""
    lower, upper, _ = S.axial_percentile_ci(draws, center, alpha)
    return [float(center + S.axial_diff(bound, center)) for bound in (lower, upper)]


def _paired_day_difference(catalogs, rng):
    shared_days = sorted(
        set(catalogs["SantoriniE"]["day"]) & set(catalogs["NLL"]["day"])
    )
    groups = {
        name: {
            day: group[["x", "y"]].to_numpy()
            for day, group in catalogs[name].groupby("day")
        }
        for name in NAMES
    }
    observed = float(
        S.axial_diff(
            _azimuth(catalogs["SantoriniE"][["x", "y"]].to_numpy()),
            _azimuth(catalogs["NLL"][["x", "y"]].to_numpy()),
        )
    )
    differences = []
    for _ in range(PAIRED_DAY_RESAMPLES):
        selected_days = rng.choice(len(shared_days), len(shared_days), replace=True)
        samples = {
            name: np.vstack([groups[name][shared_days[i]] for i in selected_days])
            for name in NAMES
        }
        differences.append(
            S.axial_diff(_azimuth(samples["SantoriniE"]), _azimuth(samples["NLL"]))
        )
    ci95 = _signed_axial_interval(differences, observed)
    ci90 = _signed_axial_interval(differences, observed, alpha=0.10)
    return {
        "diff": observed,
        "ci95": ci95,
        "ci90": ci90,
        "equivalent_15": bool(ci90[0] > -15 and ci90[1] < 15),
        "n_days": len(shared_days),
    }


def _bootstrap_geometry(catalogs, rng):
    results = {}
    for name in NAMES:
        catalog = catalogs[name]
        point = S.pca3d_geometry(
            catalog["x"].values, catalog["y"].values, catalog["dep"].values
        )
        days = sorted(set(catalog["day"]))
        groups = {
            day: group[["x", "y", "dep"]].to_numpy()
            for day, group in catalog.groupby("day")
        }
        draws = {key: [] for key in ("trend", "plunge", "strike", "dip")}
        for _ in range(GEOMETRY_RESAMPLES):
            selected_days = rng.choice(len(days), len(days), replace=True)
            sample = np.vstack([groups[days[i]] for i in selected_days])
            geometry = S.pca3d_geometry(sample[:, 0], sample[:, 1], sample[:, 2])
            for key in ("trend", "plunge"):
                draws[key].append(geometry["line"][key])
            for key in ("strike", "dip"):
                draws[key].append(geometry["plane"][key])
        line, plane = point["line"], point["plane"]
        trend_ci = S.axial_percentile_ci(draws["trend"], line["trend"])[:2]
        strike_ci = S.axial_percentile_ci(draws["strike"], plane["strike"])[:2]
        eigenvalues = point["eigenvalues"]
        results[name] = {
            "trend": round(line["trend"], 1),
            "trend_ci": [round(float(value), 1) for value in trend_ci],
            "plunge": round(line["plunge"], 1),
            "plunge_ci": [
                round(float(value), 1) for value in np.percentile(draws["plunge"], [2.5, 97.5])
            ],
            "strike": round(plane["strike"], 1),
            "strike_ci": [round(float(value), 1) for value in strike_ci],
            "dip": round(plane["dip"], 1),
            "dip_ci": [
                round(float(value), 1) for value in np.percentile(draws["dip"], [2.5, 97.5])
            ],
            "lam12": round(float(eigenvalues[0] / eigenvalues[1]), 1),
            "lam23": round(float(eigenvalues[1] / eigenvalues[2]), 1),
        }
    return results


def _above_completeness_daily_bins(catalogs, results):
    counts = {}
    for name in NAMES:
        cutoff = results["magnitude_completeness"][name]["mc_maxc_plus02"]
        selected = catalogs[name][catalogs[name]["mag"] >= cutoff]
        counts[name] = int((selected.groupby("day").size() >= 10).sum())
    return counts


def _event_level_difference(catalogs, rng):
    points_a = catalogs["SantoriniE"][["x", "y"]].to_numpy()
    points_b = catalogs["NLL"][["x", "y"]].to_numpy()
    differences = []
    for _ in range(EVENT_RESAMPLES):
        indices_a = rng.integers(0, len(points_a), len(points_a))
        indices_b = rng.integers(0, len(points_b), len(points_b))
        differences.append(
            abs(
                S.axial_diff(
                    _azimuth(points_a[indices_a]), _azimuth(points_b[indices_b])
                )
            )
        )
    return {
        "median_absdiff": float(np.median(differences)),
        "ci95": [float(value) for value in np.percentile(differences, [2.5, 97.5])],
    }


def _early_axis_results(catalogs, config, results, rng):
    """Share the primary SantoriniE estimate and apply its method to NLL."""
    cutoff = S.early_axis_cutoff(config)
    primary = results["early_axis"]
    primary_catalog = catalogs["SantoriniE"]
    early = primary_catalog[primary_catalog["t"] < cutoff]
    late = primary_catalog[primary_catalog["t"] >= cutoff]
    if (
        primary["axis_days"] != config.get("early_axis_days", 14)
        or primary["n_early"] != len(early)
        or primary["n_late"] != len(late)
    ):
        raise ValueError("Primary early-axis results do not match the configured selection")
    primary["cutoff_utc"] = cutoff.isoformat()
    primary["n_late_days"] = len(S.daily_series(late, "dep")[0])
    by_catalog = {}
    for name in NAMES:
        estimate = (
            primary
            if name == "SantoriniE"
            else S.early_axis_analysis(catalogs[name], config, rng)
        )
        if estimate is None:
            raise ValueError(f"{name}: early-axis analysis requires more than 200 events per period")
        pp = estimate["late_block_perm_p"]
        by_catalog[name] = {
            "axis_days": estimate["axis_days"],
            "cutoff_utc": estimate["cutoff_utc"],
            "n_early": estimate["n_early"],
            "n_late": estimate["n_late"],
            "early_az": round(estimate["early_az"], 1),
            "late_theil": round(estimate["late_migration_theil"], 3),
            "late_block_ci": [round(value, 3) for value in estimate["late_migration_ci"]],
            "late_block_perm_p": round(float(pp["p"] if isinstance(pp, dict) else pp), 4),
            "n_late_days": estimate["n_late_days"],
        }
    return by_catalog


def _downsample_azimuths(catalogs, rng):
    points = catalogs["SantoriniE"][["x", "y"]].to_numpy()
    target_size = len(catalogs["NLL"])
    azimuths = []
    for _ in range(DOWNSAMPLE_RESAMPLES):
        indices = rng.choice(len(points), target_size, replace=False)
        azimuths.append(_azimuth(points[indices]))
    interval = S.axial_percentile_ci(azimuths, _azimuth(points))[:2]
    return {
        "median": float(np.median(azimuths)),
        "ci95": [float(value) for value in interval],
    }


def _magnitude_sensitivity(catalogs, rng):
    results = {}
    for name in NAMES:
        results[name] = {}
        for magnitude in MAGNITUDE_CUTOFFS:
            selected = catalogs[name][catalogs[name]["mag"] >= magnitude]
            time, depth = S.daily_series(selected, "dep")
            azimuth = _azimuth(selected[["x", "y"]].to_numpy())
            if len(time) >= 8:
                trend = S.block_boot_slope(
                    time, depth, block_len=5, n_boot=MAGNITUDE_BOOTSTRAPS, rng=rng
                )
                slope, interval = trend["slope"], trend["ci"]
                pp = S.block_perm_p(
                    time, depth, block_len=5, n_perm=MAGNITUDE_PERMUTATIONS, rng=rng
                )
                p_value = float(pp["p"] if isinstance(pp, dict) else pp)
            else:
                slope = p_value = float("nan")
                interval = [float("nan"), float("nan")]
            results[name][f"M{magnitude}"] = {
                "n_events": int(len(selected)),
                "n_days": int(len(time)),
                "az": round(azimuth, 1),
                "depth_theil": round(slope, 4),
                "depth_block_ci": [round(value, 4) for value in interval],
                "depth_block_perm_p": round(p_value, 4),
            }
            print(
                f"M>={magnitude} {name}: n={len(selected)} bins={len(time)} "
                f"az={azimuth:.1f} slope={slope:+.4f} "
                f"CI=[{interval[0]:+.4f},{interval[1]:+.4f}] p={p_value:.4f}"
            )
    return results


def _manuscript_values(results):
    """Build the manuscript table from numerical analysis results."""
    values = {
        "pca": {
            name: {
                f"Q{index}": [
                    results["pca"][name][f"Q{index}"]["n"],
                    round(results["pca"][name][f"Q{index}"]["az"], 1),
                ]
                for index in range(4)
            }
            for name in NAMES
        },
        "Q1_intervals": {
            name: {
                "event": [round(value, 1) for value in results["pca"][name]["Q1"]["ci_simple"]],
                "daycluster": [
                    round(value, 1) for value in results["pca"][name]["Q1"]["ci_day_cluster"]
                ],
                "errMC": [round(value, 2) for value in results["pca"][name]["Q1"]["ci_error_mc"]],
                "dailyCentered": [round(results["geometry"][name]["daily_centered"]["az"], 1)]
                + [round(value, 1) for value in results["geometry"][name]["daily_centered"]["ci"]],
            }
            for name in NAMES
        },
        "trends_blockCI": {
            key: {
                "theil": round(trend["theil_slope"], 3),
                "blockCI": [round(value, 3) for value in trend["ci"]],
                "blockPermP": round(trend["block_perm_p"], 4),
                "nDays": trend["n_days"],
            }
            for key, trend in results["trends"].items()
            if "dropfirst" not in key
        },
        "multiplicity": {
            key: [round(value["p_bonferroni"], 3), round(value["p_bh_fdr"], 3)]
            for key, value in results["multiplicity_primary_family"].items()
            if "dropfirst" not in key
        },
        "harmonized_NLL": {
            key: [round(value, 3) for value in comparison["NLL"]["depth_slope_ci"]]
            + [round(comparison["NLL"]["depth_block_perm_p"], 4)]
            for key, comparison in results["harmonized_errZ"].items()
        },
        "Mc": {
            name: [
                results["magnitude_completeness"][name]["mc_maxc_plus02"],
                results["magnitude_completeness"][name]["mc_gft90"],
                results["magnitude_completeness"][name]["n_ge_mc"],
                round(results["magnitude_completeness"][name]["migration_ge_mc"]["block_perm_p"], 3),
                round(results["magnitude_completeness"][name]["depth_ge_mc"]["block_perm_p"], 3),
                results["aboveMc_daily_bins"][name],
            ]
            for name in NAMES
        },
        "boot3d": results["boot3d"],
        "pairedDayDiff": results["paired_day_azdiff"],
        "azdiffDayClustered": [round(results["azimuth_difference"]["diff"], 2)]
        + [round(value, 2) for value in results["azimuth_difference"]["ci"]],
        "azdiffEventLevel": results["azdiff_event_level"],
        "earlyAxis": results["early_axis_by_catalog"],
        "subsampleAzRange": results["subsample_SantoriniE_azimuths"],
        "fixedMagnitude": results["fixed_magnitude"],
    }
    matching = results["matching"]
    values["matching"] = {
        "n": matching["n_matched"],
        "dhMed": round(matching["dh_km"]["median"], 2),
        "dzMed": round(matching["dz_km"]["median"], 2),
        "residTrend": [round(matching["dh_trend_km_per_day"]["theil"], 3)]
        + [round(value, 3) for value in matching["dh_trend_km_per_day"]["ci"]],
        "demingAlong": [
            round(matching["deming_alongaxis"]["intercept_km"], 2),
            round(matching["deming_alongaxis"]["slope"], 2),
        ],
        "demingDepth": [
            round(matching["deming_depth"]["intercept_km"], 2),
            round(matching["deming_depth"]["slope"], 2),
        ],
    }
    values["dbscan"] = {}
    for name in NAMES:
        grid = results["dbscan"][name]["grid"]
        largest = [row["largest_cluster_frac"] for row in grid]
        noise = [row["noise_frac"] for row in grid]
        values["dbscan"][name] = {
            "largestPct": [round(min(largest) * 100, 1), round(max(largest) * 100, 1)],
            "noisePct": [round(min(noise) * 100, 1), round(max(noise) * 100, 1)],
        }
    values["omori"] = {
        name: {
            key: round(value, 2) if isinstance(value, (int, float)) else str(value)
            for key, value in results["omori"][name].items()
            if key in ("p", "c_days", "delta_aic_const_minus_omori", "n")
        }
        for name in NAMES
    }
    return pd.DataFrame(
        [(key, json.dumps(value, default=str)) for key, value in values.items()],
        columns=["key", "value"],
    )


def _write_summary(results, out_dir, seed):
    P.write_summary(results, str(out_dir))
    paired = results["paired_day_azdiff"]
    event = results["azdiff_event_level"]
    lines = [
        f"\n## Supplementary analyses (seed {seed})\n",
        f"- Paired-day azimuth difference: {paired['diff']:.2f} deg, 95% CI {paired['ci95']}",
        f"- Event-level |difference|: median {event['median_absdiff']:.2f} deg, 95% {event['ci95']}",
        f"- 3-D day-cluster bootstrap: {json.dumps(results['boot3d'])}",
        f"- Above-Mc daily bins: {results['aboveMc_daily_bins']}",
        f"- Early-axis late slopes: {json.dumps(results['early_axis_by_catalog'])}",
        f"- SantoriniE downsampling azimuth: {json.dumps(results['subsample_SantoriniE_azimuths'])}",
    ]
    with (out_dir / "summary.md").open("a", encoding="utf-8") as summary:
        summary.write("\n".join(lines) + "\n")


def run(cfg_path):
    """Run the supplementary procedures against the configured primary results."""
    config = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    if config.get("primary_filter", "Q1") != "Q1":
        raise ValueError("Supplementary analyses require the Q1 primary selection")
    out_dir = Path(config["out_dir"])
    seed = config.get("seed", 20260822)
    rng = np.random.default_rng(seed)
    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    catalogs = _load_catalogs(config)

    results["paired_day_azdiff"] = _paired_day_difference(catalogs, rng)
    print("paired-day:", results["paired_day_azdiff"])
    results["boot3d"] = _bootstrap_geometry(catalogs, rng)
    print("3-D geometry complete")
    results["aboveMc_daily_bins"] = _above_completeness_daily_bins(catalogs, results)
    print("above-Mc bins:", results["aboveMc_daily_bins"])
    results["azdiff_event_level"] = _event_level_difference(catalogs, rng)
    print("event-level |difference|:", results["azdiff_event_level"])
    results["early_axis_by_catalog"] = _early_axis_results(catalogs, config, results, rng)
    print("early-axis:", results["early_axis_by_catalog"])
    results["subsample_SantoriniE_azimuths"] = _downsample_azimuths(catalogs, rng)
    print("downsample azimuth:", results["subsample_SantoriniE_azimuths"])
    write_residual_figures(catalogs, out_dir, reference=results)
    print("Residual maps, model profiles, and Supplementary Figure S1 written")
    results["fixed_magnitude"] = _magnitude_sensitivity(catalogs, rng)

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8"
    )
    values = _manuscript_values(results)
    values.to_csv(out_dir / "manuscript_values.csv", index=False)
    _write_summary(results, out_dir, seed)
    print("Supplementary analyses complete; value rows:", len(values))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="config.json")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
