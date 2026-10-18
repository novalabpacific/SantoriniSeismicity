"""
analysis_pipeline.py — analysis of the Santorini-Amorgos seismicity catalogs.

Usage:  python analysis_pipeline.py config.json

Reads SantoriniE and NLL-SSST catalogs, applies the common window and quality
filters, and computes catalog agreement, geometry, temporal trends, magnitude
completeness, and clustering diagnostics. Results are written to
<out_dir>/results.json, figures to <out_dir>/fig_*.png, and an analysis digest
to <out_dir>/summary.md.
"""
from __future__ import annotations
import json, sys, os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy import stats

import analysis_stats as R

# ----------------------------------------------------------------- loading
def load_catalog(spec, cfg):
    cols = spec["cols"]
    sep = spec.get("sep", ",")
    df = (pd.read_csv(spec["path"], sep=r"\s+", comment="#")
          if sep == "whitespace" else
          pd.read_csv(spec["path"], sep=sep, skipinitialspace=True, comment="#"))
    df.columns = [str(c).strip() for c in df.columns]
    if "time_parts" in cols:  # e.g. Zenodo v1.0: year/month/day/hour/minute/seconds
        tp = cols["time_parts"]
        parts = pd.DataFrame({
            "year": df[tp["year"]].astype(int), "month": df[tp["month"]].astype(int),
            "day": df[tp["day"]].astype(int), "hour": df[tp["hour"]].astype(int),
            "minute": df[tp["minute"]].astype(int),
        })
        secs = df[tp["second"]].astype(float)
        t = pd.to_datetime(parts) + pd.to_timedelta(secs, unit="s")
    else:
        t = pd.to_datetime(df[cols["time"]])
    out = pd.DataFrame({
        "t": t,
        "lat": df[cols["lat"]].astype(float),
        "lon": df[cols["lon"]].astype(float),
        "dep": df[cols["dep"]].astype(float),
    })
    for k in ("mag", "errH", "errZ", "rms", "gap", "nphs"):
        out[k] = df[cols[k]].astype(float) if cols.get(k) in df.columns else np.nan
    return out


def add_xy(df, lat0, lon0, proj="equirect"):
    f = R.project_equirect if proj == "equirect" else R.project_azeq
    x, y = f(df["lat"].values, df["lon"].values, lat0, lon0)
    df = df.copy(); df["x"], df["y"] = x, y
    df["day"] = df["t"].dt.floor("D")
    return df


def apply_filter(df, rule):
    m = np.ones(len(df), bool)
    for key, col, op in (("errH_max", "errH", "le"), ("errZ_max", "errZ", "le"),
                         ("rms_max", "rms", "le"), ("gap_max", "gap", "le"),
                         ("nphs_min", "nphs", "ge"), ("mag_min", "mag", "ge")):
        if key in rule and df[col].notna().any():
            m &= (df[col] <= rule[key]) if op == "le" else (df[col] >= rule[key])
    return df[m].reset_index(drop=True)


def read_gmt_xy(path, bbox=None, latlon=False):
    segs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                if cur: segs.append(np.array(cur)); cur = []
                continue
            p = line.split()
            try:
                a, b = float(p[0]), float(p[1])
            except ValueError:
                continue
            lon, lat = (b, a) if latlon else (a, b)
            cur.append((lon, lat))
    if cur: segs.append(np.array(cur))
    if bbox:
        lo1, lo2, la1, la2 = bbox
        segs = [s for s in segs
                if ((s[:, 0] >= lo1) & (s[:, 0] <= lo2) &
                    (s[:, 1] >= la1) & (s[:, 1] <= la2)).any()]
    return segs


# ----------------------------------------------------------------- pipeline
def run(cfg_path):
    cfg = json.load(open(cfg_path))
    rng = np.random.default_rng(cfg.get("seed", 20260822))
    out_dir = cfg["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    lat0, lon0 = cfg["origin"]["lat0"], cfg["origin"]["lon0"]
    t1, t2 = [pd.Timestamp(s) for s in cfg["window"]]
    nb = cfg.get("n_boot", {})
    NB_S, NB_C, NB_P, NB_MC = (nb.get("simple", 2000), nb.get("cluster", 1000),
                               nb.get("perm", 1000), nb.get("error_mc", 200))
    L = cfg.get("block_len", 5)
    res = {"config": cfg}

    # ---- load
    raw = {}
    nll_spec = cfg["catalogs"]["NLL"]
    raw["NLL"] = load_catalog(nll_spec, cfg)
    santorini_e_spec = cfg["catalogs"]["SantoriniE"]
    raw["SantoriniE"] = load_catalog(santorini_e_spec, cfg)

    res["full_span"] = {k: [str(v["t"].min()), str(v["t"].max()), len(v)]
                        for k, v in raw.items()}

    # ---- common window + projection
    cat = {}
    for k, v in raw.items():
        v = v[(v["t"] >= t1) & (v["t"] < t2)].reset_index(drop=True)
        cat[k] = add_xy(v, lat0, lon0)
    res["window_counts"] = {k: len(v) for k, v in cat.items()}
    res["pre_window_2024_counts"] = {
        k: int((raw[k]["t"] < pd.Timestamp("2025-01-01")).sum()) for k in raw}

    # ---- quality filters (incl. Q-table)
    Q = {}
    qtab = {}
    for k in cat:
        Q[k] = {}
        for qname, rule in cfg["quality"][k].items():
            Q[k][qname] = apply_filter(cat[k], rule)
            qtab[f"{k}.{qname}"] = len(Q[k][qname])
    res["quality_counts"] = qtab
    P = cfg.get("primary_filter", "Q1")
    A, N = Q["SantoriniE"][P], Q["NLL"][P]

    # =================================================================
    # PCA azimuths per quality filter + clustered bootstrap CIs
    # =================================================================
    res["pca"] = {}
    for k in cat:
        res["pca"][k] = {}
        for qname, d in Q[k].items():
            pz = R.pca_azimuth(d["x"].values, d["y"].values)
            entry = {"n": len(d), "az": pz["az"], "ev1_frac": pz["ev1_frac"]}
            bs = R.boot_simple(d, NB_S, rng)
            entry["ci_simple"] = list(R.axial_percentile_ci(bs, pz["az"])[:2])
            bc = R.boot_cluster(d, "day", NB_C, rng=rng)
            lo, hi, sd = R.axial_percentile_ci(bc, pz["az"])
            entry["ci_day_cluster"] = [lo, hi]; entry["sd_day_cluster"] = sd
            be = R.boot_cluster(d, "day", NB_C, equal_weight=True, rng=rng)
            entry["ci_day_cluster_equalweight"] = list(
                R.axial_percentile_ci(be, pz["az"])[:2])
            res["pca"][k][qname] = entry

    # error Monte-Carlo + combined
    for k, d in (("SantoriniE", A), ("NLL", N)):
        if d["errH"].notna().all():
            mc = R.boot_error_mc(d, NB_MC, None, rng)
            mcc = R.boot_error_mc(d, NB_MC, "day", rng)
            az0 = res["pca"][k][P]["az"]
            res["pca"][k][P]["ci_error_mc"] = list(R.axial_percentile_ci(mc, az0)[:2])
            res["pca"][k][P]["ci_error_mc_plus_cluster"] = list(
                R.axial_percentile_ci(mcc, az0)[:2])

    # =================================================================
    # formal azimuth-difference + equivalence + permutation
    # =================================================================
    ad = R.azdiff_bootstrap(A, N, "day", NB_C, rng)
    eq = R.azdiff_equivalence(ad.pop("boot_diffs"), cfg.get("equivalence_bound_deg", 15))
    pm = R.azdiff_permutation(A, N, NB_P, rng)
    res["azimuth_difference"] = {**ad, "equivalence": eq, "permutation": pm}

    # pre-specified concordance criterion for each filter vs the primary filter
    crit = {}
    bound = cfg.get("equivalence_bound_deg", 15)
    for k in cat:
        azs = {q: res["pca"][k][q]["az"] for q in res["pca"][k]}
        ref = azs.get(P)
        crit[k] = {q: {"delta_vs_primary": float(R.axial_diff(a, ref)),
                       "within_bound": bool(abs(R.axial_diff(a, ref)) <= bound)}
                   for q, a in azs.items()}
    res["filter_concordance_criterion"] = {"bound_deg": bound, "table": crit}

    # =================================================================
    # event matching, residuals, Deming
    # =================================================================
    ia, ib, dts = R.match_catalogs(A, N, cfg.get("match_tol_s", 2.0))
    MA, MB = A.iloc[ia].reset_index(drop=True), N.iloc[ib].reset_index(drop=True)
    dx, dy = MA["x"] - MB["x"], MA["y"] - MB["y"]
    dh = np.hypot(dx, dy); dz = MA["dep"] - MB["dep"]
    pzA = R.pca_azimuth(A["x"].values, A["y"].values)
    pzN = R.pca_azimuth(N["x"].values, N["y"].values)
    sA, tA_ = R.along_perp(MA["x"].values, MA["y"].values, pzA["mu"], pzA["v"])
    sN, tN_ = R.along_perp(MB["x"].values, MB["y"].values, pzN["mu"], pzN["v"])
    daym = (MA["t"] - MA["t"].min()).dt.total_seconds().values / 86400.0
    ts_dh = stats.theilslopes(dh, daym)
    var_ratio = float(np.nanmean(MB["errH"] ** 2) / max(np.nanmean(MA["errH"] ** 2), 1e-9))
    a_dem_s, b_dem_s = R.deming(sA, sN, var_ratio)
    a_dem_z, b_dem_z = R.deming(MA["dep"].values, MB["dep"].values,
                                float(np.nanmean(MB["errZ"] ** 2) /
                                      max(np.nanmean(MA["errZ"] ** 2), 1e-9)))
    res["matching"] = {
        "tol_s": cfg.get("match_tol_s", 2.0),
        "n_matched": int(len(ia)),
        "frac_SantoriniE_matched": float(len(ia) / len(A)),
        "frac_NLL_matched": float(len(ib) / len(N)),
        "frac_SantoriniE_only": float(1 - len(ia) / len(A)),
        "frac_NLL_only": float(1 - len(ib) / len(N)),
        "dh_km": {"median": float(np.median(dh)), "p90": float(np.percentile(dh, 90))},
        "dz_km": {"median": float(np.median(dz)),
                  "median_abs": float(np.median(np.abs(dz))),
                  "p90_abs": float(np.percentile(np.abs(dz), 90))},
        "dh_trend_km_per_day": {"theil": float(ts_dh.slope),
                                "ci": [float(ts_dh.low_slope), float(ts_dh.high_slope)]},
        "orth_dist_km": {"SantoriniE_median_abs": float(np.median(np.abs(tA_))),
                         "NLL_median_abs": float(np.median(np.abs(tN_)))},
        "deming_alongaxis": {"intercept_km": a_dem_s, "slope": b_dem_s},
        "deming_depth": {"intercept_km": a_dem_z, "slope": b_dem_z},
    }

    # =================================================================
    # temporal trends + block resampling + multiplicity
    # =================================================================
    trends = {}
    for k, d, pz in (("SantoriniE", A, pzA), ("NLL", N, pzN)):
        s, _ = R.along_perp(d["x"].values, d["y"].values, pz["mu"], pz["v"])
        d = d.assign(s_along=s)
        trends[f"{k}_migration"] = R.analyze_trend(d, "s_along", L, (1, 10), NB_P, rng)
        trends[f"{k}_depth"] = R.analyze_trend(d, "dep", L, (1, 10), NB_P, rng)
        trends[f"{k}_depth_dropfirst"] = R.analyze_trend(
            d, "dep", L, (L, L), max(500, NB_P // 2), rng, drop_first_day=True)
    res["trends"] = trends
    fam = {k: trends[k]["block_perm_p"]
           for k in ("SantoriniE_migration", "SantoriniE_depth", "NLL_migration", "NLL_depth")}
    res["multiplicity_primary_family"] = R.multiplicity(fam)

    # harmonized vertical-error thresholds
    harm = {}
    for zmax in cfg.get("harmonized_errZ", [3.0, 5.0]):
        hh = {}
        for k, base in (("SantoriniE", cat["SantoriniE"]), ("NLL", cat["NLL"])):
            d = apply_filter(base, {**cfg["quality"][k][P], "errZ_max": zmax})
            pz = R.pca_azimuth(d["x"].values, d["y"].values)
            s, _ = R.along_perp(d["x"].values, d["y"].values, pz["mu"], pz["v"])
            depth_trend = R.analyze_trend(d.assign(s_along=s), "dep", L, (L, L),
                                         max(500, NB_P // 2), rng)
            hh[k] = {"n": len(d), "az": pz["az"],
                     "depth_theil": depth_trend["theil_slope"],
                     "depth_block_perm_p": depth_trend["block_perm_p"],
                     "depth_slope_ci": depth_trend["ci"]}
        harm[f"errZ<= {zmax}"] = hh
    res["harmonized_errZ"] = harm

    # =================================================================
    # geometry — stratified/weighted PCA, curvature, 3-D, residual map
    # =================================================================
    geo = {}
    for k, d, pz in (("SantoriniE", A, pzA), ("NLL", N, pzN)):
        s, tp = R.along_perp(d["x"].values, d["y"].values, pz["mu"], pz["v"])
        geo[k] = {"models": R.geometry_models(s, tp),
                  "pca3d": R.pca3d_line(d["x"].values, d["y"].values, d["dep"].values),
                  "plane": R.plane_fit(d["x"].values, d["y"].values, d["dep"].values)}
        # 3-D trend day-clustered bootstrap CI
        codes = pd.Categorical(d["day"]).codes
        tr = []
        for _ in range(max(300, NB_C // 3)):
            pick = rng.choice(codes.max() + 1, codes.max() + 1, replace=True)
            sel = np.concatenate([np.flatnonzero(codes == c) for c in pick])
            tr.append(R.pca3d_line(d["x"].values[sel], d["y"].values[sel],
                                   d["dep"].values[sel])["trend"])
        geo[k]["pca3d"]["trend_ci"] = list(
            R.axial_percentile_ci(np.array(tr), geo[k]["pca3d"]["trend"])[:2])
        # time-stratified PCA with day-clustered CI
        weekly = []
        for wk, g in d.groupby(pd.Grouper(key="t", freq=f"{cfg.get('strat_days',7)}D")):
            if len(g) < 100:
                continue
            pw = R.pca_azimuth(g["x"].values, g["y"].values)
            bw = R.boot_cluster(g, "day", max(300, NB_C // 3), rng=rng)
            lo, hi, _ = R.axial_percentile_ci(bw, pw["az"])
            weekly.append({"start": str(wk.date()), "n": len(g),
                           "az": pw["az"], "ci": [lo, hi]})
        geo[k]["time_stratified"] = weekly
        wq = 1.0 / d.groupby("day")["x"].transform("size").values
        geo[k]["equal_day_weight_az"] = R.pca_azimuth(d["x"].values,
                                                      d["y"].values, wq)["az"]
        # daily-centered PCA + CI
        dc = d.copy()
        dc["x"] -= dc.groupby("day")["x"].transform("median")
        dc["y"] -= dc.groupby("day")["y"].transform("median")
        pdc = R.pca_azimuth(dc["x"].values, dc["y"].values)
        bdc = R.boot_cluster(dc, "day", max(300, NB_C // 2), rng=rng)
        lo, hi, _ = R.axial_percentile_ci(bdc, pdc["az"])
        geo[k]["daily_centered"] = {"az": pdc["az"], "ci": [lo, hi]}
    res["geometry"] = geo

    # origin & projection invariance
    proj_tab = {}
    for k, d0 in (("SantoriniE", raw["SantoriniE"]), ("NLL", raw["NLL"])):
        d0 = d0[(d0["t"] >= t1) & (d0["t"] < t2)]
        d0 = apply_filter(add_xy(d0, lat0, lon0), cfg["quality"][k][P])
        base = R.pca_azimuth(d0["x"].values, d0["y"].values)["az"]
        row = {"base_equirect": base}
        for tag, (la, lo, pr) in {
            "meanlat_equirect": (float(d0["lat"].mean()), float(d0["lon"].mean()), "equirect"),
            "shift+0.2deg": (lat0 + 0.2, lon0, "equirect"),
            "shift-0.2deg": (lat0 - 0.2, lon0, "equirect"),
            "azimuthal_equidistant": (lat0, lon0, "azeq")}.items():
            dd = add_xy(d0.drop(columns=["x", "y"]), la, lo, pr)
            row[tag] = float(R.axial_diff(
                R.pca_azimuth(dd["x"].values, dd["y"].values)["az"], base))
        proj_tab[k] = row
    res["projection_sensitivity"] = proj_tab

    # =================================================================
    # magnitudes, Mc, b-value, M>=Mc sensitivity
    # =================================================================
    mag = {}
    for k, d in (("SantoriniE", A), ("NLL", N)):
        if not d["mag"].notna().any():
            mag[k] = {"available": False}; continue
        m = d["mag"].dropna().values
        mc_raw, mc_corr = R.mc_maxc(m)
        gft_mc, gft_R = R.mc_gft(m)
        bv = R.b_value(m, mc_corr)
        # completeness through time (weekly maxc)
        wk_mc = []
        for wk, g in d.dropna(subset=["mag"]).groupby(
                pd.Grouper(key="t", freq="7D")):
            if len(g) >= 200:
                wk_mc.append({"start": str(wk.date()),
                              "mc_maxc": R.mc_maxc(g["mag"].values)[1]})
        dmc = d[d["mag"] >= mc_corr].reset_index(drop=True)
        pzc = R.pca_azimuth(dmc["x"].values, dmc["y"].values)
        s, _ = R.along_perp(dmc["x"].values, dmc["y"].values, pzc["mu"], pzc["v"])
        migration_trend = R.analyze_trend(dmc.assign(s_along=s), "s_along", L, (L, L),
                                         max(500, NB_P // 2), rng)
        depth_trend = R.analyze_trend(dmc, "dep", L, (L, L), max(500, NB_P // 2), rng)
        mag[k] = {"available": True, "range": [float(m.min()), float(m.max())],
                  "mc_maxc_raw": mc_raw, "mc_maxc_plus02": mc_corr,
                  "mc_gft90": gft_mc, "gft_R": gft_R,
                  "b_value": bv, "weekly_mc": wk_mc,
                  "n_ge_mc": len(dmc), "az_ge_mc": pzc["az"],
                  "migration_ge_mc": {kk: migration_trend[kk] for kk in
                                      ("theil_slope", "block_perm_p")},
                  "depth_ge_mc": {kk: depth_trend[kk] for kk in
                                  ("theil_slope", "block_perm_p")}}
    res["magnitude_completeness"] = mag

    # post-largest-event rate + Omori vs constant
    rate = {}
    for k, d in (("SantoriniE", A), ("NLL", N)):
        if d["mag"].notna().any():
            t0 = d.loc[d["mag"].idxmax(), "t"]
        else:
            t0 = d.groupby("day").size().idxmax() + pd.Timedelta("1D")
        tt = (d["t"] - t0).dt.total_seconds().values / 86400.0
        T = float(tt.max())
        if T > 3:
            rate[k] = {"t0": str(t0), **R.omori_fit(tt, T)}
    res["omori"] = rate

    # =================================================================
    # DBSCAN grid + noise diagnostics (both z-scored per axis)
    # =================================================================
    db = {}
    for k, d in (("SantoriniE", A), ("NLL", N)):
        Xs = np.column_stack([stats.zscore(d["x"]), stats.zscore(d["y"]),
                              stats.zscore(d["dep"])])   # standardized per axis
        grid = []
        for eps in cfg.get("dbscan_eps", [0.25, 0.30, 0.35, 0.40, 0.50]):
            for ms in cfg.get("dbscan_min_samples", [15, 30, 50]):
                lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xs)
                ncl = int(len(set(lab)) - (1 if -1 in lab else 0))
                noise = float((lab == -1).mean())
                big = float((lab == pd.Series(lab[lab >= 0]).mode().iloc[0]).mean()) \
                    if ncl else 0.0
                grid.append({"eps": eps, "min_samples": ms, "n_clusters": ncl,
                             "noise_frac": noise, "largest_cluster_frac": big})
        db[k] = {"grid": grid}
        lab = DBSCAN(eps=0.35, min_samples=30).fit_predict(Xs)
        noise_mask = lab == -1
        diag = {}
        for colc in ("errZ", "errH", "gap", "nphs"):
            if d[colc].notna().any():
                u = stats.mannwhitneyu(d.loc[noise_mask, colc].dropna(),
                                       d.loc[~noise_mask, colc].dropna())
                diag[colc] = {"noise_median": float(d.loc[noise_mask, colc].median()),
                              "cluster_median": float(d.loc[~noise_mask, colc].median()),
                              "mwu_p": float(u.pvalue)}
        q75 = d["errZ"].quantile(0.75)
        diag["noise_frac_in_worst_errZ_quartile"] = float(
            (d.loc[noise_mask, "errZ"] > q75).mean()) if noise_mask.any() else np.nan
        db[k]["noise_frac_eps035_ms30"] = float(noise_mask.mean())
        db[k]["noise_diagnostics"] = diag
    res["dbscan"] = db

    # subsample the larger catalog to the smaller catalog's event count
    big_k, big, small = ("SantoriniE", A, N) if len(A) >= len(N) else ("NLL", N, A)
    widths = []
    for _ in range(200):
        sub = big.sample(len(small), random_state=int(rng.integers(1e9)))
        b = R.boot_cluster(sub, "day", 200, rng=rng)
        lo, hi, _ = R.axial_percentile_ci(b, R.pca_azimuth(sub["x"].values,
                                                           sub["y"].values)["az"])
        widths.append(abs(R.axial_diff(hi, lo)))
    full_b = R.boot_cluster(big, "day", 400, rng=rng)
    flo, fhi, _ = R.axial_percentile_ci(full_b, res["pca"][big_k][P]["az"])
    res["subsample_SantoriniE_to_NLL_n"] = {
        "downsampled_catalog": big_k, "target_n": int(len(small)),
        "full_ci_width_deg": float(abs(R.axial_diff(fhi, flo))),
        "subsampled_ci_width_median_deg": float(np.median(widths)),
        "subsampled_ci_width_iqr": [float(np.percentile(widths, 25)),
                                    float(np.percentile(widths, 75))]}

    # early-axis test
    early_result = R.early_axis_analysis(A, cfg, rng)
    if early_result is not None:
        res["early_axis"] = early_result

    # =================================================================
    # figures
    # =================================================================
    make_figures(cfg, out_dir, A, N, pzA, pzN, res)

    json.dump(res, open(f"{out_dir}/results.json", "w"), indent=1, default=str)
    write_summary(res, out_dir)
    print(f"[done] results -> {out_dir}/results.json ; digest -> {out_dir}/summary.md")
    return res


# ----------------------------------------------------------------- figures
def make_figures(cfg, out_dir, A, N, pzA, pzN, res):
    """Render the study overview, orientation, magnitude, and rate figures."""
    from study_figures import make_figures as draw_study_figures

    return draw_study_figures(cfg, out_dir, A, N, pzA, pzN, res)


# ----------------------------------------------------------------- summary
def write_summary(res, out_dir):
    P = res["config"].get("primary_filter", "Q1")
    L = res["config"].get("block_len", 5)
    fp = R.fmt_p
    g = res
    lines = []
    add = lines.append
    add(f"# Seismicity analysis digest — label: {g['config'].get('label','')}\n")
    add("## Data\n")
    for k, v in g["full_span"].items():
        add(f"* {k}: {v[2]} events, {v[0][:19]} -> {v[1][:19]}")
    add(f"* common-window counts: {g['window_counts']}; "
        f"pre-2025 (baseline) counts: {g['pre_window_2024_counts']}\n")
    add("## Quality-filter table\n")
    add("| filter | n | az (deg) | 95% CI day-clustered | EV1 frac |")
    add("|---|---|---|---|---|")
    for k in ("SantoriniE", "NLL"):
        for q, e in g["pca"][k].items():
            add(f"| {k}.{q} | {e['n']} | {e['az']:.1f} | "
                f"[{e['ci_day_cluster'][0]:.1f}, {e['ci_day_cluster'][1]:.1f}] | "
                f"{e['ev1_frac']:.3f} |")
    add("")
    ad = g["azimuth_difference"]
    add("## Formal azimuth comparison (primary filter "
        f"{P}, day-clustered)\n")
    add(f"* az(SantoriniE) = {ad['azA']:.2f}, az(NLL) = {ad['azB']:.2f}, "
        f"diff = {ad['diff']:.2f} deg, 95% CI [{ad['ci'][0]:.2f}, {ad['ci'][1]:.2f}], "
        f"bootstrap {fp(ad['p_boot'])}; day-stratified permutation "
        f"{fp(ad['permutation']['p_perm'])}")
    eq = ad["equivalence"]
    add(f"* pre-specified equivalence bound +/-{eq['bound']} deg: 90% CI "
        f"[{eq['ci90'][0]:.2f}, {eq['ci90'][1]:.2f}] -> equivalent(TOST) = "
        f"{eq['equivalent_90']}\n")
    add("## Matched-event comparison\n")
    m = g["matching"]
    add(f"* matched n = {m['n_matched']} (SantoriniE matched {m['frac_SantoriniE_matched']:.1%}, "
        f"NLL matched {m['frac_NLL_matched']:.1%}; SantoriniE-only {m['frac_SantoriniE_only']:.1%}, "
        f"NLL-only {m['frac_NLL_only']:.1%})")
    add(f"* horizontal residual median {m['dh_km']['median']:.2f} km "
        f"(p90 {m['dh_km']['p90']:.2f}); depth residual median "
        f"{m['dz_km']['median']:+.2f} km, |dz| median {m['dz_km']['median_abs']:.2f}")
    add(f"* residual-vs-time Theil slope {m['dh_trend_km_per_day']['theil']:.4f} km/day "
        f"CI [{m['dh_trend_km_per_day']['ci'][0]:.4f}, {m['dh_trend_km_per_day']['ci'][1]:.4f}]")
    add(f"* Deming along-axis slope {m['deming_alongaxis']['slope']:.3f}; "
        f"depth slope {m['deming_depth']['slope']:.3f}\n")
    add(f"## Primary temporal family (block-permutation, L={L} retained daily bins) "
        "with multiplicity control\n")
    add("| test | Theil slope | raw p | Bonferroni | BH-FDR | sig(BH) |")
    add("|---|---|---|---|---|---|")
    for k, v in g["multiplicity_primary_family"].items():
        t = g["trends"][k]
        add(f"| {k} | {t['theil_slope']:+.3f} | {fp(v['p_raw'])} | "
            f"{fp(v['p_bonferroni'])} | {fp(v['p_bh_fdr'])} | "
            f"{'YES' if v['significant_bh_05'] else 'no'} |")
    add("")
    add("## Harmonized vertical-error thresholds (depth trend)\n")
    for zc, hh in g["harmonized_errZ"].items():
        for k, v in hh.items():
            add(f"* {zc} | {k}: n={v['n']}, az={v['az']:.1f}, depth Theil "
                f"{v['depth_theil']:+.3f} km/d (CI [{v['depth_slope_ci'][0]:.3f}, {v['depth_slope_ci'][1]:.3f}]), "
                f"block {fp(v['depth_block_perm_p'])}")
    add("")
    add("## Geometry model comparison\n")
    for k in ("SantoriniE", "NLL"):
        ge = g["geometry"][k]
        d = ge["models"]["delta_aic_vs_line"]
        add(f"* {k}: dAIC vs straight line -> tilted {d['tilted_line']:+.0f}, "
            f"quadratic {d['quadratic']:+.0f}, piecewise {d['piecewise']:+.0f} "
            f"(break at s={ge['models']['piecewise_break_s_km']:.1f} km); "
            f"3-D trend {ge['pca3d']['trend']:.1f} deg "
            f"(CI [{ge['pca3d']['trend_ci'][0]:.1f}, {ge['pca3d']['trend_ci'][1]:.1f}]), "
            f"plunge {ge['pca3d']['plunge']:.1f} deg; plane strike "
            f"{ge['plane']['strike']:.1f}, dip {ge['plane']['dip']:.1f}; "
            f"equal-day-weight az {ge['equal_day_weight_az']:.1f}; "
            f"daily-centered az {ge['daily_centered']['az']:.1f} "
            f"CI [{ge['daily_centered']['ci'][0]:.1f}, {ge['daily_centered']['ci'][1]:.1f}]")
    add("")
    add("## Origin / projection sensitivity (deg change vs base)\n")
    for k, row in g["projection_sensitivity"].items():
        add(f"* {k}: " + ", ".join(f"{t}={v:+.2f}" if t != "base_equirect"
                                   else f"base={v:.2f}" for t, v in row.items()))
    add("")
    add("## Magnitudes & completeness\n")
    for k, mm in g["magnitude_completeness"].items():
        if not mm.get("available"):
            add(f"* {k}: magnitudes not available in file"); continue
        b = mm["b_value"]
        add(f"* {k}: M range [{mm['range'][0]:.2f}, {mm['range'][1]:.2f}], "
            f"Mc(maxc+0.2)={mm['mc_maxc_plus02']:.2f}, Mc(GFT90)={mm['mc_gft90']:.2f}, "
            f"b={b['b']:.2f}+/-{b['b_err_aki']:.2f} (n={b['n']}); "
            f"M>=Mc subset: n={mm['n_ge_mc']}, az={mm['az_ge_mc']:.1f}, "
            f"migration block {fp(mm['migration_ge_mc']['block_perm_p'])}, "
            f"depth block {fp(mm['depth_ge_mc']['block_perm_p'])}")
    add("")
    add("## Omori vs constant rate\n")
    for k, om in g["omori"].items():
        add(f"* {k}: t0={om['t0'][:16]}, p={om['p']:.2f}, c={om['c_days']:.2f} d, "
            f"dAIC(const-Omori)={om['delta_aic_const_minus_omori']:.0f} "
            f"({'Omori strongly favored' if om['delta_aic_const_minus_omori']>10 else 'weak'})")
    add("")
    ss = g["subsample_SantoriniE_to_NLL_n"]
    downsampled = ss["downsampled_catalog"]
    target = "NLL" if downsampled == "SantoriniE" else "SantoriniE"
    add(f"## {downsampled} subsampled to {target} n (CI width, deg)\n")
    add(f"* full-{downsampled} day-clustered width {ss['full_ci_width_deg']:.2f}; "
        f"subsampled median {ss['subsampled_ci_width_median_deg']:.2f} "
        f"(IQR [{ss['subsampled_ci_width_iqr'][0]:.2f}, {ss['subsampled_ci_width_iqr'][1]:.2f}])\n")
    add("## DBSCAN sensitivity & noise diagnostics\n")
    for k in ("SantoriniE", "NLL"):
        gr = g["dbscan"][k]["grid"]
        nf = [f"{r['eps']}/{r['min_samples']}:{r['noise_frac']:.2f}" for r in gr]
        add(f"* {k}: noise fraction over grid -> " + ", ".join(nf))
        dgn = g["dbscan"][k]["noise_diagnostics"]
        if "errZ" in dgn:
            add(f"  - noise vs cluster errZ medians {dgn['errZ']['noise_median']:.2f} "
                f"vs {dgn['errZ']['cluster_median']:.2f} ({fp(dgn['errZ']['mwu_p'])}); "
                f"noise in worst-errZ quartile: "
                f"{dgn['noise_frac_in_worst_errZ_quartile']:.1%}")
    add("")
    if "early_axis" in g:
        e = g["early_axis"]
        add("## Early-defined axis test (SantoriniE catalog)\n")
        add(f"* axis from first {e['axis_days']} d (az {e['early_az']:.1f}, "
            f"n={e['n_early']}); later events (n={e['n_late']}): migration Theil "
            f"{e['late_migration_theil']:+.3f} km/d CI [{e['late_migration_ci'][0]:.3f}, {e['late_migration_ci'][1]:.3f}], "
            f"block {fp(e['late_block_perm_p'])}\n")
    open(f"{out_dir}/summary.md", "w").write("\n".join(lines))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "config.json")
