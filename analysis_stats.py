"""
analysis_stats.py — statistical methods for Santorini-Amorgos seismicity analysis.

Functions cover catalog geometry, uncertainty, cross-catalog comparisons,
temporal trends, and magnitude completeness.

Conventions
-----------
* Azimuths are AXIAL quantities in degrees east of North, in [0, 180).
* x = local East (km), y = local North (km), z = depth (km, positive down).
* All bootstrap CIs handle the 180-degree wrap-around via rotation into a
  +/-90 degree window around the point estimate before taking percentiles.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats, optimize

RNG = np.random.default_rng(20260822)
R_EARTH = 6371.0

# ----------------------------------------------------------------------
# angles (axial data)
# ----------------------------------------------------------------------
def wrap_axial(a):
    """Wrap azimuth(s) into [0, 180)."""
    return np.mod(a, 180.0)

def axial_diff(a, b):
    """Smallest signed axial difference a-b, in [-90, 90)."""
    return (np.asarray(a) - np.asarray(b) + 90.0) % 180.0 - 90.0

def axial_mean(az):
    az = np.asarray(az, float)
    ang = np.deg2rad(2.0 * az)
    return wrap_axial(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0)

def axial_percentile_ci(boot_az, center, alpha=0.05):
    """Percentile CI for axial bootstrap sample, rotated around `center`."""
    d = axial_diff(np.asarray(boot_az, float), center)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return wrap_axial(center + lo), wrap_axial(center + hi), float(np.std(d, ddof=1))

# ----------------------------------------------------------------------
# projections (origin/projection sensitivity and systematic effects)
# ----------------------------------------------------------------------
def project_equirect(lat, lon, lat0, lon0):
    x = R_EARTH * np.cos(np.deg2rad(lat0)) * np.deg2rad(np.asarray(lon) - lon0)
    y = R_EARTH * np.deg2rad(np.asarray(lat) - lat0)
    return x, y

def project_azeq(lat, lon, lat0, lon0):
    """Azimuthal-equidistant projection (exact great-circle range & bearing)."""
    la, lo = np.deg2rad(np.asarray(lat)), np.deg2rad(np.asarray(lon))
    la0, lo0 = np.deg2rad(lat0), np.deg2rad(lon0)
    dlon = lo - lo0
    c = np.arccos(np.clip(np.sin(la0) * np.sin(la) + np.cos(la0) * np.cos(la) * np.cos(dlon), -1, 1))
    az = np.arctan2(np.sin(dlon) * np.cos(la),
                    np.cos(la0) * np.sin(la) - np.sin(la0) * np.cos(la) * np.cos(dlon))
    d = R_EARTH * c
    return d * np.sin(az), d * np.cos(az)

# ----------------------------------------------------------------------
# PCA azimuth (weighted and stratified variants)
# ----------------------------------------------------------------------
def pca_azimuth(x, y, w=None):
    X = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    if w is None:
        mu = X.mean(0)
        C = np.cov(X.T, bias=False)
    else:
        w = np.asarray(w, float)
        mu = np.average(X, axis=0, weights=w)
        Xc = X - mu
        C = (Xc * w[:, None]).T @ Xc / w.sum()
    evals, evecs = np.linalg.eigh(C)          # ascending
    v = evecs[:, -1]                          # PC1 (unit)
    if v[1] < 0 or (np.isclose(v[1], 0.0) and v[0] < 0):
        v = -v                                # orient +s toward north/NE
    az = wrap_axial(np.degrees(np.arctan2(v[0], v[1])))   # atan2(East, North)
    ev1 = float(evals[-1] / evals.sum())
    return {"az": float(az), "ev1_frac": ev1, "mu": mu, "v": v, "evals": evals}

def along_perp(x, y, mu, v):
    X = np.column_stack([x, y]) - mu
    n = np.array([-v[1], v[0]])
    return X @ v, X @ n           # s (along-axis), t (orthogonal residual)

def pca3d_geometry(x, y, z):
    """Return 3-D line and plane fits with descending covariance eigenvalues."""
    X = np.column_stack([x, y, -np.asarray(z, float)])   # up = +
    Xc = X - X.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Xc.T))
    v = evecs[:, -1]
    if v[2] > 0:
        v = -v                                            # plunge downward
    trend = wrap_axial(np.degrees(np.arctan2(v[0], v[1])))
    plunge = float(np.degrees(np.arcsin(-v[2] / np.linalg.norm(v))))
    nrm = evecs[:, 0]
    if nrm[2] < 0:
        nrm = -nrm
    strike = wrap_axial(np.degrees(np.arctan2(nrm[0], nrm[1])) + 90.0)
    dip = float(np.degrees(np.arccos(np.clip(abs(nrm[2]), 0, 1))))
    return {
        "line": {
            "trend": float(trend),
            "plunge": plunge,
            "ev1_frac": float(evals[-1] / evals.sum()),
        },
        "plane": {
            "strike": float(strike),
            "dip": dip,
            "planarity": float(1.0 - evals[0] / evals.sum()),
        },
        "eigenvalues": evals[::-1],
    }


def pca3d_line(x, y, z):
    """3-D PCA line fit. Returns trend (axial, deg) and plunge (deg)."""
    return pca3d_geometry(x, y, z)["line"]


def plane_fit(x, y, z):
    """Best-fit plane via smallest PCA eigenvector. Strike axial deg, dip deg."""
    return pca3d_geometry(x, y, z)["plane"]

# ----------------------------------------------------------------------
# bootstraps (clustered and error-perturbed resampling)
# ----------------------------------------------------------------------
def _stat_az(df, w=None):
    return pca_azimuth(df["x"].values, df["y"].values, w)["az"]

def boot_simple(df, n_boot=2000, rng=RNG):
    n = len(df)
    idx = rng.integers(0, n, size=(n_boot, n))
    x, y = df["x"].values, df["y"].values
    return np.array([pca_azimuth(x[i], y[i])["az"] for i in idx])

def boot_cluster(df, cluster_col, n_boot=1000, equal_weight=False, rng=RNG):
    """Resample whole clusters (days / episodes) with replacement.
    equal_weight=True additionally weights every drawn cluster equally
    (equal-time-bin weighting)."""
    groups = {k: g[["x", "y"]].values for k, g in df.groupby(cluster_col, observed=True)}
    keys = list(groups)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        if equal_weight:
            xs, ys, ws = [], [], []
            for j in pick:
                g = groups[keys[j]]
                xs.append(g[:, 0]); ys.append(g[:, 1])
                ws.append(np.full(len(g), 1.0 / len(g)))
            out[b] = pca_azimuth(np.concatenate(xs), np.concatenate(ys),
                                 np.concatenate(ws))["az"]
        else:
            g = np.vstack([groups[keys[j]] for j in pick])
            out[b] = pca_azimuth(g[:, 0], g[:, 1])["az"]
    return out

def boot_error_mc(df, n_mc=200, cluster_col=None, rng=RNG):
    """Monte-Carlo perturbation with per-event location errors.
    errH split isotropically; optionally combined with day-clustered resampling."""
    x, y = df["x"].values, df["y"].values
    sh = df["errH"].values / np.sqrt(2.0)
    out = np.empty(n_mc)
    if cluster_col is not None:
        codes = pd.Categorical(df[cluster_col]).codes
        keys = np.unique(codes)
    for b in range(n_mc):
        xp = x + rng.normal(0, sh)
        yp = y + rng.normal(0, sh)
        if cluster_col is None:
            out[b] = pca_azimuth(xp, yp)["az"]
        else:
            pick = rng.choice(keys, size=len(keys), replace=True)
            sel = np.concatenate([np.flatnonzero(codes == k) for k in pick])
            out[b] = pca_azimuth(xp[sel], yp[sel])["az"]
    return out

# ----------------------------------------------------------------------
# azimuth difference between catalogs
# ----------------------------------------------------------------------
def azdiff_bootstrap(dfA, dfB, cluster_col="day", n_boot=1000, rng=RNG):
    bA = boot_cluster(dfA, cluster_col, n_boot, rng=rng)
    bB = boot_cluster(dfB, cluster_col, n_boot, rng=rng)
    azA = _stat_az(dfA); azB = _stat_az(dfB)
    d0 = float(axial_diff(azA, azB))
    dd = axial_diff(bA, bB)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    # two-sided bootstrap p for H0: diff = 0
    p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
    p = min(1.0, max(p, 1.0 / n_boot))
    return {"azA": azA, "azB": azB, "diff": d0,
            "ci": [float(lo), float(hi)], "p_boot": float(p), "boot_diffs": dd}

def azdiff_equivalence(boot_diffs, bound_deg=15.0):
    """TOST-style equivalence vs a PRE-SPECIFIED bound.
    Equivalent if 90% CI of the difference lies inside +/-bound."""
    lo90, hi90 = np.percentile(boot_diffs, [5, 95])
    lo95, hi95 = np.percentile(boot_diffs, [2.5, 97.5])
    return {"bound": bound_deg,
            "ci90": [float(lo90), float(hi90)],
            "ci95": [float(lo95), float(hi95)],
            "equivalent_90": bool(lo90 > -bound_deg and hi90 < bound_deg),
            "equivalent_95": bool(lo95 > -bound_deg and hi95 < bound_deg)}

def azdiff_permutation(dfA, dfB, n_perm=1000, rng=RNG):
    """Day-stratified label permutation null for the azimuth difference."""
    d0 = abs(axial_diff(_stat_az(dfA), _stat_az(dfB)))
    both = pd.concat([dfA.assign(_lab=0), dfB.assign(_lab=1)], ignore_index=True)
    day_codes = pd.Categorical(both["day"]).codes
    labs = both["_lab"].values.copy()
    x, y = both["x"].values, both["y"].values
    order_by_day = [np.flatnonzero(day_codes == k) for k in np.unique(day_codes)]
    cnt = 0
    for _ in range(n_perm):
        lp = labs.copy()
        for idx in order_by_day:
            lp[idx] = rng.permutation(lp[idx])
        azA = pca_azimuth(x[lp == 0], y[lp == 0])["az"]
        azB = pca_azimuth(x[lp == 1], y[lp == 1])["az"]
        if abs(axial_diff(azA, azB)) >= d0:
            cnt += 1
    return {"obs_absdiff": float(d0), "p_perm": (cnt + 1) / (n_perm + 1)}

# ----------------------------------------------------------------------
# temporal trends with block resampling
# ----------------------------------------------------------------------
def daily_series(df, value_col, min_count=10):
    g = df.groupby("day", observed=True)[value_col].agg(["median", "size"]).sort_index()
    g = g[g["size"] >= min_count]  # Retain dates meeting the selected-event threshold.
    t = np.array([(d - g.index.min()).days for d in g.index], float)
    return t, g["median"].to_numpy(float)

def trend_basic(t, v):
    rho, p = stats.spearmanr(t, v)
    ts = stats.theilslopes(v, t)               # robust slope
    ols = stats.linregress(t, v)
    return {"spearman_rho": float(rho), "spearman_p": float(p),
            "theil_slope": float(ts.slope),
            "theil_lo": float(ts.low_slope), "theil_hi": float(ts.high_slope),
            "ols_slope": float(ols.slope), "n_days": int(len(t))}

def block_perm_p(t, v, block_len=5, n_perm=2000, rng=RNG):
    """Two-sided block-permutation p-value for Spearman rho, preserving
    within-block autocorrelation. Blocks contain `block_len` retained daily
    bins, which need not be consecutive calendar days."""
    rho0 = abs(stats.spearmanr(t, v).statistic)
    n = len(v)
    blocks = [np.arange(i, min(i + block_len, n)) for i in range(0, n, block_len)]
    cnt = 0
    for _ in range(n_perm):
        order = rng.permutation(len(blocks))
        vp = np.concatenate([v[blocks[j]] for j in order])
        if abs(stats.spearmanr(t[:len(vp)], vp).statistic) >= rho0:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)

def block_boot_slope(t, v, block_len=5, n_boot=2000, rng=RNG):
    """Residual moving-block bootstrap CI for the Theil-Sen slope.
    Blocks contain `block_len` retained daily bins, which need not be
    consecutive calendar days."""
    ts = stats.theilslopes(v, t)
    fit = ts.intercept + ts.slope * t
    res = v - fit
    n = len(v)
    nblk = int(np.ceil(n / block_len))
    starts_max = n - block_len
    out = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, max(starts_max + 1, 1), size=nblk)
        rb = np.concatenate([res[s:s + block_len] for s in st])[:n]
        out[b] = stats.theilslopes(fit + rb, t).slope
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"slope": float(ts.slope), "ci": [float(lo), float(hi)]}

def analyze_trend(df, value_col, block_len=5, block_scan=(1, 10),
                  n_perm=2000, rng=RNG, drop_first_day=False):
    t, v = daily_series(df, value_col)
    if drop_first_day:
        t, v = t[1:], v[1:]
    base = trend_basic(t, v)
    base["block_perm_p"] = float(block_perm_p(t, v, block_len, n_perm, rng))
    base["block_len"] = block_len
    scan = {}
    for L in range(block_scan[0], block_scan[1] + 1):
        scan[L] = float(block_perm_p(t, v, L, max(500, n_perm // 4), rng))
    base["block_scan_p"] = scan
    base.update(block_boot_slope(t, v, block_len, n_perm, rng))
    return base


def early_axis_cutoff(cfg):
    """Return the full-timestamp boundary between the early and late samples."""
    return pd.Timestamp(cfg["window"][0]) + pd.Timedelta(
        days=cfg.get("early_axis_days", 14)
    )


def early_axis_analysis(df, cfg, rng):
    """Estimate an early PCA axis and the subsequent daily migration trend.

    The input contains the selected catalog events within the analysis window.
    At least 201 events are required on each side of the time boundary.
    """
    axis_days = cfg.get("early_axis_days", 14)
    cutoff = early_axis_cutoff(cfg)
    early = df[df["t"] < cutoff]
    late = df[df["t"] >= cutoff].copy()
    if len(early) <= 200 or len(late) <= 200:
        return None
    axis = pca_azimuth(early["x"].values, early["y"].values)
    along, _ = along_perp(
        late["x"].values, late["y"].values, axis["mu"], axis["v"]
    )
    block_len = cfg.get("block_len", 5)
    n_perm = max(500, cfg.get("n_boot", {}).get("perm", 1000) // 2)
    trend = analyze_trend(
        late.assign(s_along=along),
        "s_along",
        block_len,
        (block_len, block_len),
        n_perm,
        rng,
    )
    return {
        "axis_days": axis_days,
        "cutoff_utc": cutoff.isoformat(),
        "early_az": axis["az"],
        "n_early": len(early),
        "n_late": len(late),
        "late_migration_theil": trend["theil_slope"],
        "late_migration_ci": trend["ci"],
        "late_block_perm_p": trend["block_perm_p"],
        "n_late_days": trend["n_days"],
    }


def multiplicity(pvals: dict, alpha=0.05):
    """Bonferroni + Benjamini-Hochberg over a pre-specified family."""
    keys = list(pvals)
    p = np.array([pvals[k] for k in keys], float)
    m = len(p)
    bonf = np.minimum(p * m, 1.0)
    order = np.argsort(p)
    bh = np.empty(m)
    prev = 1.0
    for rank, i in list(enumerate(order))[::-1]:
        val = min(prev, p[i] * m / (rank + 1))
        bh[i] = val
        prev = val
    return {k: {"p_raw": float(p[i]), "p_bonferroni": float(bonf[i]),
                "p_bh_fdr": float(bh[i]),
                "significant_bh_05": bool(bh[i] < alpha)}
            for i, k in enumerate(keys)}

# ----------------------------------------------------------------------
# event matching between catalogs
# ----------------------------------------------------------------------
def match_catalogs(dfA, dfB, tol_s=2.0):
    """Greedy origin-time matching within +/- tol_s seconds.
    Checks the insertion index and its two adjacent indices. Used candidates
    are skipped without searching farther within the tolerance."""
    tA = dfA["t"].values.astype("datetime64[ns]").astype("int64") / 1e9
    tB = dfB["t"].values.astype("datetime64[ns]").astype("int64") / 1e9
    oA, oB = np.argsort(tA), np.argsort(tB)
    tA_s, tB_s = tA[oA], tB[oB]
    usedB = np.zeros(len(tB_s), bool)
    pairs = []
    for i, ta in enumerate(tA_s):
        j = np.searchsorted(tB_s, ta)
        best, bestdt = -1, tol_s + 1
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < len(tB_s) and not usedB[jj]:
                dt = abs(tB_s[jj] - ta)
                if dt < bestdt:
                    best, bestdt = jj, dt
        if best >= 0 and bestdt <= tol_s:
            usedB[best] = True
            pairs.append((oA[i], oB[best], bestdt))
    ia = np.array([p[0] for p in pairs], int)
    ib = np.array([p[1] for p in pairs], int)
    return ia, ib, np.array([p[2] for p in pairs])

def deming(xv, yv, delta=1.0):
    """Deming regression y = a + b x with error-variance ratio delta = var(ey)/var(ex)."""
    xv, yv = np.asarray(xv, float), np.asarray(yv, float)
    mx, my = xv.mean(), yv.mean()
    sxx = np.mean((xv - mx) ** 2); syy = np.mean((yv - my) ** 2)
    sxy = np.mean((xv - mx) * (yv - my))
    b = (syy - delta * sxx + np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2)) / (2 * sxy)
    a = my - b * mx
    return float(a), float(b)

# ----------------------------------------------------------------------
# straight line vs curvature
# ----------------------------------------------------------------------
def _gauss_ic(res, k, n):
    s2 = np.mean(res ** 2)
    ll = -0.5 * n * (np.log(2 * np.pi * s2) + 1.0)
    return {"aic": 2 * k - 2 * ll, "bic": k * np.log(n) - 2 * ll, "rms": float(np.sqrt(s2))}

def geometry_models(s, tperp):
    """Compare straight line vs quadratic vs 1-break piecewise linear in the
    rotated (s, t) frame. Straight line == t ~ const (PCA axis)."""
    n = len(s)
    out = {}
    res0 = tperp - tperp.mean()
    out["line"] = _gauss_ic(res0, 2, n)                       # mean + var
    X1 = np.column_stack([np.ones(n), s])
    b1, *_ = np.linalg.lstsq(X1, tperp, rcond=None)
    out["tilted_line"] = _gauss_ic(tperp - X1 @ b1, 3, n)
    X2 = np.column_stack([np.ones(n), s, s ** 2])
    b2, *_ = np.linalg.lstsq(X2, tperp, rcond=None)
    out["quadratic"] = _gauss_ic(tperp - X2 @ b2, 4, n)
    out["quad_coeff"] = float(b2[2])
    # piecewise linear, break on interior 10-90% grid
    qs = np.percentile(s, np.linspace(10, 90, 17))
    best = None
    for c in qs:
        Xp = np.column_stack([np.ones(n), s, np.maximum(s - c, 0)])
        bp, *_ = np.linalg.lstsq(Xp, tperp, rcond=None)
        r = tperp - Xp @ bp
        ic = _gauss_ic(r, 5, n)                                # +break param
        if best is None or ic["aic"] < best[0]["aic"]:
            best = (ic, c)
    out["piecewise"] = best[0]; out["piecewise_break_s_km"] = float(best[1])
    a0 = out["line"]["aic"]
    out["delta_aic_vs_line"] = {k: float(out[k]["aic"] - a0)
                                for k in ("tilted_line", "quadratic", "piecewise")}
    return out

# ----------------------------------------------------------------------
# magnitudes & completeness
# ----------------------------------------------------------------------
def mc_maxc(mags, dm=0.1, correction=0.2):
    m = np.asarray(mags, float)
    bins = np.arange(np.floor(m.min() / dm) * dm, m.max() + dm, dm)
    hist, edges = np.histogram(m, bins)
    mc_raw = edges[np.argmax(hist)] + dm / 2
    return float(mc_raw), float(mc_raw + correction)

def b_value(mags, mc, dm=0.1):
    m = np.asarray(mags, float)
    m = m[m >= mc - dm / 2 + 1e-9]
    if len(m) < 30:
        return None
    b = np.log10(np.e) / (m.mean() - (mc - dm / 2))
    return {"b": float(b), "b_err_aki": float(b / np.sqrt(len(m))), "n": int(len(m))}

def mc_gft(mags, dm=0.1, conf=0.90):
    """Goodness-of-fit Mc (Wiemer & Wyss 2000)."""
    m = np.asarray(mags, float)
    trial = np.arange(np.round(m.min(), 1), np.round(m.max(), 1) - 0.5, dm)
    best = None
    for mc in trial:
        sel = m[m >= mc - dm / 2]
        if len(sel) < 50:
            break
        bv = b_value(m, mc, dm)
        if bv is None:
            continue
        a = np.log10(len(sel)) + bv["b"] * mc
        mags_bins = np.arange(mc, m.max() + dm, dm)
        pred = 10 ** (a - bv["b"] * mags_bins)
        obs = np.array([(m >= mb - dm / 2).sum() for mb in mags_bins], float)
        R = 100 * (1 - np.abs(obs - pred).sum() / obs.sum())
        if best is None:
            best = (mc, R)
        if R >= conf * 100:
            return float(mc), float(R)
    return (float(best[0]), float(best[1])) if best else (np.nan, np.nan)

# ----------------------------------------------------------------------
# Omori vs constant rate
# ----------------------------------------------------------------------
def omori_fit(t_days, T):
    """MLE of modified Omori K/(t+c)^p on event times t in (0, T] days."""
    t = np.asarray(t_days, float)
    t = t[(t > 0) & (t <= T)]
    n = len(t)

    def nll(theta):
        logK, logc, p = theta
        K, c = np.exp(logK), np.exp(logc)
        lam = K / (t + c) ** p
        if p == 1.0:
            I = K * (np.log(T + c) - np.log(c))
        else:
            I = K / (1 - p) * ((T + c) ** (1 - p) - c ** (1 - p))
        return -(np.sum(np.log(lam)) - I)

    x0 = np.array([np.log(n / 2.0), np.log(0.5), 1.1])
    r = optimize.minimize(nll, x0, method="Nelder-Mead",
                          options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
    K, c, p = np.exp(r.x[0]), np.exp(r.x[1]), r.x[2]
    aic_om = 2 * 3 + 2 * r.fun
    lam0 = n / T
    ll_const = n * np.log(lam0) - lam0 * T
    aic_const = 2 * 1 - 2 * ll_const
    return {"K": float(K), "c_days": float(c), "p": float(p),
            "aic_omori": float(aic_om), "aic_const": float(aic_const),
            "delta_aic_const_minus_omori": float(aic_const - aic_om), "n": n}

# ----------------------------------------------------------------------
# p-value formatting
# ----------------------------------------------------------------------
def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p = n/a"
    if p < 1e-4:
        return "p < 1.0 x 10^-4"
    if p < 1e-3:
        return f"p = {p:.1e}".replace("e-0", " x 10^-").replace("e-", " x 10^-")
    return f"p = {p:.2g}"
