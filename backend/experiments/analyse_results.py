"""Analysis of the scheduling experiment (runs anywhere; no cluster needed).

Inputs : results/live/live_runs.csv      (measured jobs; profile + subset runs)
         results/live/live_subset_manifest.csv  (optional)
         rolling-validation forecasts, carbon history, pricing, latency, energy coefficients
Outputs: results/final/*.csv, results/final/*.png, results/final/summary.md

Method
  1. PROFILE  Measured per-(provider, workload type) behaviour from live runs:
              throughput, service latency, CPU utilisation, memory.
  2. REPLAY   At every hourly decision point (all of them, not a sample) every
              strategy places the workload mix. Carbon = modelled energy of the
              measured utilisation profile x real grid intensity of that hour.
  3. STATS    Paired comparisons across the same hours; 95% CI by day-block
              bootstrap (hours within a day are autocorrelated); Wilcoxon + paired t;
              Holm correction.
  4. SENSITIVITY  Web SLA factor and scoring weights.
  5. LIVE SUBSET  Metrics measured on the jobs that were actually executed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "backend" / "energy"))

import replay_common as rc  # noqa: E402
import energy_model as em  # noqa: E402

sim = rc.sim
LIVE_RUNS = ROOT / "results" / "live" / "live_runs.csv"
MANIFEST = ROOT / "results" / "live" / "live_subset_manifest.csv"
OUT = ROOT / "results" / "final"
TYPES = ["web", "cpu_bound", "mem_bound", "background"]
REF = "round_robin"
PREDICTIVE = ["persistence_predictive", "random_forest_predictive", "xgboost_predictive"]

# Light-mode palette (validated reference palette; slots 1-3 for providers)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8985"
PROV_COLOR = {"AWS": BLUE, "AZURE": ORANGE, "GCP": AQUA}


# ---------------------------------------------------------------- profile ----
def build_profile(min_runs):
    runs = pd.read_csv(LIVE_RUNS)
    ok = runs[runs["status"] == "ok"].copy()
    prof = ok[ok["strategy"] == "profile"]
    counts = prof.groupby(["provider", "workload_type"]).size()
    provisional = len(counts) < 12 or counts.min() < min_runs
    if provisional:
        print(f"\nWARNING: fewer than {min_runs} profile runs per pair; falling back to ALL "
              "successful runs. Results are PROVISIONAL until stage 'profile' has run.")
        prof = ok
    for col in ("cpu_seconds", "peak_cpu_m"):
        if col not in prof.columns:
            prof[col] = np.nan
    rows = []
    for (p, t), g in prof.groupby(["provider", "workload_type"]):
        spec = sim_spec(t)
        u = [em.utilisation(spec["cpu"], r.wall_seconds, r.cpu_seconds, r.peak_cpu_m)
             for r in g.itertuples()]
        src = pd.Series([s for _, s in u]).mode().iloc[0]
        rows.append({
            "provider": p, "workload_type": t, "runs": len(g),
            "util": float(np.mean([x for x, _ in u])), "util_source": src,
            "throughput_per_s": g["throughput_per_s"].mean(),
            "svc_p50_ms": pd.to_numeric(g["service_latency_p50_ms"], errors="coerce").mean(),
            "svc_p95_ms": pd.to_numeric(g["service_latency_p95_ms"], errors="coerce").mean(),
            "mem_peak_mb": pd.to_numeric(g.get("mem_peak_mb"), errors="coerce").mean()
            if "mem_peak_mb" in g else np.nan,
        })
    table = pd.DataFrame(rows).fillna({"svc_p50_ms": 0.0, "svc_p95_ms": 0.0})
    missing = {(p, t) for p in ("AWS", "AZURE", "GCP") for t in TYPES} - \
        set(zip(table["provider"], table["workload_type"]))
    if missing:
        raise SystemExit(f"No successful runs for: {sorted(missing)}")
    return table.set_index(["provider", "workload_type"]), provisional


def sim_spec(t):
    return next(w for w in sim.WORKLOAD_MIX if w["type"] == t)


# ----------------------------------------------------------------- replay ----
def replay(exp, profile, mix, strategies=None, weights=None):
    providers = exp["providers"]
    old = dict(sim.SCORING_WEIGHTS)
    if weights:
        sim.SCORING_WEIGHTS.update(weights)
    rows = []
    try:
        for s in (strategies or exp["strategies"]):
            rr = {"i": 0}
            for d in exp["decisions"].itertuples(index=False):
                placed = rc.place_decision(s, d.prediction_time, providers,
                                           exp["forecasts"], exp["current"], mix, rr)
                load = {p: 0 for p in providers}
                for n, wl, prov, _ in placed:
                    if prov is None:
                        rows.append({"strategy": s, "decision": d.prediction_time,
                                     "workload_type": wl["type"], "rejected": True})
                        continue
                    load[prov] += wl["cpu"]
                    pr = profile.loc[(prov, wl["type"])]
                    zone = providers[prov]["zone"]
                    watts = em.power_watts(prov, wl["cpu"], wl["mem"], pr["util"])
                    kwh = em.energy_kwh(watts, sim.DECISION_INTERVAL_HOURS)
                    intensity = exp["actual"][(d.target_time, zone)]
                    share = max(wl["cpu"] / sim.NODE_CPU_MILLICORES,
                                wl["mem"] / sim.NODE_MEMORY_MB)
                    net = providers[prov]["latency_ms"]
                    rows.append({
                        "strategy": s, "decision": d.prediction_time,
                        "workload_type": wl["type"], "provider": prov, "rejected": False,
                        "intensity": intensity, "energy_kwh": kwh,
                        "gco2e": em.gco2e(kwh, intensity),
                        "cost_usd": providers[prov]["price_per_hour"] * share
                                    * sim.DECISION_INTERVAL_HOURS,
                        "response_ms": net + pr["svc_p50_ms"],
                        "web_p95_ms": net + pr["svc_p95_ms"] if wl["type"] == "web" else np.nan,
                        "sla_ms": wl["sla_ms"],
                        "breach": (net + pr["svc_p50_ms"]) > wl["sla_ms"],
                        "pod_util": pr["util"],
                        "max_cluster_cpu_pct": np.nan,
                    })
                rows.append({"strategy": s, "decision": d.prediction_time,
                             "workload_type": "_load", "rejected": False,
                             "max_cluster_cpu_pct": 100 * max(load.values())
                             / sim.USABLE_CPU_MILLICORES})
    finally:
        sim.SCORING_WEIGHTS.clear()
        sim.SCORING_WEIGHTS.update(old)
    out = pd.DataFrame(rows)
    out["breach"] = pd.to_numeric(out["breach"], errors="coerce")  # 1.0 / 0.0 / NaN
    return out


def per_decision(df):
    placed = df[(df["workload_type"] != "_load") & (~df["rejected"].astype(bool))]
    g = placed.groupby(["strategy", "decision"])
    return pd.DataFrame({
        "gco2e": g["gco2e"].sum(), "cost_usd": g["cost_usd"].sum(),
        "breaches": g["breach"].sum(), "workloads": g["breach"].count(),
    }).reset_index()


# ------------------------------------------------------------------ stats ----
def day_bootstrap_ci(series, n=5000, seed=1):
    """95% CI of the mean, resampling whole UTC days."""
    days = series.index.floor("D")
    g = series.groupby(days)
    s, c = g.sum().values, g.count().values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(means, [2.5, 97.5])


def wide(pdc, col):
    return pdc.pivot(index="decision", columns="strategy", values=col).sort_index()


def paired(a, b, w):
    diff = (w[a] - w[b]).dropna()
    lo, hi = day_bootstrap_ci(diff)
    try:
        pw = stats.wilcoxon(diff).pvalue
    except ValueError:
        pw = 1.0
    pt = stats.ttest_rel(w[a].loc[diff.index], w[b].loc[diff.index]).pvalue \
        if diff.std() > 0 else 1.0
    return {"strategy": a, "versus": b, "mean_diff": diff.mean(),
            "pct_diff": 100 * diff.mean() / w[b].mean(), "ci_lo": lo, "ci_hi": hi,
            "p_wilcoxon": pw, "p_ttest": pt, "hours": len(diff)}


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def comparisons(strategies):
    pairs = [(s, REF) for s in strategies if s != REF]
    pairs += [("persistence_predictive", "reactive_carbon"),
              ("persistence_predictive", "carbon_only"),
              ("random_forest_predictive", "persistence_predictive"),
              ("xgboost_predictive", "persistence_predictive")]
    return [p for p in dict.fromkeys(pairs) if p[0] in strategies and p[1] in strategies]


# ------------------------------------------------------------ sensitivity ----
WEIGHT_SCENARIOS = {
    "default 0.40/0.25/0.20/0.15": {"carbon": .40, "cost": .25, "latency": .20, "resources": .15},
    "carbon-heavy": {"carbon": .70, "cost": .10, "latency": .10, "resources": .10},
    "cost-heavy": {"carbon": .10, "cost": .70, "latency": .10, "resources": .10},
    "latency-heavy": {"carbon": .10, "cost": .10, "latency": .70, "resources": .10},
    "equal": {"carbon": .25, "cost": .25, "latency": .25, "resources": .25},
    "carbon-only weights": {"carbon": 1.0, "cost": 0.0, "latency": 0.0, "resources": 0.0},
}
SLA_FACTORS = [1.0, 1.1, 1.2, 1.5, 2.0]
SENS_STRATEGIES = PREDICTIVE + ["carbon_only"]


def sens_row(label, df, extra):
    pdc = per_decision(df)
    g = pdc.groupby("strategy")
    out = pd.DataFrame({"mean_gco2e_per_hour": g["gco2e"].mean(),
                        "mean_cost_usd_per_hour": g["cost_usd"].mean(),
                        "sla_breach_rate_pct": 100 * g["breaches"].sum() / g["workloads"].sum()
                        }).reset_index()
    out.insert(0, extra[0], label)
    return out


# ----------------------------------------------------------------- charts ----
def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def new_fig(w=8.2, h=4.2):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    style(ax)
    return plt, fig, ax


def finish(plt, fig, name, title, sub):
    fig.text(0.01, 0.975, title, fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.925, sub, fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def group_color(s):
    return BLUE if s in PREDICTIVE else GRAY


def chart_carbon(summary):
    plt, fig, ax = new_fig()
    d = summary.sort_values("mean_gco2e_per_hour", ascending=False)
    y = np.arange(len(d))
    ax.barh(y, d["mean_gco2e_per_hour"], height=0.55,
            color=[group_color(s) for s in d["strategy"]],
            xerr=[d["mean_gco2e_per_hour"] - d["ci_lo"], d["ci_hi"] - d["mean_gco2e_per_hour"]],
            error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
    for yi, (v, hi) in enumerate(zip(d["mean_gco2e_per_hour"], d["ci_hi"])):
        ax.text(hi + 0.02 * d["ci_hi"].max(), yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, d["strategy"], color=INK)
    ax.set_xlabel("gCO2e per scheduling hour (6 workloads)", color=INK2, fontsize=9)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=GRAY)],
              labels=["predictive (multi-objective)", "baseline"], frameon=False,
              fontsize=9, loc="upper right", labelcolor=INK2)
    finish(plt, fig, "carbon_by_strategy.png", "Carbon emitted per scheduling hour",
           "Mean over all decision hours; whiskers = 95% CI (day-block bootstrap)")


def chart_tradeoff(summary):
    plt, fig, ax = new_fig()
    ax.grid(axis="y", color=GRID, lw=0.6)
    offsets = {"carbon_only": (22, -34), "reactive_carbon": (9, 8), "cost_only": (8, 6),
               "round_robin": (-8, -30)}
    for r in summary.itertuples():
        ax.scatter(r.mean_cost_usd_per_hour, r.mean_gco2e_per_hour, s=70,
                   color=group_color(r.strategy), edgecolor=SURFACE, lw=2, zorder=3)
        if r.strategy in PREDICTIVE:
            continue
        dx, dy = offsets.get(r.strategy, (8, 6))
        ax.annotate(f"{r.strategy}\nSLA breach {r.sla_breach_rate_pct:.1f}%",
                    (r.mean_cost_usd_per_hour, r.mean_gco2e_per_hour), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color=INK,
                    ha="right" if dx < 0 else "left",
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    pr = summary[summary["strategy"].isin(PREDICTIVE)]
    ax.annotate("3 predictive strategies (overlap)\nSLA breach "
                + "/".join(f"{v:.1f}" for v in pr["sla_breach_rate_pct"]) + "%",
                (pr["mean_cost_usd_per_hour"].mean(), pr["mean_gco2e_per_hour"].mean()),
                xytext=(12, 16), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("USD per scheduling hour", color=INK2, fontsize=9)
    ax.set_ylabel("gCO2e per scheduling hour", color=INK2, fontsize=9)
    finish(plt, fig, "carbon_cost_tradeoff.png", "Carbon vs cost trade-off",
           "Lower-left is better; label shows the share of workloads that breached their latency SLA")


def chart_placement(df):
    plt, fig, ax = new_fig()
    placed = df[(df["workload_type"] != "_load") & (~df["rejected"].astype(bool))]
    share = (placed.groupby(["strategy", "provider"]).size()
             .unstack(fill_value=0).pipe(lambda t: 100 * t.div(t.sum(axis=1), axis=0)))
    share = share.reindex(columns=["AWS", "AZURE", "GCP"], fill_value=0)
    share = share.sort_values("GCP")
    left = np.zeros(len(share))
    for p in share.columns:
        ax.barh(share.index, share[p], left=left, color=PROV_COLOR[p], height=0.6,
                edgecolor=SURFACE, lw=2, label=f"{p} ({ {'AWS':'IE','AZURE':'DE','GCP':'FR'}[p] })")
        for i, (l, v) in enumerate(zip(left, share[p])):
            if v >= 8:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="#ffffff")
        left += share[p].values
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of workloads placed (%)", color=INK2, fontsize=9)
    ax.tick_params(axis="y", colors=INK)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.12), labelcolor=INK2)
    finish(plt, fig, "placement_share.png", "Where each strategy places workloads",
           "AWS = Ireland grid, Azure = Germany grid, GCP = France grid (assumed zone mapping)")


def chart_sla(sens):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), facecolor=SURFACE)
    spec = {"persistence_predictive": (BLUE, "o", "-"), "random_forest_predictive": (ORANGE, "s", "--"),
            "xgboost_predictive": (AQUA, "^", ":"), "carbon_only": (YELLOW, "D", "-")}
    for ax, col, lab in ((axes[0], "mean_gco2e_per_hour", "gCO2e per hour"),
                         (axes[1], "sla_breach_rate_pct", "SLA breach rate (%)")):
        style(ax)
        for s, (c, m, ls) in spec.items():
            d = sens[sens["strategy"] == s].sort_values("sla_factor")
            ax.plot(d["sla_factor"], d[col], color=c, marker=m, ls=ls, lw=2, ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=s)
        ax.set_xlabel("web SLA factor (x lowest latency)", color=INK2, fontsize=9)
        ax.set_ylabel(lab, color=INK2, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    finish(plt, fig, "sensitivity_sla.png", "Sensitivity to the latency SLA",
           "Tighter SLA forces more workloads onto the nearest region")


def chart_weights(sw):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4), facecolor=SURFACE, sharey=True)
    d = sw[sw["strategy"] == "persistence_predictive"].reset_index(drop=True)
    for ax, col, lab in zip(axes, ["mean_gco2e_per_hour", "mean_cost_usd_per_hour", "sla_breach_rate_pct"],
                            ["gCO2e per hour", "USD per hour", "SLA breach (%)"]):
        style(ax)
        ax.barh(d["scenario"], d[col], color=BLUE, height=0.55)
        if d[col].max() == 0:
            ax.set_xlim(0, 1)
            ax.text(0.5, 0.5, "0 in every scenario", transform=ax.transAxes, ha="center",
                    va="center", fontsize=10, color=INK2)
        ax.set_xlabel(lab, color=INK2, fontsize=9)
        ax.invert_yaxis()
    axes[0].tick_params(axis="y", colors=INK)
    finish(plt, fig, "sensitivity_weights.png", "Sensitivity to the scoring weights",
           "persistence_predictive under different carbon/cost/latency/resource weights")


# ------------------------------------------------------------------ live ----
def live_subset(exp, profile_mode):
    if not MANIFEST.exists():
        return None
    m = pd.read_csv(MANIFEST)
    runs = pd.read_csv(LIVE_RUNS)
    j = m.merge(runs, on="run_id", how="left", suffixes=("", "_run"))
    j["ok"] = j["status_run"].eq("ok")
    rows = []
    for r in j.itertuples():
        g = np.nan
        if r.ok and pd.notna(r.actual_carbon):
            spec = sim_spec(r.workload_type)
            u, _ = em.utilisation(spec["cpu"], r.wall_seconds,
                                  getattr(r, "cpu_seconds", None), r.peak_cpu_m)
            kwh = em.energy_kwh(em.power_watts(r.provider, spec["cpu"], spec["mem"], u))
            g = em.gco2e(kwh, r.actual_carbon)
        rows.append(g)
    j["gco2e"] = rows
    per = j.groupby(["strategy", "decision_time"]).agg(
        gco2e=("gco2e", "sum"), jobs=("ok", "size"), ok=("ok", "sum"),
        breaches=("sla_breach", lambda s: (s.astype(str) == "True").sum())).reset_index()
    out = per.groupby("strategy").agg(
        origins=("decision_time", "nunique"), jobs=("jobs", "sum"), ok_jobs=("ok", "sum"),
        mean_gco2e_per_hour=("gco2e", "mean"), sd_gco2e=("gco2e", "std"),
        breaches=("breaches", "sum")).reset_index()
    out["sla_breach_rate_pct"] = 100 * out["breaches"] / out["ok_jobs"]
    tp = (j[j["ok"]].groupby(["strategy", "workload_type"])["throughput_per_s"].mean()
          .unstack().add_prefix("throughput_"))
    return out.merge(tp.reset_index(), on="strategy", how="left"), j


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-profile-runs", type=int, default=3)
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--live-runs", help="override path to live_runs.csv (testing)")
    ap.add_argument("--out", help="override output folder (testing)")
    a = ap.parse_args()
    global LIVE_RUNS, OUT
    if a.live_runs:
        LIVE_RUNS = Path(a.live_runs)
    if a.out:
        OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)

    exp = rc.load_experiment()
    providers, strategies = exp["providers"], exp["strategies"]
    profile, provisional = build_profile(a.min_profile_runs)
    profile.round(4).to_csv(OUT / "measured_profile.csv")
    print("\nMeasured profile (per provider x workload):")
    print(profile.round(3).to_string())
    if (profile["util_source"] != "cgroup").any():
        print("NOTE: some utilisation values come from kubectl top, not cgroup CPU time.")

    df = replay(exp, profile, sim.WORKLOAD_MIX)
    pdc = per_decision(df)
    pdc.to_csv(OUT / "per_decision_totals.csv", index=False)
    wg, wc, wb = wide(pdc, "gco2e"), wide(pdc, "cost_usd"), wide(pdc, "breaches")

    placed = df[(df["workload_type"] != "_load") & (~df["rejected"].astype(bool))]
    rows = []
    for s in strategies:
        lo, hi = day_bootstrap_ci(wg[s])
        pl = placed[placed["strategy"] == s]
        rows.append({
            "strategy": s, "decisions": len(wg), "workloads": len(pl),
            "rejected": int(df[(df["strategy"] == s)]["rejected"].astype(bool).sum()),
            "mean_gco2e_per_hour": wg[s].mean(), "ci_lo": lo, "ci_hi": hi,
            "mean_cost_usd_per_hour": wc[s].mean(),
            "mean_energy_wh_per_hour": 1000 * pl.groupby("decision")["energy_kwh"].sum().mean(),
            "sla_breach_rate_pct": 100 * pl["breach"].mean(),
            "mean_response_ms": pl["response_ms"].mean(),
            "web_p95_ms": np.nanpercentile(pl["web_p95_ms"], 95),
            "mean_pod_cpu_util_pct": 100 * pl["pod_util"].mean(),
            "mean_max_cluster_cpu_pct": df[(df["strategy"] == s) & (df["workload_type"] == "_load")]
            ["max_cluster_cpu_pct"].mean(),
            **{f"pct_{p}": 100 * (pl["provider"] == p).mean() for p in ("AWS", "AZURE", "GCP")},
        })
    summary = pd.DataFrame(rows)
    ref = summary.loc[summary["strategy"] == REF, "mean_gco2e_per_hour"].iloc[0]
    summary["gco2e_vs_round_robin_pct"] = 100 * (summary["mean_gco2e_per_hour"] - ref) / ref
    summary = summary.sort_values("mean_gco2e_per_hour").reset_index(drop=True)
    summary.to_csv(OUT / "strategy_summary.csv", index=False)

    comp = []
    for metric, w in (("gco2e", wg), ("cost_usd", wc)):
        for x, y in comparisons(strategies):
            r = paired(x, y, w)
            r["metric"] = metric
            comp.append(r)
    comp = pd.DataFrame(comp)
    for metric in comp["metric"].unique():
        mask = comp["metric"] == metric
        comp.loc[mask, "p_holm"] = holm(comp.loc[mask, "p_wilcoxon"].values)
    comp.to_csv(OUT / "paired_comparisons.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nSTRATEGY SUMMARY")
    print(summary.round(3).to_string(index=False))
    print("\nPAIRED COMPARISONS (gCO2e per hour)")
    print(comp[comp["metric"] == "gco2e"].drop(columns="metric").round(4).to_string(index=False))

    chart_carbon(summary)
    chart_tradeoff(summary)
    chart_placement(df)

    sens_sla = sens_w = None
    if not a.skip_sensitivity:
        parts = []
        for f in SLA_FACTORS:
            r = replay(exp, profile, rc.mix_with_web_sla(f, providers), SENS_STRATEGIES)
            parts.append(sens_row(f, r, ("sla_factor",)))
        sens_sla = pd.concat(parts)
        sens_sla.to_csv(OUT / "sensitivity_sla.csv", index=False)
        parts = []
        for name, wts in WEIGHT_SCENARIOS.items():
            r = replay(exp, profile, sim.WORKLOAD_MIX, SENS_STRATEGIES, weights=wts)
            parts.append(sens_row(name, r, ("scenario",)))
        sens_w = pd.concat(parts)
        sens_w.to_csv(OUT / "sensitivity_weights.csv", index=False)
        chart_sla(sens_sla)
        chart_weights(sens_w)
        print("\nSLA SENSITIVITY");  print(sens_sla.round(3).to_string(index=False))
        print("\nWEIGHT SENSITIVITY"); print(sens_w.round(3).to_string(index=False))

    live = live_subset(exp, provisional)
    if live is not None:
        live[0].to_csv(OUT / "live_subset_summary.csv", index=False)
        print("\nLIVE SUBSET (measured jobs)"); print(live[0].round(3).to_string(index=False))

    write_summary_md(summary, comp, provisional, live is not None)
    print(f"\nSaved tables, charts and summary.md to {OUT}")


def write_summary_md(summary, comp, provisional, has_live):
    best = summary.iloc[0]
    rr = summary[summary["strategy"] == REF].iloc[0]
    lines = ["# Results summary (auto-generated)", ""]
    if provisional:
        lines += ["> PROVISIONAL: measured profile is based on few runs. Re-run after stage 'profile'.", ""]
    lines += [
        f"* Decision hours replayed: {int(summary['decisions'].iloc[0])}; strategies: {len(summary)}.",
        f"* Lowest carbon: **{best['strategy']}** at {best['mean_gco2e_per_hour']:.3f} gCO2e/hour "
        f"({best['gco2e_vs_round_robin_pct']:.1f}% vs round_robin {rr['mean_gco2e_per_hour']:.3f}).",
        "", "## gCO2e per hour", "", "| strategy | gCO2e/h | 95% CI | vs RR % | cost USD/h | SLA breach % | web p95 ms |",
        "|---|---|---|---|---|---|---|"]
    for r in summary.itertuples():
        lines.append(f"| {r.strategy} | {r.mean_gco2e_per_hour:.3f} | {r.ci_lo:.3f}-{r.ci_hi:.3f} | "
                     f"{r.gco2e_vs_round_robin_pct:.1f} | {r.mean_cost_usd_per_hour:.4f} | "
                     f"{r.sla_breach_rate_pct:.1f} | {r.web_p95_ms:.1f} |")
    lines += ["", "## Paired comparisons (gCO2e per hour, Holm-adjusted Wilcoxon)", "",
              "| strategy | versus | mean diff | % | 95% CI | p (Holm) |", "|---|---|---|---|---|---|"]
    for r in comp[comp["metric"] == "gco2e"].itertuples():
        lines.append(f"| {r.strategy} | {r.versus} | {r.mean_diff:.4f} | {r.pct_diff:.1f} | "
                     f"{r.ci_lo:.4f} to {r.ci_hi:.4f} | {r.p_holm:.4f} |")
    lines += ["", "Charts: carbon_by_strategy.png, carbon_cost_tradeoff.png, placement_share.png, "
              "sensitivity_sla.png, sensitivity_weights.png."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
