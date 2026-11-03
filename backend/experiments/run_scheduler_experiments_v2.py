"""
Offline Carbon-Aware Multi-Cloud Scheduling Experiment (v2).

What changed from v1
--------------------
* Cost comes from config/pricing.json (real on-demand prices), not made-up numbers.
* Latency comes from config/latency.json (run scripts/measure_latency.py to make
  it a real measurement; until then it holds clearly-labelled placeholders).
* Each hour a MIX of workloads arrives (web, CPU-bound, memory-bound,
  background), each with its own latency SLA and resource request.
* Clusters have finite capacity. When the greenest/cheapest cluster is full,
  the remaining workloads must go elsewhere, so single-objective strategies
  no longer collapse into the same choice.
* Single-objective baselines (carbon-only, cost-only, reactive, round-robin)
  ignore the latency SLA by definition, so they can BREACH it.
  The multi-objective strategies treat the SLA as a hard constraint.
* New metrics: SLA breach rate, p95 latency, estimated gCO2e, total cost,
  rejected workloads.

Still offline / not yet a live deployment: this replays historical carbon data
and the rolling-validation forecasts. Live dispatch to clusters is Day 2.

Run from the project root:
    python backend/experiments/run_scheduler_experiments_v2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_scheduler_experiments as v1  # reuse the validated data loaders

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICING_FILE = PROJECT_ROOT / "config" / "pricing.json"
LATENCY_FILE = PROJECT_ROOT / "config" / "latency.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments_v2"

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Usable capacity per cluster = 2 vCPU / 4 GB node minus system reserve.
NODE_CPU_MILLICORES = 2000
NODE_MEMORY_MB = 4096
USABLE_CPU_MILLICORES = 1600
USABLE_MEMORY_MB = 3200

# One scheduling round per hour; every workload runs for one hour.
DECISION_INTERVAL_HOURS = 1.0

# ASSUMPTION (replace on Day 3 with measured values): average electrical power
# drawn per requested vCPU, including a share of server overhead.
WATTS_PER_VCPU = 10.0

# One hour's arrivals. Latency SLAs are in ms (round-trip from the client).
WORKLOAD_MIX = [
    {"type": "web",        "cpu": 250,  "mem": 256,  "sla_ms": 40.0},
    {"type": "web",        "cpu": 250,  "mem": 256,  "sla_ms": 40.0},
    {"type": "cpu_bound",  "cpu": 1000, "mem": 512,  "sla_ms": 150.0},
    {"type": "mem_bound",  "cpu": 500,  "mem": 1500, "sla_ms": 150.0},
    {"type": "background", "cpu": 250,  "mem": 256,  "sla_ms": 500.0},
    {"type": "background", "cpu": 250,  "mem": 256,  "sla_ms": 500.0},
]

SCORING_WEIGHTS = {"carbon": 0.40, "cost": 0.25, "latency": 0.20, "resources": 0.15}

BASE_STRATEGIES = [
    "random_forest_predictive",
    "persistence_predictive",
    "carbon_only",
    "cost_only",
    "reactive_carbon",
    "round_robin",
]
# Strategies that weigh several objectives and treat the SLA as a hard limit.
MULTI_OBJECTIVE = {
    "random_forest_predictive": "random_forest_all_features",
    "persistence_predictive": "persistence",
    "xgboost_predictive": "xgboost_all_features",
}


# ---------------------------------------------------------------------------
# Provider configuration (from files)
# ---------------------------------------------------------------------------

def load_providers():
    pricing = json.loads(PRICING_FILE.read_text())
    latency = json.loads(LATENCY_FILE.read_text())

    providers = {}
    for name, info in pricing["providers"].items():
        providers[name] = {
            "zone": info["carbon_zone"],
            "price_per_hour": float(info["price_per_hour_usd"]),
            "latency_ms": float(latency["latency_ms"][name]),
        }

    print("\nProvider configuration")
    print(f"  prices retrieved: {pricing['retrieved']} ({pricing['currency']})")
    print(f"  latency measured: {latency.get('measured', False)}")
    if not latency.get("measured", False):
        print("  WARNING: latency values are PLACEHOLDERS. "
              "Run scripts/measure_latency.py.")
    for name, p in providers.items():
        print(f"  {name}: zone={p['zone']} ${p['price_per_hour']}/h "
              f"latency={p['latency_ms']} ms")
    return providers, bool(latency.get("measured", False))


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def has_capacity(remaining, provider, workload):
    return (remaining[provider]["cpu"] >= workload["cpu"]
            and remaining[provider]["mem"] >= workload["mem"])


def weighted_scores(candidates, carbon_values, workload, remaining, providers):
    cost = {p: providers[p]["price_per_hour"] for p in candidates}
    lat = {p: providers[p]["latency_ms"] for p in candidates}
    headroom = {
        p: min((remaining[p]["cpu"] - workload["cpu"]) / USABLE_CPU_MILLICORES,
               (remaining[p]["mem"] - workload["mem"]) / USABLE_MEMORY_MB)
        for p in candidates
    }
    c_s = v1.min_max_normalise({p: carbon_values[p] for p in candidates})
    k_s = v1.min_max_normalise(cost)
    l_s = v1.min_max_normalise(lat)
    r_s = v1.min_max_normalise(headroom, higher_is_better=True)
    w = SCORING_WEIGHTS
    return {p: w["carbon"] * c_s[p] + w["cost"] * k_s[p]
               + w["latency"] * l_s[p] + w["resources"] * r_s[p]
            for p in candidates}


def choose(strategy, workload, remaining, providers, forecasts, current,
           timestamp, rr_state):
    """Return (provider or None, score or None)."""
    fits = [p for p in providers if has_capacity(remaining, p, workload)]
    if not fits:
        return None, None

    def forecast(p, model):
        return forecasts[(timestamp, providers[p]["zone"], model)]

    if strategy in MULTI_OBJECTIVE:
        model = MULTI_OBJECTIVE[strategy]
        # SLA is a hard constraint for multi-objective strategies.
        ok = [p for p in fits
              if providers[p]["latency_ms"] <= workload["sla_ms"]]
        candidates = ok or fits  # if nothing meets the SLA, fall back (will breach)
        carbon = {p: forecast(p, model) for p in candidates}
        scores = weighted_scores(candidates, carbon, workload, remaining, providers)
        best = max(candidates, key=lambda p: scores[p])
        return best, scores[best]

    # Single-objective baselines: capacity is respected, SLA is not.
    if strategy == "carbon_only":
        return min(fits, key=lambda p: forecast(p, "random_forest_all_features")), None
    if strategy == "cost_only":
        return min(fits, key=lambda p: providers[p]["price_per_hour"]), None
    if strategy == "reactive_carbon":
        return min(fits, key=lambda p: current[(timestamp, providers[p]["zone"])]), None
    if strategy == "round_robin":
        names = list(providers)
        for _ in range(len(names)):
            candidate = names[rr_state["i"] % len(names)]
            rr_state["i"] += 1
            if candidate in fits:
                return candidate, None
    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run(experiment_data, providers):
    forecasts, current, actual, _ = v1.build_experiment_lookups(experiment_data)

    available_models = set(experiment_data["model"].unique())
    strategies = list(BASE_STRATEGIES)
    if "xgboost_all_features" in available_models:
        strategies.insert(2, "xgboost_predictive")

    decisions = (
        experiment_data[experiment_data["model"] == "persistence"]
        [["fold", "prediction_time", "target_time"]]
        .drop_duplicates().sort_values("prediction_time").reset_index(drop=True)
    )

    rows = []
    for strategy in strategies:
        rr_state = {"i": 0}
        for d in decisions.itertuples(index=False):
            remaining = {p: {"cpu": USABLE_CPU_MILLICORES, "mem": USABLE_MEMORY_MB}
                         for p in providers}
            for n, wl in enumerate(WORKLOAD_MIX):
                provider, score = choose(strategy, wl, remaining, providers,
                                         forecasts, current, d.prediction_time,
                                         rr_state)
                row = {
                    "strategy": strategy, "fold": d.fold,
                    "prediction_time": d.prediction_time,
                    "target_time": d.target_time,
                    "workload_index": n, "workload_type": wl["type"],
                    "cpu_millicores": wl["cpu"], "memory_mb": wl["mem"],
                    "sla_ms": wl["sla_ms"],
                }
                if provider is None:
                    row.update({"selected_provider": None, "rejected": True})
                    rows.append(row)
                    continue

                remaining[provider]["cpu"] -= wl["cpu"]
                remaining[provider]["mem"] -= wl["mem"]
                zone = providers[provider]["zone"]
                intensity = actual[(d.target_time, zone)]
                share = max(wl["cpu"] / NODE_CPU_MILLICORES,
                            wl["mem"] / NODE_MEMORY_MB)
                energy_kwh = (wl["cpu"] / 1000.0) * WATTS_PER_VCPU / 1000.0 \
                    * DECISION_INTERVAL_HOURS
                row.update({
                    "selected_provider": provider, "selected_zone": zone,
                    "rejected": False,
                    "forecast_carbon": forecasts.get(
                        (d.prediction_time, zone, "random_forest_all_features")),
                    "current_carbon": current[(d.prediction_time, zone)],
                    "actual_carbon_intensity": intensity,
                    "latency_ms": providers[provider]["latency_ms"],
                    "sla_breach": providers[provider]["latency_ms"] > wl["sla_ms"],
                    "cost_usd": providers[provider]["price_per_hour"] * share
                                * DECISION_INTERVAL_HOURS,
                    "energy_kwh": energy_kwh,
                    "gco2e": energy_kwh * intensity,
                    "weighted_score": score,
                })
                rows.append(row)

    return pd.DataFrame(rows), strategies


def summarise(df, strategies):
    out = []
    for s in strategies:
        g = df[df["strategy"] == s]
        placed = g[~g["rejected"]]
        out.append({
            "strategy": s,
            "workloads": len(g),
            "rejected": int(g["rejected"].sum()),
            "sla_breach_rate_pct": 100 * placed["sla_breach"].mean(),
            "total_gco2e": placed["gco2e"].sum(),
            "mean_carbon_intensity": placed["actual_carbon_intensity"].mean(),
            "total_cost_usd": placed["cost_usd"].sum(),
            "mean_latency_ms": placed["latency_ms"].mean(),
            "p95_latency_ms": np.percentile(placed["latency_ms"], 95),
            "pct_AWS": 100 * (placed["selected_provider"] == "AWS").mean(),
            "pct_AZURE": 100 * (placed["selected_provider"] == "AZURE").mean(),
            "pct_GCP": 100 * (placed["selected_provider"] == "GCP").mean(),
        })
    return pd.DataFrame(out).sort_values("total_gco2e").reset_index(drop=True)


def main():
    print("=" * 65)
    print("CARBON-AWARE MULTI-CLOUD SCHEDULING EXPERIMENT v2")
    print("=" * 65)

    providers, latency_measured = load_providers()
    predictions = v1.load_and_validate_predictions()
    carbon = v1.load_historical_observations()
    data = v1.attach_current_observations(predictions, carbon)

    df, strategies = run(data, providers)
    summary = summarise(df, strategies)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "scheduler_decisions_v2.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "scheduler_strategy_summary_v2.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 65)
    print("STRATEGY SUMMARY  (sorted by total gCO2e)")
    print("=" * 65)
    print(summary.round(2).to_string(index=False))
    if not latency_measured:
        print("\nNOTE: latency values are placeholders; SLA results are "
              "provisional until scripts/measure_latency.py has been run.")
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()