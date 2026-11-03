"""Pick a cluster with a scheduling strategy using the latest carbon data,
then dispatch the workload there.

    python backend/dispatcher/live_scheduler.py --strategy persistence_predictive --type web
    python backend/dispatcher/live_scheduler.py --strategy cost_only --type cpu_bound --dry-run

Strategies: random_forest_predictive, persistence_predictive, carbon_only,
cost_only, reactive_carbon, round_robin.
Uses the newest hour that exists for all three zones in carbon_history.csv
(the collector appends this hourly). Forecasts come from the saved models.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # backend/ for ml_engine

import dispatch  # noqa: E402
from dispatch import sim  # noqa: E402
from ml_engine import prediction_service as ps  # noqa: E402

STRATEGIES = sim.BASE_STRATEGIES
RR_STATE_FILE = dispatch.PROJECT_ROOT / "results" / "live" / ".rr_state"


def latest_common_time(df):
    return min(df[df["zone"] == z]["datetime"].max() for z in ps.ZONES)


def build_inputs(providers, strategy):
    df = ps.load_historical_data()
    ts = latest_common_time(df)
    current, forecasts = {}, {}
    for p in providers.values():
        z = p["zone"]
        row = df[(df["zone"] == z) & (df["datetime"] == ts)]
        current[(ts, z)] = float(row["carbon_intensity"].iloc[0])
        forecasts[(ts, z, "persistence")] = float(
            ps.predict_persistence(z, ts)["predicted_carbon_intensity"])
        if strategy in ("random_forest_predictive", "carbon_only"):
            forecasts[(ts, z, "random_forest_all_features")] = float(
                ps.predict_historical(z, ts)["predicted_carbon_intensity"])
    return ts, current, forecasts


def next_rr_index():
    RR_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    i = int(RR_STATE_FILE.read_text()) if RR_STATE_FILE.exists() else 0
    RR_STATE_FILE.write_text(str(i + 1))
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--type", required=True, choices=dispatch.TYPES)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    providers, measured = sim.load_providers()
    ts, current, forecasts = build_inputs(providers, a.strategy)
    spec = dispatch.SPECS[a.type]
    remaining = {p: {"cpu": sim.USABLE_CPU_MILLICORES, "mem": sim.USABLE_MEMORY_MB}
                 for p in providers}
    provider, score = sim.choose(a.strategy, spec, remaining, providers,
                                 forecasts, current, ts,
                                 {"i": next_rr_index()})
    print(f"\nDecision time (UTC): {ts}")
    for p, info in providers.items():
        print(f"  {p}: zone={info['zone']} current={current[(ts, info['zone'])]:.0f} g/kWh")
    print(f"Strategy {a.strategy} -> {provider}\n")
    dispatch.show(dispatch.run_workload(provider, a.type, a.seconds,
                                        a.strategy, a.dry_run))


if __name__ == "__main__":
    main()