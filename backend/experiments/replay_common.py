"""Shared loading and placement logic for the live and replay experiments.

Decision points are the hourly forecast origins in
results/ml/rolling_validation_predictions.csv (all strategies see the same
origins, the same carbon data, prices and latencies).
XGBoost is included automatically when its predictions are present.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_scheduler_experiments as v1  # noqa: E402
import run_scheduler_experiments_v2 as sim  # noqa: E402

EXTRA_MODEL = "xgboost_all_features"


def load_experiment():
    """Return dict with providers, decisions, forecasts, current, actual, strategies."""
    if EXTRA_MODEL not in v1.FORECAST_MODELS:
        v1.FORECAST_MODELS = list(v1.FORECAST_MODELS) + [EXTRA_MODEL]
    providers, _ = sim.load_providers()
    predictions = v1.load_and_validate_predictions()
    carbon = v1.load_historical_observations()
    data = v1.attach_current_observations(predictions, carbon)
    forecasts, current, actual, _ = v1.build_experiment_lookups(data)

    strategies = list(sim.BASE_STRATEGIES)
    strategies.insert(2, "xgboost_predictive")

    decisions = (
        data[data["model"] == "persistence"]
        [["fold", "prediction_time", "target_time"]]
        .drop_duplicates().sort_values("prediction_time").reset_index(drop=True)
    )
    return {"providers": providers, "decisions": decisions,
            "forecasts": forecasts, "current": current, "actual": actual,
            "strategies": strategies}


def place_decision(strategy, prediction_time, providers, forecasts, current,
                   mix, rr_state):
    """Place one hour's workload mix. Returns [(index, workload, provider|None, score)]."""
    remaining = {p: {"cpu": sim.USABLE_CPU_MILLICORES, "mem": sim.USABLE_MEMORY_MB}
                 for p in providers}
    out = []
    for n, wl in enumerate(mix):
        provider, score = sim.choose(strategy, wl, remaining, providers,
                                     forecasts, current, prediction_time,
                                     rr_state)
        if provider is not None:
            remaining[provider]["cpu"] -= wl["cpu"]
            remaining[provider]["mem"] -= wl["mem"]
        out.append((n, wl, provider, score))
    return out


def mix_with_web_sla(factor, providers):
    """WORKLOAD_MIX with the web SLA = factor x lowest network latency."""
    web_sla = round(factor * min(p["latency_ms"] for p in providers.values()), 1)
    mix = []
    for wl in sim.WORKLOAD_MIX:
        w = dict(wl)
        if w["type"] == "web":
            w["sla_ms"] = web_sla
        mix.append(w)
    return mix
