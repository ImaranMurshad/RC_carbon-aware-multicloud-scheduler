
"""
Offline Carbon-Aware Multi-Cloud Scheduling Experiment.

Simulates six scheduling strategies using historical rolling-validation
predictions and historical carbon-intensity observations.

The experiment:
- Does not modify the Django database.
- Does not deploy workloads.
- Uses illustrative provider cost, latency and capacity assumptions.
- Reports regional carbon intensity as a proxy, not measured workload emissions.
- Uses identical decision timestamps for all strategies.
- Allows weighted_score to be empty for strategies that do not use
  multi-objective weighted scoring.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "ml"
    / "rolling_validation_predictions.csv"
)

CARBON_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "historical"
    / "carbon_history.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiments"
)


# =========================================================
# 2. DATA CONFIGURATION
# =========================================================

ZONES = ["IE", "DE", "FR"]

FORECAST_MODELS = [
    "persistence",
    "random_forest_all_features",
]

REQUIRED_PREDICTION_COLUMNS = {
    "zone",
    "fold",
    "model",
    "prediction_time",
    "target_time",
    "actual_carbon",
    "prediction",
    "target_is_estimated",
}

REQUIRED_CARBON_COLUMNS = {
    "datetime",
    "zone",
    "carbon_intensity",
    "is_estimated",
}


# =========================================================
# 3. EXPERIMENTAL PROVIDER CONFIGURATION
# =========================================================

# Illustrative scenario assumptions.
# These are not verified provider prices, production capacities,
# or measured network latency values.

PROVIDERS = {
    "AWS": {
        "zone": "IE",
        "cost_per_hour": 0.08,
        "latency_ms": 20.0,
        "available_cpu_millicores": 2000,
        "available_memory_mb": 4096,
    },
    "AZURE": {
        "zone": "DE",
        "cost_per_hour": 0.09,
        "latency_ms": 35.0,
        "available_cpu_millicores": 1500,
        "available_memory_mb": 3072,
    },
    "GCP": {
        "zone": "FR",
        "cost_per_hour": 0.07,
        "latency_ms": 45.0,
        "available_cpu_millicores": 1000,
        "available_memory_mb": 2048,
    },
}


# =========================================================
# 4. WORKLOAD CONFIGURATION
# =========================================================

WORKLOAD = {
    "name": "standard_test_workload",
    "cpu_millicores": 500,
    "memory_mb": 512,
    "max_latency_ms": 100.0,
    "max_cost_per_hour": 0.15,
}


# Each scheduling decision represents one assumed hour.
# This is a modelling assumption, not a measured billing interval.

DECISION_INTERVAL_HOURS = 1.0


# =========================================================
# 5. MULTI-OBJECTIVE SCORING WEIGHTS
# =========================================================

SCORING_WEIGHTS = {
    "carbon": 0.40,
    "cost": 0.25,
    "latency": 0.20,
    "resources": 0.15,
}


# =========================================================
# 6. SCHEDULING STRATEGIES
# =========================================================

STRATEGIES = [
    "random_forest_predictive",
    "persistence_predictive",
    "carbon_only",
    "cost_only",
    "reactive_carbon",
    "round_robin",
]


# =========================================================
# 7. LOAD AND VALIDATE PREDICTIONS
# =========================================================

def load_and_validate_predictions():
    """Load and validate rolling-validation predictions."""

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    missing = REQUIRED_PREDICTION_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    df = df[
        df["model"].isin(FORECAST_MODELS)
    ].copy()

    if df.empty:
        raise ValueError("No supported prediction records found.")

    if set(df["zone"].unique()) != set(ZONES):
        raise ValueError(
            "Expected exactly IE, DE and FR zones."
        )

    df["prediction_time"] = pd.to_datetime(
        df["prediction_time"],
        utc=True,
        errors="raise",
    )

    df["target_time"] = pd.to_datetime(
        df["target_time"],
        utc=True,
        errors="raise",
    )

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="raise",
    )

    df["actual_carbon"] = pd.to_numeric(
        df["actual_carbon"],
        errors="raise",
    )

    numeric_values = df[
        ["prediction", "actual_carbon"]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Non-finite prediction or actual carbon values detected."
        )

    if df["target_is_estimated"].isna().any():
        raise ValueError(
            "Missing target estimation flags detected."
        )

    key_columns = [
        "zone",
        "fold",
        "prediction_time",
        "target_time",
        "model",
    ]

    if df.duplicated(subset=key_columns).any():
        raise ValueError(
            "Duplicate prediction records detected."
        )

    # Confirm identical timestamps across all three zones.
    timestamp_sets = {
        zone: set(
            zip(
                group["fold"],
                group["prediction_time"],
                group["target_time"],
            )
        )
        for zone, group in df.groupby("zone")
    }

    if not (
        timestamp_sets["IE"]
        == timestamp_sets["DE"]
        == timestamp_sets["FR"]
    ):
        raise ValueError(
            "Cross-zone timestamp alignment failed."
        )

    # Require both forecasting models at every decision time.
    counts = df.groupby(
        [
            "zone",
            "fold",
            "prediction_time",
            "model",
        ]
    ).size()

    if not (counts == 1).all():
        raise ValueError(
            "Prediction records are incomplete or duplicated."
        )

    expected_models = set(FORECAST_MODELS)

    model_sets = df.groupby(
        ["zone", "fold", "prediction_time"]
    )["model"].apply(set)

    if not model_sets.apply(
        lambda models: models == expected_models
    ).all():
        raise ValueError(
            "Both forecasting models must be present "
            "for every decision time."
        )

    print("\nPrediction data validated.")
    print(f"Prediction records: {len(df)}")
    print(
        f"Shared decision times: {len(timestamp_sets['IE'])}"
    )

    return df


# =========================================================
# 8. LOAD HISTORICAL CARBON OBSERVATIONS
# =========================================================

def load_historical_observations():
    """Load historical carbon-intensity observations."""

    if not CARBON_FILE.exists():
        raise FileNotFoundError(
            f"Carbon dataset not found: {CARBON_FILE}"
        )

    carbon = pd.read_csv(CARBON_FILE)

    missing = REQUIRED_CARBON_COLUMNS - set(carbon.columns)

    if missing:
        raise ValueError(
            f"Missing carbon columns: {sorted(missing)}"
        )

    carbon = carbon[
        [
            "datetime",
            "zone",
            "carbon_intensity",
            "is_estimated",
        ]
    ].copy()

    carbon["datetime"] = pd.to_datetime(
        carbon["datetime"],
        utc=True,
        errors="raise",
    )

    carbon["carbon_intensity"] = pd.to_numeric(
        carbon["carbon_intensity"],
        errors="raise",
    )

    if not np.isfinite(
        carbon["carbon_intensity"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite carbon-intensity values detected."
        )

    if carbon["is_estimated"].isna().any():
        raise ValueError(
            "Missing historical estimation flags detected."
        )

    if carbon.duplicated(
        subset=["zone", "datetime"]
    ).any():
        raise ValueError(
            "Duplicate historical carbon observations detected."
        )

    print("\nHistorical carbon data validated.")
    print(f"Historical observations: {len(carbon)}")

    print("\nEstimated observations by zone:")

    print(
        carbon.groupby("zone")["is_estimated"]
        .sum()
        .to_string()
    )

    return carbon


# =========================================================
# 9. MATCH CURRENT CARBON OBSERVATIONS
# =========================================================

def attach_current_observations(predictions, carbon):
    """Attach carbon observed at each prediction origin."""

    observations = carbon.rename(
        columns={
            "datetime": "prediction_time",
            "carbon_intensity": "current_carbon",
            "is_estimated": "current_is_estimated",
        }
    )

    merged = predictions.merge(
        observations,
        on=["zone", "prediction_time"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = merged["_merge"] != "both"

    if unmatched.any():
        raise ValueError(
            f"{unmatched.sum()} prediction records "
            "have no matching current observation."
        )

    merged = merged.drop(columns=["_merge"])

    if merged["current_carbon"].isna().any():
        raise ValueError(
            "Missing current carbon observations detected."
        )

    print("\nCurrent observations successfully matched.")
    print(f"Matched prediction records: {len(merged)}")

    print("\nCurrent observation estimated flags:")

    print(
        merged[
            merged["model"] == "persistence"
        ].groupby("zone")["current_is_estimated"]
        .sum()
        .to_string()
    )

    return merged


# =========================================================
# 10. VALIDATE PROVIDER AND WORKLOAD CONFIGURATION
# =========================================================

def validate_experiment_configuration():
    """Check scoring weights and provider eligibility."""

    if abs(sum(SCORING_WEIGHTS.values()) - 1.0) > 1e-9:
        raise ValueError(
            "Scoring weights must sum to 1.0."
        )

    for name, weight in SCORING_WEIGHTS.items():
        if weight < 0:
            raise ValueError(
                f"Scoring weight cannot be negative: {name}"
            )

    eligible_providers = []

    print("\nProvider eligibility:")

    for provider, metrics in PROVIDERS.items():

        eligible = (
            metrics["available_cpu_millicores"]
            >= WORKLOAD["cpu_millicores"]
            and metrics["available_memory_mb"]
            >= WORKLOAD["memory_mb"]
            and metrics["latency_ms"]
            <= WORKLOAD["max_latency_ms"]
            and metrics["cost_per_hour"]
            <= WORKLOAD["max_cost_per_hour"]
        )

        print(
            f"{provider}: "
            f"{'ELIGIBLE' if eligible else 'INELIGIBLE'}"
        )

        if eligible:
            eligible_providers.append(provider)

    if not eligible_providers:
        raise ValueError("No eligible providers found.")

    print("\nConfiguration validated.")
    print("Eligible providers:", eligible_providers)
    print("Scoring weights:", SCORING_WEIGHTS)

    return eligible_providers


# =========================================================
# 11. BUILD DATA LOOKUPS
# =========================================================

def build_experiment_lookups(experiment_data):
    """Build dictionaries for efficient historical data access."""

    forecasts = {}
    current_carbon = {}
    actual_carbon = {}
    estimated_flags = {}

    for row in experiment_data.itertuples():

        forecasts[
            (row.prediction_time, row.zone, row.model)
        ] = float(row.prediction)

        if row.model == "persistence":

            current_carbon[
                (row.prediction_time, row.zone)
            ] = float(row.current_carbon)

            actual_carbon[
                (row.target_time, row.zone)
            ] = float(row.actual_carbon)

            estimated_flags[
                (row.prediction_time, row.zone)
            ] = (
                bool(row.current_is_estimated),
                bool(row.target_is_estimated),
            )

    return (
        forecasts,
        current_carbon,
        actual_carbon,
        estimated_flags,
    )


# =========================================================
# 12. MIN-MAX NORMALISATION
# =========================================================

def min_max_normalise(values, higher_is_better=False):
    """Normalise objective values to a 0-1 range."""

    minimum = min(values.values())
    maximum = max(values.values())

    if maximum == minimum:
        return {
            key: 1.0 for key in values
        }

    result = {}

    for key, value in values.items():

        if higher_is_better:
            score = (
                (value - minimum)
                / (maximum - minimum)
            )
        else:
            score = (
                (maximum - value)
                / (maximum - minimum)
            )

        result[key] = score

    return result


# =========================================================
# 13. WEIGHTED MULTI-OBJECTIVE SCORING
# =========================================================

def calculate_weighted_scores(carbon_values, eligible):
    """Calculate carbon, cost, latency and resource scores."""

    cost_values = {
        provider: PROVIDERS[provider]["cost_per_hour"]
        for provider in eligible
    }

    latency_values = {
        provider: PROVIDERS[provider]["latency_ms"]
        for provider in eligible
    }

    resource_values = {
        provider: min(
            PROVIDERS[provider]["available_cpu_millicores"]
            / WORKLOAD["cpu_millicores"],
            PROVIDERS[provider]["available_memory_mb"]
            / WORKLOAD["memory_mb"],
        )
        for provider in eligible
    }

    carbon_scores = min_max_normalise(carbon_values)
    cost_scores = min_max_normalise(cost_values)
    latency_scores = min_max_normalise(latency_values)

    resource_scores = min_max_normalise(
        resource_values,
        higher_is_better=True,
    )

    scores = {}

    for provider in eligible:

        scores[provider] = (
            SCORING_WEIGHTS["carbon"]
            * carbon_scores[provider]
            + SCORING_WEIGHTS["cost"]
            * cost_scores[provider]
            + SCORING_WEIGHTS["latency"]
            * latency_scores[provider]
            + SCORING_WEIGHTS["resources"]
            * resource_scores[provider]
        )

    return scores


# =========================================================
# 14. PROVIDER SELECTION
# =========================================================

def choose_provider(
    strategy,
    timestamp,
    forecasts,
    current_carbon,
    eligible,
    round_robin_index,
):
    """Apply the selection rule for one scheduling decision."""

    carbon_values = {}

    if strategy in [
        "random_forest_predictive",
        "persistence_predictive",
        "carbon_only",
    ]:

        model = (
            "persistence"
            if strategy == "persistence_predictive"
            else "random_forest_all_features"
        )

        for provider in eligible:

            zone = PROVIDERS[provider]["zone"]
            key = (timestamp, zone, model)

            if key not in forecasts:
                raise ValueError(
                    f"Missing forecast for {provider}: {key}"
                )

            carbon_values[provider] = forecasts[key]

    if strategy in [
        "random_forest_predictive",
        "persistence_predictive",
    ]:

        scores = calculate_weighted_scores(
            carbon_values,
            eligible,
        )

        selected = max(
            eligible,
            key=lambda provider: scores[provider],
        )

        return selected, scores

    if strategy == "carbon_only":

        selected = min(
            eligible,
            key=lambda provider: carbon_values[provider],
        )

        return selected, {}

    if strategy == "cost_only":

        selected = min(
            eligible,
            key=lambda provider: PROVIDERS[provider]["cost_per_hour"],
        )

        return selected, {}

    if strategy == "reactive_carbon":

        selected = min(
            eligible,
            key=lambda provider: current_carbon[
                (timestamp, PROVIDERS[provider]["zone"])
            ],
        )

        return selected, {}

    if strategy == "round_robin":

        selected = eligible[
            round_robin_index % len(eligible)
        ]

        return selected, {}

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# =========================================================
# 15. RUN SCHEDULING EXPERIMENTS
# =========================================================

def run_scheduling_experiments(
    experiment_data,
    eligible_providers,
):
    """Simulate all strategies using identical decision times."""

    (
        forecasts,
        current_carbon,
        actual_carbon,
        estimated_flags,
    ) = build_experiment_lookups(experiment_data)

    base_records = (
        experiment_data[
            experiment_data["model"] == "persistence"
        ][
            [
                "fold",
                "prediction_time",
                "target_time",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["prediction_time", "fold"]
        )
        .reset_index(drop=True)
    )

    if base_records.empty:
        raise ValueError("No scheduling decision times found.")

    if base_records.duplicated(
        subset=["prediction_time"]
    ).any():
        raise ValueError(
            "A prediction timestamp maps to multiple decision records."
        )

    results = []

    for strategy in STRATEGIES:

        print(f"\nRunning strategy: {strategy}")

        round_robin_index = 0

        for decision in base_records.itertuples(index=False):

            timestamp = decision.prediction_time
            target_time = decision.target_time

            selected, scores = choose_provider(
                strategy=strategy,
                timestamp=timestamp,
                forecasts=forecasts,
                current_carbon=current_carbon,
                eligible=eligible_providers,
                round_robin_index=round_robin_index,
            )

            if strategy == "round_robin":
                round_robin_index += 1

            selected_zone = PROVIDERS[selected]["zone"]

            # The predictive strategies use their named model.
            # Carbon-only uses RF forecasts.
            # Other strategies retain RF forecasts for diagnostic
            # comparison only; they do not use them for selection.

            if strategy == "persistence_predictive":
                forecast_model = "persistence"
                forecast_usage = "decision_input"

            elif strategy in [
                "random_forest_predictive",
                "carbon_only",
            ]:
                forecast_model = "random_forest_all_features"
                forecast_usage = "decision_input"

            else:
                forecast_model = "random_forest_all_features"
                forecast_usage = "diagnostic_only"

            forecast_key = (
                timestamp,
                selected_zone,
                forecast_model,
            )

            current_key = (
                timestamp,
                selected_zone,
            )

            outcome_key = (
                target_time,
                selected_zone,
            )

            if forecast_key not in forecasts:
                raise ValueError(
                    f"Missing forecast: {forecast_key}"
                )

            if current_key not in current_carbon:
                raise ValueError(
                    f"Missing current carbon: {current_key}"
                )

            if outcome_key not in actual_carbon:
                raise ValueError(
                    f"Missing actual outcome: {outcome_key}"
                )

            current_estimated, target_estimated = (
                estimated_flags[current_key]
            )

            provider_metrics = PROVIDERS[selected]

            results.append({
                "strategy": strategy,
                "fold": decision.fold,
                "prediction_time": timestamp,
                "target_time": target_time,
                "selected_provider": selected,
                "selected_zone": selected_zone,
                "forecast_model": forecast_model,
                "forecast_usage": forecast_usage,
                "forecast_carbon": forecasts[forecast_key],
                "current_carbon": current_carbon[current_key],
                "actual_target_carbon": actual_carbon[outcome_key],
                "current_is_estimated": current_estimated,
                "target_is_estimated": target_estimated,
                "cost_per_hour": provider_metrics["cost_per_hour"],
                "decision_interval_hours": DECISION_INTERVAL_HOURS,
                "assumed_decision_cost_eur": (
                    provider_metrics["cost_per_hour"]
                    * DECISION_INTERVAL_HOURS
                ),
                "latency_ms": provider_metrics["latency_ms"],
                "weighted_score": scores.get(selected),
            })

    result_df = pd.DataFrame(results)

    expected_records = (
        len(STRATEGIES) * len(base_records)
    )

    if len(result_df) != expected_records:
        raise ValueError(
            f"Expected {expected_records} decisions, "
            f"generated {len(result_df)}."
        )

    # Validate all fields that must be present for every strategy.
    # weighted_score is intentionally excluded here because
    # non-weighted strategies do not calculate it.

    required_result_columns = [
        "strategy",
        "fold",
        "prediction_time",
        "target_time",
        "selected_provider",
        "selected_zone",
        "forecast_model",
        "forecast_usage",
        "forecast_carbon",
        "current_carbon",
        "actual_target_carbon",
        "current_is_estimated",
        "target_is_estimated",
        "cost_per_hour",
        "decision_interval_hours",
        "assumed_decision_cost_eur",
        "latency_ms",
    ]

    missing_values = result_df[
        required_result_columns
    ].isna().sum()

    if missing_values.any():
        raise ValueError(
            "Missing required scheduling values:\n"
            f"{missing_values[missing_values > 0]}"
        )

    # Validate weighted scores only for strategies that calculate them.

    weighted_strategies = [
        "random_forest_predictive",
        "persistence_predictive",
    ]

    weighted_rows = result_df[
        result_df["strategy"].isin(weighted_strategies)
    ]

    if weighted_rows["weighted_score"].isna().any():
        raise ValueError(
            "Weighted scores missing for predictive strategies."
        )

    # Confirm non-weighted strategies have no accidental scores.

    non_weighted_rows = result_df[
        ~result_df["strategy"].isin(weighted_strategies)
    ]

    if non_weighted_rows["weighted_score"].notna().any():
        raise ValueError(
            "Unexpected weighted scores found for non-weighted strategies."
        )

    # Confirm all strategies have the same number of decisions.

    strategy_counts = result_df.groupby("strategy").size()

    if not (strategy_counts == len(base_records)).all():
        raise ValueError(
            "Strategies do not have equal decision counts."
        )

    print("\nScheduling result validation passed.")
    print("\nScheduling simulation completed.")
    print(f"Strategies: {len(STRATEGIES)}")
    print(f"Decision times per strategy: {len(base_records)}")
    print(f"Total decisions: {len(result_df)}")

    return result_df


# =========================================================
# 16. DISPLAY PROVIDER SELECTION RESULTS
# =========================================================

def display_experiment_results(result_df):
    """Display provider selection counts and sample decisions."""

    print("\n" + "=" * 65)
    print("PROVIDER SELECTION COUNTS")
    print("=" * 65)

    selection_counts = (
        result_df.groupby(
            ["strategy", "selected_provider"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print(selection_counts.to_string())

    print("\nSAMPLE SCHEDULING DECISIONS")

    preview_columns = [
        "strategy",
        "fold",
        "prediction_time",
        "selected_provider",
        "selected_zone",
        "forecast_model",
        "forecast_usage",
        "forecast_carbon",
        "current_carbon",
        "actual_target_carbon",
        "cost_per_hour",
        "latency_ms",
    ]

    print(
        result_df[preview_columns]
        .head(18)
        .to_string(index=False)
    )


# =========================================================
# 17. EVALUATE STRATEGY PERFORMANCE
# =========================================================

def evaluate_experiment_results(result_df):
    """Calculate descriptive metrics for each strategy."""

    summaries = []

    for strategy, group in result_df.groupby("strategy"):

        summaries.append({
            "strategy": strategy,

            "decision_count": len(group),

            "mean_actual_carbon_intensity":
                group["actual_target_carbon"].mean(),

            "median_actual_carbon_intensity":
                group["actual_target_carbon"].median(),

            "mean_forecast_carbon":
                group["forecast_carbon"].mean(),

            "mean_assumed_cost_per_hour":
                group["cost_per_hour"].mean(),

            "total_assumed_cost_eur":
                group["assumed_decision_cost_eur"].sum(),

            "mean_assumed_latency_ms":
                group["latency_ms"].mean(),

            "estimated_target_percentage":
                group["target_is_estimated"].mean() * 100,

            "estimated_current_percentage":
                group["current_is_estimated"].mean() * 100,
        })

    summary_df = pd.DataFrame(summaries)

    summary_df = summary_df.sort_values(
        "mean_actual_carbon_intensity"
    ).reset_index(drop=True)

    return summary_df


# =========================================================
# 18. EXPORT EXPERIMENT RESULTS
# =========================================================

def export_experiment_results(result_df, summary_df):
    """Export decision records, summary and selection counts."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    decisions_path = (
        OUTPUT_DIR / "scheduler_decisions.csv"
    )

    summary_path = (
        OUTPUT_DIR / "scheduler_strategy_summary.csv"
    )

    selections_path = (
        OUTPUT_DIR / "scheduler_provider_selections.csv"
    )

    result_df.to_csv(
        decisions_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    selection_summary = (
        result_df.groupby(
            ["strategy", "selected_provider"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    selection_summary.to_csv(
        selections_path,
        index=False,
    )

    print("\nExperiment files exported:")

    print(f"Decision records: {decisions_path}")
    print(f"Strategy summary: {summary_path}")
    print(f"Provider selections: {selections_path}")


# =========================================================
# 19. MAIN EXPERIMENT EXECUTION
# =========================================================

def main():

    print("=" * 65)
    print("CARBON-AWARE MULTI-CLOUD SCHEDULING EXPERIMENT")
    print("=" * 65)

    predictions = load_and_validate_predictions()

    carbon = load_historical_observations()

    experiment_data = attach_current_observations(
        predictions,
        carbon,
    )

    eligible_providers = validate_experiment_configuration()

    result_df = run_scheduling_experiments(
        experiment_data,
        eligible_providers,
    )

    display_experiment_results(result_df)

    summary_df = evaluate_experiment_results(
        result_df
    )

    print("\n" + "=" * 65)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("=" * 65)

    print(
        summary_df.round(3).to_string(index=False)
    )

    export_experiment_results(
        result_df,
        summary_df,
    )

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
