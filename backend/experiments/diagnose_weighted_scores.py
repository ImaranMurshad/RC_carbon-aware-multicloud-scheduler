
"""
Diagnose weighted provider scores for predictive scheduling.

Reads existing experiment inputs and reconstructs provider scores.
Does not modify the database or existing experiment outputs.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    ROOT / "results/ml/rolling_validation_predictions.csv"
)

CARBON_FILE = (
    ROOT / "datasets/historical/carbon_history.csv"
)

DECISIONS_FILE = (
    ROOT / "results/experiments/scheduler_decisions.csv"
)

PROVIDERS = {
    "AWS": {
        "zone": "IE",
        "cost": 0.08,
        "latency": 20.0,
        "cpu": 2000,
        "memory": 4096,
    },
    "AZURE": {
        "zone": "DE",
        "cost": 0.09,
        "latency": 35.0,
        "cpu": 1500,
        "memory": 3072,
    },
    "GCP": {
        "zone": "FR",
        "cost": 0.07,
        "latency": 45.0,
        "cpu": 1000,
        "memory": 2048,
    },
}

WEIGHTS = {
    "carbon": 0.40,
    "cost": 0.25,
    "latency": 0.20,
    "resources": 0.15,
}

CPU_REQUEST = 500
MEMORY_REQUEST = 512


def normalise(values, higher_is_better=False):

    minimum = min(values.values())
    maximum = max(values.values())

    if maximum == minimum:
        return {key: 1.0 for key in values}

    result = {}

    for key, value in values.items():

        if higher_is_better:
            result[key] = (
                (value - minimum) / (maximum - minimum)
            )
        else:
            result[key] = (
                (maximum - value) / (maximum - minimum)
            )

    return result


def calculate_scores(carbon_values):

    costs = {
        p: PROVIDERS[p]["cost"]
        for p in PROVIDERS
    }

    latencies = {
        p: PROVIDERS[p]["latency"]
        for p in PROVIDERS
    }

    resources = {
        p: min(
            PROVIDERS[p]["cpu"] / CPU_REQUEST,
            PROVIDERS[p]["memory"] / MEMORY_REQUEST,
        )
        for p in PROVIDERS
    }

    carbon_scores = normalise(carbon_values)
    cost_scores = normalise(costs)
    latency_scores = normalise(latencies)
    resource_scores = normalise(
        resources,
        higher_is_better=True,
    )

    scores = {}

    for provider in PROVIDERS:

        scores[provider] = (
            WEIGHTS["carbon"] * carbon_scores[provider]
            + WEIGHTS["cost"] * cost_scores[provider]
            + WEIGHTS["latency"] * latency_scores[provider]
            + WEIGHTS["resources"] * resource_scores[provider]
        )

    return scores, carbon_scores, cost_scores, latency_scores, resource_scores


def main():

    predictions = pd.read_csv(PREDICTIONS_FILE)
    decisions = pd.read_csv(DECISIONS_FILE)

    predictions = predictions[
        predictions["model"] == "random_forest_all_features"
    ].copy()

    predictions["prediction_time"] = pd.to_datetime(
        predictions["prediction_time"], utc=True
    )

    decisions["prediction_time"] = pd.to_datetime(
        decisions["prediction_time"], utc=True
    )

    predictions = predictions[
        predictions["zone"].isin(["IE", "DE", "FR"])
    ]

    lookup = predictions.pivot(
        index="prediction_time",
        columns="zone",
        values="prediction",
    )

    rows = []

    for decision in decisions[
        decisions["strategy"] == "random_forest_predictive"
    ].itertuples():

        timestamp = decision.prediction_time

        carbon_values = {
            provider: float(
                lookup.loc[
                    timestamp,
                    PROVIDERS[provider]["zone"],
                ]
            )
            for provider in PROVIDERS
        }

        (
            scores,
            carbon_scores,
            cost_scores,
            latency_scores,
            resource_scores,
        ) = calculate_scores(carbon_values)

        calculated_choice = max(
            scores,
            key=scores.get,
        )

        rows.append({
            "prediction_time": timestamp,
            "recorded_choice": decision.selected_provider,
            "calculated_choice": calculated_choice,

            "aws_forecast": carbon_values["AWS"],
            "azure_forecast": carbon_values["AZURE"],
            "gcp_forecast": carbon_values["GCP"],

            "aws_score": scores["AWS"],
            "azure_score": scores["AZURE"],
            "gcp_score": scores["GCP"],

            "aws_carbon_score": carbon_scores["AWS"],
            "gcp_carbon_score": carbon_scores["GCP"],

            "aws_cost_score": cost_scores["AWS"],
            "gcp_cost_score": cost_scores["GCP"],

            "aws_latency_score": latency_scores["AWS"],
            "gcp_latency_score": latency_scores["GCP"],

            "aws_resource_score": resource_scores["AWS"],
            "gcp_resource_score": resource_scores["GCP"],
        })

    result = pd.DataFrame(rows)

    mismatches = result[
        result["recorded_choice"]
        != result["calculated_choice"]
    ]

    print("\nWEIGHTED SCORE DIAGNOSTIC")
    print("=" * 70)

    print("Decisions checked:", len(result))
    print("Recorded/calculated mismatches:", len(mismatches))

    print("\nCalculated provider selections:")
    print(result["calculated_choice"].value_counts().to_string())

    print("\nFirst 15 scheduling decisions:")
    print(
        result.head(15).round(4).to_string(index=False)
    )

    print("\nAWS selection examples:")
    aws_examples = result[
        result["recorded_choice"] == "AWS"
    ]

    print(
        aws_examples.head(10).round(4).to_string(index=False)
    )

    output = ROOT / "results/experiments/weighted_score_diagnostics.csv"

    result.to_csv(output, index=False)

    print("\nDiagnostic file saved:")
    print(output)


if __name__ == "__main__":
    main()
