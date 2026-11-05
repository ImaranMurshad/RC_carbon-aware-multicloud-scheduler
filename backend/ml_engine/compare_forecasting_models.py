from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "historical"
    / "carbon_history.csv"
)

RESULT_DIR = PROJECT_ROOT / "results" / "ml"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ZONES = ["IE", "DE", "FR"]

TRAIN_RATIO = 0.80
RANDOM_STATE = 42

ALL_FEATURES = [
    "current_carbon",
    "carbon_1h_ago",
    "carbon_2h_ago",
    "carbon_3h_ago",
    "hour",
    "day_of_week",
]

FEATURE_GROUPS = {
    "random_forest_all_features": ALL_FEATURES,
    "random_forest_current_only": [
        "current_carbon",
    ],
    "random_forest_lags_only": [
        "carbon_1h_ago",
        "carbon_2h_ago",
        "carbon_3h_ago",
    ],
    "gradient_boosting_all_features": ALL_FEATURES,
    "xgboost_all_features": ALL_FEATURES,
}


# ==================================================
# DATA LOADING
# ==================================================

def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    required = [
        "datetime",
        "zone",
        "carbon_intensity",
        "is_estimated",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    df["carbon_intensity"] = pd.to_numeric(
        df["carbon_intensity"],
        errors="coerce",
    )

    if df["is_estimated"].dtype != bool:
        df["is_estimated"] = (
            df["is_estimated"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
        )

    df = df.dropna(
        subset=required
    ).copy()

    if df.duplicated(
        subset=["zone", "datetime"]
    ).any():
        raise ValueError("Duplicate zone/timestamp records found.")

    return df.sort_values(
        ["zone", "datetime"]
    ).reset_index(drop=True)


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def prepare_zone_data(df, zone):

    data = (
        df[df["zone"] == zone]
        .sort_values("datetime")
        .copy()
        .reset_index(drop=True)
    )

    data["target"] = (
        data["carbon_intensity"].shift(-1)
    )

    data["target_is_estimated"] = (
        data["is_estimated"].shift(-1)
    )

    data["current_carbon"] = (
        data["carbon_intensity"]
    )

    data["carbon_1h_ago"] = (
        data["carbon_intensity"].shift(1)
    )

    data["carbon_2h_ago"] = (
        data["carbon_intensity"].shift(2)
    )

    data["carbon_3h_ago"] = (
        data["carbon_intensity"].shift(3)
    )

    data["hour"] = data["datetime"].dt.hour

    data["day_of_week"] = (
        data["datetime"].dt.day_of_week
    )

    # Keep only rows whose three lags and the target are exactly 1 hour apart,
    # so gaps in the collected data cannot create wrong features or targets.
    step = pd.Timedelta(hours=1)
    dt = data["datetime"]
    contiguous = (
        (dt.shift(-1) - dt == step)
        & (dt - dt.shift(1) == step)
        & (dt.shift(1) - dt.shift(2) == step)
        & (dt.shift(2) - dt.shift(3) == step)
    )
    data = data[contiguous].copy()

    data = data.dropna(
        subset=ALL_FEATURES + [
            "target",
            "target_is_estimated",
        ]
    ).reset_index(drop=True)

    return data


# ==================================================
# MODEL DEFINITIONS
# ==================================================

def build_model(model_name):

    if model_name.startswith("random_forest"):

        return RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "gradient_boosting_all_features":

        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_split=4,
            min_samples_leaf=2,
            loss="huber",
            random_state=RANDOM_STATE,
        )

    if model_name == "xgboost_all_features":

        return XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


# ==================================================
# EVALUATION
# ==================================================

def calculate_metrics(actual, predicted):

    return {
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "bias": float(
            np.mean(actual - predicted)
        ),
    }


def evaluate_predictions(test_df, model_name, predictions):

    actual = test_df["target"].to_numpy()

    estimated_mask = (
        test_df["target_is_estimated"]
        .astype(bool)
        .to_numpy()
    )

    actual_mask = ~estimated_mask

    overall = calculate_metrics(
        actual,
        predictions,
    )

    if actual_mask.any():
        actual_only = calculate_metrics(
            actual[actual_mask],
            predictions[actual_mask],
        )
    else:
        actual_only = {
            "mae": None,
            "rmse": None,
            "bias": None,
        }

    if estimated_mask.any():
        estimated_only = calculate_metrics(
            actual[estimated_mask],
            predictions[estimated_mask],
        )
    else:
        estimated_only = {
            "mae": None,
            "rmse": None,
            "bias": None,
        }

    return {
        "model": model_name,
        "overall_mae": overall["mae"],
        "overall_rmse": overall["rmse"],
        "overall_bias": overall["bias"],
        "actual_mae": actual_only["mae"],
        "actual_rmse": actual_only["rmse"],
        "estimated_mae": estimated_only["mae"],
        "estimated_rmse": estimated_only["rmse"],
        "test_records": len(test_df),
        "actual_records": int(actual_mask.sum()),
        "estimated_records": int(estimated_mask.sum()),
    }


# ==================================================
# EXPERIMENT
# ==================================================

def run_experiment(df, zone):

    data = prepare_zone_data(df, zone)

    split_index = int(
        len(data) * TRAIN_RATIO
    )

    train_df = data.iloc[:split_index].copy()
    test_df = data.iloc[split_index:].copy()

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["target"]

    print(f"\n========== {zone} ==========")
    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))

    results = []
    prediction_rows = []

    # Persistence baseline
    baseline_predictions = (
        test_df["current_carbon"].to_numpy()
    )

    baseline_metrics = evaluate_predictions(
        test_df,
        "persistence",
        baseline_predictions,
    )

    baseline_metrics["zone"] = zone
    results.append(baseline_metrics)

    prediction_rows.append(
        pd.DataFrame({
            "zone": zone,
            "model": "persistence",
            "target_time": (
                test_df["datetime"]
                + pd.Timedelta(hours=1)
            ),
            "actual_carbon": test_df["target"].to_numpy(),
            "prediction": baseline_predictions,
            "target_is_estimated": (
                test_df["target_is_estimated"].to_numpy()
            ),
        })
    )

    # Train and evaluate each ML model
    for model_name, features in FEATURE_GROUPS.items():

        model = build_model(model_name)

        model.fit(
            train_df[features],
            y_train,
        )

        predictions = model.predict(
            test_df[features]
        )

        metrics = evaluate_predictions(
            test_df,
            model_name,
            predictions,
        )

        metrics["zone"] = zone
        results.append(metrics)

        prediction_rows.append(
            pd.DataFrame({
                "zone": zone,
                "model": model_name,
                "target_time": (
                    test_df["datetime"]
                    + pd.Timedelta(hours=1)
                ),
                "actual_carbon": test_df["target"].to_numpy(),
                "prediction": predictions,
                "target_is_estimated": (
                    test_df["target_is_estimated"].to_numpy()
                ),
            })
        )

        print(
            f"{model_name}: "
            f"MAE={metrics['overall_mae']:.3f}, "
            f"RMSE={metrics['overall_rmse']:.3f}, "
            f"Bias={metrics['overall_bias']:.3f}"
        )

    return results, prediction_rows


# ==================================================
# MAIN
# ==================================================

def main():

    print("==========================================")
    print("MULTI-MODEL CARBON FORECAST COMPARISON")
    print("==========================================")

    df = load_dataset()

    all_results = []
    all_predictions = []

    for zone in ZONES:

        if zone not in df["zone"].unique():
            continue

        results, predictions = run_experiment(
            df,
            zone,
        )

        all_results.extend(results)
        all_predictions.extend(predictions)

    metrics_df = pd.DataFrame(all_results)

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics_path = (
        RESULT_DIR
        / "model_comparison_metrics.csv"
    )

    predictions_path = (
        RESULT_DIR
        / "model_comparison_predictions.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print("\n==========================================")
    print("FINAL COMPARISON")
    print("==========================================")

    print(
        metrics_df[
            [
                "zone",
                "model",
                "overall_mae",
                "overall_rmse",
                "overall_bias",
                "actual_mae",
                "estimated_mae",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nMetrics saved:", metrics_path)
    print("Predictions saved:", predictions_path)

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()