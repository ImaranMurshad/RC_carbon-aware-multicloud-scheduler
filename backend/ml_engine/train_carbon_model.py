from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "historical"
    / "carbon_history.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "carbon_forecasting"
RESULT_DIR = PROJECT_ROOT / "results" / "ml"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# CONFIGURATION
# ==================================================

ZONES = ["IE", "DE", "FR"]

TRAIN_RATIO = 0.80
RANDOM_STATE = 42

FEATURES = [
    "current_carbon",
    "carbon_1h_ago",
    "carbon_2h_ago",
    "carbon_3h_ago",
    "hour",
    "day_of_week",
]


# ==================================================
# DATA LOADING
# ==================================================

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "datetime",
        "zone",
        "carbon_intensity",
        "is_estimated",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    df["carbon_intensity"] = pd.to_numeric(
        df["carbon_intensity"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "datetime",
            "zone",
            "carbon_intensity",
            "is_estimated",
        ]
    ).copy()

    # Normalize estimation flags to boolean.
    if df["is_estimated"].dtype != bool:
        df["is_estimated"] = (
            df["is_estimated"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
        )

    if df["is_estimated"].isna().any():
        raise ValueError(
            "Invalid values found in is_estimated column."
        )

    # Ensure one reading per zone and timestamp.
    duplicates = df.duplicated(
        subset=["zone", "datetime"]
    )

    if duplicates.any():
        raise ValueError(
            f"Found {duplicates.sum()} duplicate zone/timestamp records."
        )

    df = df.sort_values(
        ["zone", "datetime"]
    ).reset_index(drop=True)

    return df


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def prepare_zone_data(df, zone):

    zone_df = (
        df[df["zone"] == zone]
        .sort_values("datetime")
        .copy()
        .reset_index(drop=True)
    )

    if zone_df.empty:
        return zone_df

    # Next-hour prediction target.
    zone_df["target"] = (
        zone_df["carbon_intensity"].shift(-1)
    )

    # IMPORTANT:
    # This flag belongs to the future target,
    # not the current input observation.
    zone_df["target_is_estimated"] = (
        zone_df["is_estimated"].shift(-1)
    )

    # Historical carbon features.
    zone_df["current_carbon"] = (
        zone_df["carbon_intensity"]
    )

    zone_df["carbon_1h_ago"] = (
        zone_df["carbon_intensity"].shift(1)
    )

    zone_df["carbon_2h_ago"] = (
        zone_df["carbon_intensity"].shift(2)
    )

    zone_df["carbon_3h_ago"] = (
        zone_df["carbon_intensity"].shift(3)
    )

    # Time-based features at prediction time.
    zone_df["hour"] = (
        zone_df["datetime"].dt.hour
    )

    zone_df["day_of_week"] = (
        zone_df["datetime"].dt.day_of_week
    )

    # Remove rows without sufficient history or target.
    # Keep only rows whose three lags and the target are exactly 1 hour apart,
    # so gaps in the collected data cannot create wrong features or targets.
    step = pd.Timedelta(hours=1)
    dt = zone_df["datetime"]
    contiguous = (
        (dt.shift(-1) - dt == step)
        & (dt - dt.shift(1) == step)
        & (dt.shift(1) - dt.shift(2) == step)
        & (dt.shift(2) - dt.shift(3) == step)
    )
    zone_df = zone_df[contiguous].copy()

    zone_df = zone_df.dropna(
        subset=FEATURES + [
            "target",
            "target_is_estimated",
        ]
    ).reset_index(drop=True)

    return zone_df


# ==================================================
# EVALUATION METRICS
# ==================================================

def calculate_metrics(actual, predicted):

    if len(actual) == 0:
        return {
            "mae": None,
            "rmse": None,
            "records": 0,
        }

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "records": int(len(actual)),
    }


def evaluate_subset(test_df, model_predictions, baseline_predictions, mask):

    subset = test_df.loc[mask]

    if subset.empty:
        return {
            "model_mae": None,
            "model_rmse": None,
            "baseline_mae": None,
            "baseline_rmse": None,
            "records": 0,
        }

    positions = np.flatnonzero(mask.to_numpy())

    actual = subset["target"].to_numpy()

    model_subset_predictions = model_predictions[positions]

    baseline_subset_predictions = baseline_predictions[positions]

    model_metrics = calculate_metrics(
        actual,
        model_subset_predictions,
    )

    baseline_metrics = calculate_metrics(
        actual,
        baseline_subset_predictions,
    )

    return {
        "model_mae": model_metrics["mae"],
        "model_rmse": model_metrics["rmse"],
        "baseline_mae": baseline_metrics["mae"],
        "baseline_rmse": baseline_metrics["rmse"],
        "records": len(subset),
    }


# ==================================================
# MODEL TRAINING AND EVALUATION
# ==================================================

def train_zone_model(df, zone):

    zone_df = prepare_zone_data(df, zone)

    if len(zone_df) < 10:
        raise ValueError(
            f"Insufficient usable records for zone {zone}: "
            f"{len(zone_df)}"
        )

    # Chronological 80/20 split.
    split_index = int(
        len(zone_df) * TRAIN_RATIO
    )

    train_df = zone_df.iloc[:split_index].copy()

    test_df = zone_df.iloc[split_index:].copy()

    X_train = train_df[FEATURES]

    y_train = train_df["target"]

    X_test = test_df[FEATURES]

    y_test = test_df["target"]

    # Random Forest model.
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    model_predictions = model.predict(
        X_test
    )

    # Persistence baseline:
    # Next-hour intensity = current-hour intensity.
    baseline_predictions = (
        test_df["current_carbon"].to_numpy()
    )

    actual_values = y_test.to_numpy()

    # Overall metrics.
    model_metrics = calculate_metrics(
        actual_values,
        model_predictions,
    )

    baseline_metrics = calculate_metrics(
        actual_values,
        baseline_predictions,
    )

    # Target classification.
    actual_mask = (
        ~test_df["target_is_estimated"].astype(bool)
    )

    estimated_mask = (
        test_df["target_is_estimated"].astype(bool)
    )

    # Separate evaluation.
    actual_metrics = evaluate_subset(
        test_df,
        model_predictions,
        baseline_predictions,
        actual_mask,
    )

    estimated_metrics = evaluate_subset(
        test_df,
        model_predictions,
        baseline_predictions,
        estimated_mask,
    )

    # Save trained model.
    model_path = (
        MODEL_DIR / f"random_forest_{zone}.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "zone": zone,
            "forecast_horizon_hours": 1,
            "training_records": len(train_df),
            "random_state": RANDOM_STATE,
        },
        model_path,
    )

    # Save individual predictions.
    results = []

    for index, (_, row) in enumerate(test_df.iterrows()):

        results.append({
            "zone": zone,
            "prediction_time": row["datetime"],
            "target_time": (
                row["datetime"] + pd.Timedelta(hours=1)
            ),
            "actual_carbon": row["target"],
            "model_prediction": model_predictions[index],
            "baseline_prediction": baseline_predictions[index],
            "target_is_estimated": bool(
                row["target_is_estimated"]
            ),
        })

    predictions_df = pd.DataFrame(results)

    # Print results.
    print(f"\n========== {zone} ==========")

    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))

    print("\nOverall Random Forest:")
    print(model_metrics)

    print("\nOverall Persistence Baseline:")
    print(baseline_metrics)

    print("\nActual-only evaluation:")
    print(actual_metrics)

    print("\nEstimated-only evaluation:")
    print(estimated_metrics)

    print("\nSaved model:", model_path)

    metrics_row = {
        "zone": zone,
        "train_records": len(train_df),
        "test_records": len(test_df),

        "model_mae": model_metrics["mae"],
        "model_rmse": model_metrics["rmse"],

        "baseline_mae": baseline_metrics["mae"],
        "baseline_rmse": baseline_metrics["rmse"],

        "actual_test_records": actual_metrics["records"],
        "actual_model_mae": actual_metrics["model_mae"],
        "actual_model_rmse": actual_metrics["model_rmse"],
        "actual_baseline_mae": actual_metrics["baseline_mae"],
        "actual_baseline_rmse": actual_metrics["baseline_rmse"],

        "estimated_test_records": estimated_metrics["records"],
        "estimated_model_mae": estimated_metrics["model_mae"],
        "estimated_model_rmse": estimated_metrics["model_rmse"],
        "estimated_baseline_mae": estimated_metrics["baseline_mae"],
        "estimated_baseline_rmse": estimated_metrics["baseline_rmse"],
    }

    return metrics_row, predictions_df


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():

    print("==========================================")
    print("CARBON INTENSITY FORECASTING")
    print("==========================================")

    print("\nLoading historical carbon dataset...")

    df = load_dataset()

    print("Total records:", len(df))

    print("\nDataset records per zone:")
    print(df.groupby("zone").size())

    all_metrics = []

    all_predictions = []

    for zone in ZONES:

        if zone not in df["zone"].unique():
            print(f"\nSkipping {zone}: no dataset records")
            continue

        metrics, predictions = train_zone_model(
            df,
            zone,
        )

        all_metrics.append(metrics)

        all_predictions.append(predictions)

    if not all_metrics:
        raise RuntimeError(
            "No zone models were successfully trained."
        )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # Save evaluation results.
    metrics_path = (
        RESULT_DIR / "prediction_metrics.csv"
    )

    predictions_path = (
        RESULT_DIR / "carbon_predictions.csv"
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
    print("FINAL MODEL COMPARISON")
    print("==========================================")

    print(
        metrics_df[
            [
                "zone",
                "model_mae",
                "baseline_mae",
                "model_rmse",
                "baseline_rmse",
                "actual_test_records",
                "estimated_test_records",
            ]
        ].to_string(index=False)
    )

    print("\nMetrics saved:", metrics_path)

    print("Predictions saved:", predictions_path)

    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    main()