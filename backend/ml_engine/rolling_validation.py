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

RANDOM_STATE = 42

FEATURES = [
    "current_carbon",
    "carbon_1h_ago",
    "carbon_2h_ago",
    "carbon_3h_ago",
    "hour",
    "day_of_week",
]

FEATURE_GROUPS = {
    "random_forest_all_features": FEATURES,
    "random_forest_current_only": [
        "current_carbon",
    ],
    "random_forest_lags_only": [
        "carbon_1h_ago",
        "carbon_2h_ago",
        "carbon_3h_ago",
    ],
    "gradient_boosting_all_features": FEATURES,
    "xgboost_all_features": FEATURES,
}

# Each fold trains on the earlier observations
# and validates on the immediately following 10%.
FOLDS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
]


# ==================================================
# DATA LOADING
# ==================================================

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

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
        raise ValueError(
            f"Missing required columns: {missing}"
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

    if df["is_estimated"].dtype != bool:
        df["is_estimated"] = (
            df["is_estimated"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
            })
        )

    df = df.dropna(
        subset=required
    ).copy()

    if df["is_estimated"].isna().any():
        raise ValueError(
            "Invalid estimation flags found."
        )

    if df.duplicated(
        subset=["zone", "datetime"]
    ).any():
        raise ValueError(
            "Duplicate zone/timestamp records found."
        )

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
        subset=FEATURES + [
            "target",
            "target_is_estimated",
        ]
    ).reset_index(drop=True)

    return data


# ==================================================
# MODEL FACTORY
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

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(actual, predicted):

    return {
        "mae": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "bias": float(
            np.mean(actual - predicted)
        ),
    }


def evaluate_model(
    zone,
    fold_number,
    model_name,
    validation_df,
    predictions,
):

    actual = validation_df["target"].to_numpy()

    estimated_mask = (
        validation_df["target_is_estimated"]
        .astype(bool)
        .to_numpy()
    )

    actual_mask = ~estimated_mask

    overall = calculate_metrics(
        actual,
        predictions,
    )

    actual_metrics = (
        calculate_metrics(
            actual[actual_mask],
            predictions[actual_mask],
        )
        if actual_mask.any()
        else {
            "mae": None,
            "rmse": None,
            "bias": None,
        }
    )

    estimated_metrics = (
        calculate_metrics(
            actual[estimated_mask],
            predictions[estimated_mask],
        )
        if estimated_mask.any()
        else {
            "mae": None,
            "rmse": None,
            "bias": None,
        }
    )

    return {
        "zone": zone,
        "fold": fold_number,
        "model": model_name,
        "train_records": None,
        "validation_records": len(validation_df),
        "overall_mae": overall["mae"],
        "overall_rmse": overall["rmse"],
        "overall_bias": overall["bias"],
        "actual_records": int(actual_mask.sum()),
        "actual_mae": actual_metrics["mae"],
        "actual_rmse": actual_metrics["rmse"],
        "estimated_records": int(estimated_mask.sum()),
        "estimated_mae": estimated_metrics["mae"],
        "estimated_rmse": estimated_metrics["rmse"],
    }


# ==================================================
# ROLLING-ORIGIN VALIDATION
# ==================================================

def run_zone_validation(df, zone):

    data = prepare_zone_data(df, zone)

    if len(data) < 20:
        raise ValueError(
            f"Insufficient records for {zone}"
        )

    total_records = len(data)

    print(f"\n========== {zone} ==========")
    print("Usable records:", total_records)

    all_metrics = []
    all_predictions = []

    for fold_number, (
        train_ratio,
        validation_end_ratio,
    ) in enumerate(FOLDS, start=1):

        train_end = int(
            total_records * train_ratio
        )

        validation_end = int(
            total_records * validation_end_ratio
        )

        train_df = data.iloc[:train_end].copy()

        validation_df = data.iloc[
            train_end:validation_end
        ].copy()

        if train_df.empty or validation_df.empty:
            raise ValueError(
                f"Empty fold {fold_number} for {zone}"
            )

        print(f"\n----- FOLD {fold_number} -----")
        print("Training records:", len(train_df))
        print("Validation records:", len(validation_df))
        print(
            "Training period:",
            train_df["datetime"].min(),
            "to",
            train_df["datetime"].max(),
        )
        print(
            "Validation period:",
            validation_df["datetime"].min(),
            "to",
            validation_df["datetime"].max(),
        )

        X_train = train_df[FEATURES]
        y_train = train_df["target"]

        # Persistence baseline
        baseline_predictions = (
            validation_df["current_carbon"].to_numpy()
        )

        baseline_metrics = evaluate_model(
            zone,
            fold_number,
            "persistence",
            validation_df,
            baseline_predictions,
        )

        baseline_metrics["train_records"] = len(train_df)

        all_metrics.append(baseline_metrics)

        all_predictions.append(
            create_prediction_records(
                zone,
                fold_number,
                "persistence",
                validation_df,
                baseline_predictions,
            )
        )

        # Machine-learning models
        for model_name, features in FEATURE_GROUPS.items():

            model = build_model(model_name)

            model.fit(
                train_df[features],
                y_train,
            )

            predictions = model.predict(
                validation_df[features]
            )

            metrics = evaluate_model(
                zone,
                fold_number,
                model_name,
                validation_df,
                predictions,
            )

            metrics["train_records"] = len(train_df)

            all_metrics.append(metrics)

            all_predictions.append(
                create_prediction_records(
                    zone,
                    fold_number,
                    model_name,
                    validation_df,
                    predictions,
                )
            )

            print(
                f"{model_name}: "
                f"MAE={metrics['overall_mae']:.3f}, "
                f"RMSE={metrics['overall_rmse']:.3f}, "
                f"Bias={metrics['overall_bias']:.3f}"
            )

        print(
            "Persistence MAE:",
            round(baseline_metrics["overall_mae"], 3),
        )

    return all_metrics, all_predictions


# ==================================================
# PREDICTION RECORDS
# ==================================================

def create_prediction_records(
    zone,
    fold_number,
    model_name,
    validation_df,
    predictions,
):

    return pd.DataFrame({
        "zone": zone,
        "fold": fold_number,
        "model": model_name,
        "prediction_time": (
            validation_df["datetime"].to_numpy()
        ),
        "target_time": (
            validation_df["datetime"]
            + pd.Timedelta(hours=1)
        ).to_numpy(),
        "actual_carbon": (
            validation_df["target"].to_numpy()
        ),
        "prediction": predictions,
        "target_is_estimated": (
            validation_df["target_is_estimated"].to_numpy()
        ),
    })


# ==================================================
# FINAL SUMMARY
# ==================================================

def create_summary(metrics_df):

    summary = (
        metrics_df
        .groupby(
            ["zone", "model"],
            as_index=False,
        )
        .agg(
            mean_fold_mae=("overall_mae", "mean"),
            std_fold_mae=("overall_mae", "std"),
            mean_fold_rmse=("overall_rmse", "mean"),
            mean_fold_bias=("overall_bias", "mean"),
            total_validation_records=(
                "validation_records",
                "sum",
            ),
        )
    )

    return summary


# ==================================================
# MAIN
# ==================================================

def main():

    print("==========================================")
    print("ROLLING-ORIGIN CARBON VALIDATION")
    print("==========================================")

    df = load_dataset()

    all_metrics = []
    all_predictions = []

    for zone in ZONES:

        if zone not in df["zone"].unique():
            continue

        metrics, predictions = run_zone_validation(
            df,
            zone,
        )

        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

    metrics_df = pd.DataFrame(all_metrics)

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary_df = create_summary(
        metrics_df
    )

    metrics_path = (
        RESULT_DIR
        / "rolling_validation_metrics.csv"
    )

    predictions_path = (
        RESULT_DIR
        / "rolling_validation_predictions.csv"
    )

    summary_path = (
        RESULT_DIR
        / "rolling_validation_summary.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n==========================================")
    print("ROLLING VALIDATION SUMMARY")
    print("==========================================")

    print(
        summary_df.round(3).to_string(
            index=False
        )
    )

    print("\nSaved fold metrics:", metrics_path)
    print("Saved predictions:", predictions_path)
    print("Saved summary:", summary_path)

    print("\nValidation completed successfully.")

    print(
        "\nNote: The final 10% of the chronological "
        "dataset was not used in these validation folds."
    )


if __name__ == "__main__":
    main()