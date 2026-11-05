from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = Path("datasets/historical/carbon_history.csv")
OUTPUT_DIR = Path("results/ml")

REQUIRED_COLUMNS = {
    "datetime",
    "zone",
    "carbon_intensity",
    "is_estimated",
}


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["carbon_intensity"] = pd.to_numeric(
        df["carbon_intensity"], errors="raise"
    )

    flags = df["is_estimated"].astype(str).str.strip().str.lower()

    if not flags.isin(["true", "false"]).all():
        raise ValueError("Unexpected values in is_estimated column.")

    df["is_estimated"] = flags.eq("true")

    if df.duplicated(["zone", "datetime"]).any():
        raise ValueError("Duplicate zone/timestamp records detected.")

    return df


def evaluate_zone(zone, group):
    group = group.sort_values("datetime").reset_index(drop=True)

    # Construct one-hour-ahead targets.
    group["actual_next_hour"] = group["carbon_intensity"].shift(-1)
    group["target_datetime"] = group["datetime"].shift(-1)
    group["target_is_estimated"] = group["is_estimated"].shift(-1)

    # Match the feature-engineering rows used in rolling validation.
    # Rows whose three lags and next-hour target are all exactly 1 h apart
    # (identical to iloc[3:-1] when there are no gaps).
    step = pd.Timedelta(hours=1)
    dt = group["datetime"]
    contiguous = (
        (dt.shift(-1) - dt == step)
        & (dt - dt.shift(1) == step)
        & (dt.shift(1) - dt.shift(2) == step)
        & (dt.shift(2) - dt.shift(3) == step)
    )
    usable = group[contiguous].copy().reset_index(drop=True)

    usable["prediction"] = usable["carbon_intensity"]

    # Keep the final 10% completely separate from model development.
    holdout_start = int(len(usable) * 0.90)
    holdout = usable.iloc[holdout_start:].copy()

    if holdout.empty or holdout["actual_next_hour"].isna().any():
        raise ValueError(f"Invalid holdout targets for zone {zone}.")

    y_true = holdout["actual_next_hour"]
    y_pred = holdout["prediction"]

    actual = holdout.loc[~holdout["target_is_estimated"].astype(bool)]
    estimated = holdout.loc[holdout["target_is_estimated"].astype(bool)]

    metrics = {
        "zone": zone,
        "prediction_method": "persistence",
        "forecast_horizon": "1 hour",
        "validation_records": len(holdout),
        "actual_target_records": len(actual),
        "estimated_target_records": len(estimated),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "bias": (y_true - y_pred).mean(),
        "actual_only_mae": (
            mean_absolute_error(
                actual["actual_next_hour"], actual["prediction"]
            )
            if not actual.empty else np.nan
        ),
        "estimated_only_mae": (
            mean_absolute_error(
                estimated["actual_next_hour"], estimated["prediction"]
            )
            if not estimated.empty else np.nan
        ),
    }

    predictions = holdout[
        ["zone", "datetime", "target_datetime",
         "carbon_intensity", "actual_next_hour",
         "prediction", "target_is_estimated"]
    ].copy()

    predictions = predictions.rename(
        columns={
            "datetime": "forecast_origin",
            "carbon_intensity": "current_carbon_intensity",
            "actual_next_hour": "actual_carbon_intensity",
            "prediction": "predicted_carbon_intensity",
        }
    )

    predictions["prediction_method"] = "persistence"

    return metrics, predictions


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    all_metrics = []
    all_predictions = []

    for zone, group in sorted(df.groupby("zone")):
        metrics, predictions = evaluate_zone(zone, group)

        all_metrics.append(metrics)
        all_predictions.append(predictions)

        print(f"\n===== {zone} FINAL HOLDOUT =====")
        print(f"Validation records: {metrics['validation_records']}")
        print(f"MAE:  {metrics['mae']:.3f}")
        print(f"RMSE: {metrics['rmse']:.3f}")
        print(f"Bias: {metrics['bias']:.3f}")
        print(f"Actual targets: {metrics['actual_target_records']}")
        print(f"Estimated targets: {metrics['estimated_target_records']}")
        print(f"Actual-only MAE: {metrics['actual_only_mae']:.3f}")
        print(f"Estimated-only MAE: {metrics['estimated_only_mae']:.3f}")

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics_path = OUTPUT_DIR / "final_holdout_metrics.csv"
    predictions_path = OUTPUT_DIR / "final_holdout_predictions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\n===== FILES SAVED =====")
    print(metrics_path)
    print(predictions_path)
    print("\nFinal holdout evaluation completed successfully.")


if __name__ == "__main__":
    main()