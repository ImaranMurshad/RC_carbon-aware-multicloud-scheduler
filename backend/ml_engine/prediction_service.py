
from pathlib import Path

import joblib
import pandas as pd


# ==================================================
# PROJECT PATHS AND CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "carbon_forecasting"

HISTORICAL_DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "historical"
    / "carbon_history.csv"
)

ZONES = ["IE", "DE", "FR"]

FEATURES = [
    "current_carbon",
    "carbon_1h_ago",
    "carbon_2h_ago",
    "carbon_3h_ago",
    "hour",
    "day_of_week",
]

FORECAST_HORIZON_HOURS = 1


# ==================================================
# COMMON VALIDATION
# ==================================================

def validate_prediction_time(prediction_time):
    """
    Validate and normalise a timezone-aware forecast origin.

    Forecast origins must be aligned to an exact UTC hour.
    """

    prediction_time = pd.Timestamp(prediction_time)

    if prediction_time.tzinfo is None:
        raise ValueError(
            "prediction_time must include a timezone."
        )

    prediction_time = prediction_time.tz_convert("UTC")

    if (
        prediction_time.minute != 0
        or prediction_time.second != 0
        or prediction_time.microsecond != 0
        or prediction_time.nanosecond != 0
    ):
        raise ValueError(
            "prediction_time must be aligned to an exact hour."
        )

    return prediction_time


def validate_zone(zone):
    """Validate the requested electricity carbon-intensity zone."""

    if zone not in ZONES:
        raise ValueError(f"Unsupported zone: {zone}")


# ==================================================
# MODEL LOADING
# ==================================================

def load_model(zone):
    """Load and validate the saved zone-specific forecasting model."""

    validate_zone(zone)

    model_path = MODEL_DIR / f"random_forest_{zone}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found for zone {zone}: {model_path}"
        )

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("Invalid model bundle.")

    if bundle.get("zone") != zone:
        raise ValueError("Model zone mismatch.")

    if bundle.get("features") != FEATURES:
        raise ValueError("Unexpected model feature configuration.")

    if bundle.get("forecast_horizon_hours") != FORECAST_HORIZON_HOURS:
        raise ValueError("Unexpected model forecast horizon.")

    if "model" not in bundle:
        raise ValueError("Model bundle does not contain a model.")

    return bundle


# ==================================================
# HISTORICAL DATA LOADING
# ==================================================

def load_historical_data():
    """Load and validate the historical carbon-intensity dataset."""

    if not HISTORICAL_DATASET.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {HISTORICAL_DATASET}"
        )

    df = pd.read_csv(HISTORICAL_DATASET)

    required_columns = [
        "datetime",
        "zone",
        "carbon_intensity",
        "is_estimated",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
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
        ]
    ).copy()

    if df["zone"].isin(ZONES).all() is False:
        raise ValueError(
            "Dataset contains unsupported or invalid zone values."
        )

    if df.duplicated(["zone", "datetime"]).any():
        raise ValueError(
            "Duplicate zone/timestamp observations found."
        )

    if (
        (df["carbon_intensity"] < 0).any()
        or (~df["carbon_intensity"].map(pd.notna)).any()
    ):
        raise ValueError(
            "Invalid carbon-intensity values found."
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

    if df["is_estimated"].isna().any():
        raise ValueError(
            "Invalid is_estimated values found."
        )

    return (
        df.sort_values(["zone", "datetime"])
        .reset_index(drop=True)
    )


# ==================================================
# HISTORICAL CARBON PREDICTION
# RANDOM FOREST MODEL
# ==================================================

def predict_historical(zone, prediction_time):
    """
    Generate a one-hour-ahead carbon forecast using
    the trained zone-specific Random Forest model.

    prediction_time represents the forecast origin,
    not the target time.

    The model uses:
        - Current carbon intensity
        - Carbon intensity 1 hour ago
        - Carbon intensity 2 hours ago
        - Carbon intensity 3 hours ago
        - Hour of day
        - Day of week
    """

    validate_zone(zone)

    prediction_time = validate_prediction_time(
        prediction_time
    )

    # Load historical carbon observations.
    df = load_historical_data()

    zone_df = (
        df[df["zone"] == zone]
        .set_index("datetime")
        .sort_index()
    )

    # Retrieve current hour and three previous hours.
    input_times = [
        prediction_time - pd.Timedelta(hours=hours)
        for hours in range(4)
    ]

    observations = zone_df.reindex(input_times)

    if observations["carbon_intensity"].isna().any():
        raise ValueError(
            f"Insufficient continuous hourly history for {zone} "
            f"at {prediction_time.isoformat()}."
        )

    # Construct the six model features.
    features = pd.DataFrame(
        [
            {
                "current_carbon": observations.iloc[0]["carbon_intensity"],
                "carbon_1h_ago": observations.iloc[1]["carbon_intensity"],
                "carbon_2h_ago": observations.iloc[2]["carbon_intensity"],
                "carbon_3h_ago": observations.iloc[3]["carbon_intensity"],
                "hour": prediction_time.hour,
                "day_of_week": prediction_time.day_of_week,
            }
        ],
        columns=FEATURES,
    )

    # Load the zone-specific trained model.
    bundle = load_model(zone)

    model = bundle["model"]

    # Generate the one-hour-ahead forecast.
    predicted_carbon = float(
        model.predict(features)[0]
    )

    if not pd.notna(predicted_carbon) or predicted_carbon < 0:
        raise ValueError(
            "Random Forest generated an invalid carbon prediction."
        )

    return {
        "zone": zone,
        "prediction_time": prediction_time.isoformat(),
        "target_time": (
            prediction_time
            + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
        ).isoformat(),
        "predicted_carbon_intensity": predicted_carbon,
        "input_readings_estimated": [
            bool(value)
            for value in observations["is_estimated"].tolist()
        ],
        "input_data_source": "historical_csv",
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "model_type": type(model).__name__,
    }


# ==================================================
# PERSISTENCE FORECASTING BASELINE
# ==================================================

def predict_persistence(zone, prediction_time):
    """
    Generate a one-hour-ahead carbon forecast using
    the persistence baseline.

    The latest observed carbon intensity at the forecast
    origin is used as the prediction for the next hour.

    This baseline provides a reference against which
    the Random Forest model can be evaluated.
    """

    validate_zone(zone)

    prediction_time = validate_prediction_time(
        prediction_time
    )

    # Load the validated historical carbon dataset.
    df = load_historical_data()

    zone_df = (
        df[df["zone"] == zone]
        .set_index("datetime")
        .sort_index()
    )

    # Retrieve the observation available at the forecast origin.
    if prediction_time not in zone_df.index:
        raise ValueError(
            f"No historical observation available for {zone} "
            f"at {prediction_time.isoformat()}."
        )

    observation = zone_df.loc[prediction_time]

    observed_carbon = float(
        observation["carbon_intensity"]
    )

    if not pd.notna(observed_carbon) or observed_carbon < 0:
        raise ValueError(
            "Invalid carbon intensity in persistence input."
        )

    # Persistence assumption:
    # Next-hour carbon intensity equals the latest observation.
    predicted_carbon = observed_carbon

    return {
        "zone": zone,
        "prediction_time": prediction_time.isoformat(),
        "target_time": (
            prediction_time
            + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
        ).isoformat(),
        "predicted_carbon_intensity": predicted_carbon,
        "input_readings_estimated": [
            bool(observation["is_estimated"])
        ],
        "input_data_source": "historical_csv",
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "model_type": "PersistenceBaseline",
    }
