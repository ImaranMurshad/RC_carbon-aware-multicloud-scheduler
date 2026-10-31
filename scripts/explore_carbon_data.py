
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from pathlib import Path

# ==========================================
# CARBON-AWARE MULTI-CLOUD SCHEDULER
# Exploratory Data Analysis
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "datasets" / "carbon_readings.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eda"

ZONES = ["IE", "DE", "FR"]
ZONE_NAMES = {
    "IE": "Ireland",
    "DE": "Germany",
    "FR": "France"
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = {
        "zone",
        "carbon_intensity",
        "timestamp",
        "is_estimated"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    )

    df["carbon_intensity"] = pd.to_numeric(
        df["carbon_intensity"], errors="coerce"
    )

    df["is_estimated"] = (
        df["is_estimated"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    df = df.dropna(
        subset=["zone", "timestamp", "carbon_intensity"]
    )

    return df.sort_values(["zone", "timestamp"])


def generate_statistics(df):

    print("\n==========================================")
    print("CARBON INTENSITY STATISTICS")
    print("==========================================")

    statistics = (
        df.groupby("zone")["carbon_intensity"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max"
        )
        .round(3)
    )

    print(statistics)

    statistics.to_csv(
        OUTPUT_DIR / "carbon_statistics.csv"
    )

    return statistics


def analyse_data_quality(df):

    print("\n==========================================")
    print("DATA QUALITY ANALYSIS")
    print("==========================================")

    print(f"Total observations: {len(df)}")
    print(f"Duplicate zone/timestamps: "
          f"{df.duplicated(['zone', 'timestamp']).sum()}")

    print("\nEstimated readings:")
    print(df.groupby("zone")["is_estimated"].value_counts(
        dropna=False
    ))

    missing_records = []

    for zone in ZONES:

        zone_data = df[df["zone"] == zone]

        if zone_data.empty:
            continue

        expected_timestamps = pd.date_range(
            start=zone_data["timestamp"].min(),
            end=zone_data["timestamp"].max(),
            freq="15min"
        )

        actual_timestamps = pd.DatetimeIndex(
            zone_data["timestamp"].drop_duplicates()
        )

        missing_timestamps = expected_timestamps.difference(
            actual_timestamps
        )

        for timestamp in missing_timestamps:
            missing_records.append({
                "zone": zone,
                "missing_timestamp": timestamp
            })

        print(
            f"{zone}: {len(zone_data)} observations, "
            f"{len(missing_timestamps)} missing intervals"
        )

    missing_df = pd.DataFrame(
        missing_records,
        columns=["zone", "missing_timestamp"]
    )

    missing_df.to_csv(
        OUTPUT_DIR / "missing_timestamps.csv",
        index=False
    )

    return missing_df


def plot_carbon_trends(df):

    # Create a separate time series for each zone.
    fig, ax = plt.subplots(figsize=(12, 6))

    for zone in ZONES:

        zone_data = (
            df[df["zone"] == zone]
            .drop_duplicates("timestamp")
            .set_index("timestamp")
            .sort_index()
        )

        if zone_data.empty:
            continue

        # Reindex to expose missing 15-minute intervals.
        complete_index = pd.date_range(
            start=zone_data.index.min(),
            end=zone_data.index.max(),
            freq="15min"
        )

        zone_data = zone_data.reindex(complete_index)

        ax.plot(
            zone_data.index,
            zone_data["carbon_intensity"],
            marker="o",
            label=ZONE_NAMES[zone]
        )

    ax.set_title("Carbon Intensity Trends Across Three Zones")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Carbon Intensity (gCO2e/kWh)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "carbon_trends.png",
        dpi=300
    )

    plt.close(fig)


def plot_average_comparison(df):

    averages = (
        df.groupby("zone")["carbon_intensity"]
        .mean()
        .reindex(ZONES)
        .dropna()
    )

    labels = [ZONE_NAMES[zone] for zone in averages.index]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(labels, averages.values)

    ax.set_title("Average Carbon Intensity by Zone")
    ax.set_xlabel("Electricity Zone")
    ax.set_ylabel("Average Carbon Intensity (gCO2e/kWh)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "average_carbon_comparison.png",
        dpi=300
    )

    plt.close(fig)


def create_interactive_chart(df):

    chart_data = df.copy()

    chart_data["region"] = chart_data["zone"].map(
        ZONE_NAMES
    )

    fig = px.line(
        chart_data,
        x="timestamp",
        y="carbon_intensity",
        color="region",
        markers=True,
        title="Interactive Carbon Intensity Trends",
        labels={
            "timestamp": "Timestamp (UTC)",
            "carbon_intensity": "Carbon Intensity (gCO2e/kWh)",
            "region": "Region"
        }
    )

    fig.write_html(
        OUTPUT_DIR / "interactive_carbon_trends.html"
    )


def main():

    print("\n==========================================")
    print("EXPLORATORY DATA ANALYSIS")
    print("==========================================")

    df = load_data()

    print(f"\nDataset loaded: {len(df)} observations")
    print(f"Zones: {sorted(df['zone'].unique())}")

    generate_statistics(df)
    analyse_data_quality(df)

    plot_carbon_trends(df)
    plot_average_comparison(df)
    create_interactive_chart(df)

    print("\n==========================================")
    print("EDA COMPLETED SUCCESSFULLY")
    print("==========================================")

    print(f"\nOutput directory: {OUTPUT_DIR}")

    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.iterdir()):
        print(f"- {file.name}")


if __name__ == "__main__":
    main()
