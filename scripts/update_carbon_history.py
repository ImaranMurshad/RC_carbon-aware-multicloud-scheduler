"""Append new hourly carbon data from the collector database to
datasets/historical/carbon_history.csv.

Why this exists: the collector (run_carbon_collector.sh) stores 15-minute
readings in the database, but the ML and scheduling code reads the hourly file
datasets/historical/carbon_history.csv, which nothing was updating.

What it does
  * Averages the 15-minute readings of each COMPLETE hour (needs >= 2 readings).
  * Appends only hours newer than the last row already in the CSV.
  * Fills small gaps (default <= 3 hours, e.g. laptop asleep) by linear
    interpolation and labels those rows source="interpolated", is_estimated=True.
    The ML scripts assume consecutive hourly rows, so a gap would silently
    corrupt the lag features.
  * Refuses to append past a larger gap and tells you where it is.
  * Saves a backup copy first.

Run from the project root:
    python scripts/update_carbon_history.py --dry-run     # preview
    python scripts/update_carbon_history.py               # write
"""
import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY = PROJECT_ROOT / "datasets" / "historical" / "carbon_history.csv"
SOURCE_LABEL = "Electricity Maps (collector, hourly mean)"
MIN_READINGS_PER_HOUR = 2


def load_readings_from_db():
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django
    django.setup()
    from data_collection.models import CarbonReading

    rows = list(CarbonReading.objects.values_list(
        "zone", "timestamp", "carbon_intensity", "is_estimated"))
    df = pd.DataFrame(rows, columns=["zone", "timestamp", "carbon_intensity",
                                     "is_estimated"])
    df["carbon_intensity"] = df["carbon_intensity"].astype(float)
    return df


def hourly_means(readings, now):
    """15-minute readings -> one row per zone per COMPLETE hour."""
    df = readings.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.floor("h")
    df = df[df["hour"] < pd.Timestamp(now).floor("h")]        # complete hours only
    g = df.groupby(["zone", "hour"]).agg(
        carbon_intensity=("carbon_intensity", "mean"),
        is_estimated=("is_estimated", "max"),
        n=("carbon_intensity", "size"),
    ).reset_index()
    g = g[g["n"] >= MIN_READINGS_PER_HOUR]
    g["carbon_intensity"] = g["carbon_intensity"].round(1)
    return g.rename(columns={"hour": "datetime"})


def extend_zone(history_zone, new_zone, zone, max_gap):
    """Return (rows_to_append, notes) for one zone, filling small gaps."""
    last = history_zone["datetime"].max()
    new_zone = new_zone[new_zone["datetime"] > last].sort_values("datetime")
    series = history_zone.set_index("datetime")["carbon_intensity"].to_dict()
    flags = {}
    notes, appended = [], []
    prev_time, prev_val = last, series[last]
    for row in new_zone.itertuples(index=False):
        gap = int((row.datetime - prev_time) / pd.Timedelta(hours=1)) - 1
        if gap > max_gap:
            notes.append(f"{zone}: gap of {gap} h after {prev_time} - stopped "
                         f"(larger than --max-gap {max_gap}); later data NOT appended")
            break
        for i in range(1, gap + 1):                   # fill the small gap
            t = prev_time + pd.Timedelta(hours=i)
            v = prev_val + (row.carbon_intensity - prev_val) * i / (gap + 1)
            appended.append((t, round(v, 1), True, "interpolated"))
        if gap:
            notes.append(f"{zone}: interpolated {gap} h between {prev_time} and {row.datetime}")
        appended.append((row.datetime, row.carbon_intensity,
                         bool(row.is_estimated), SOURCE_LABEL))
        prev_time, prev_val = row.datetime, row.carbon_intensity
    return appended, notes


def build_update(history, readings, now, max_gap):
    hourly = hourly_means(readings, now)
    out, notes = [], []
    for zone in sorted(history["zone"].unique()):
        hz = history[history["zone"] == zone]
        env = hz["environment"].iloc[0]
        rows, n = extend_zone(hz, hourly[hourly["zone"] == zone], zone, max_gap)
        notes += n
        for t, v, est, src in rows:
            out.append({"datetime": t, "zone": zone, "environment": env,
                        "carbon_intensity": v, "is_estimated": est, "source": src})
    return pd.DataFrame(out), notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-gap", type=int, default=3)
    a = ap.parse_args()

    history = pd.read_csv(HISTORY)
    history["datetime"] = pd.to_datetime(history["datetime"], utc=True)
    readings = load_readings_from_db()
    now = datetime.now(timezone.utc)

    new, notes = build_update(history, readings, now, a.max_gap)
    print(f"Existing rows: {len(history)}")
    print("Last hour per zone (before):")
    print(history.groupby("zone")["datetime"].max().to_string())
    for n in notes:
        print("NOTE:", n)
    if new.empty:
        print("\nNothing new to append.")
        return
    print(f"\nNew rows: {len(new)}  (interpolated: {(new['source'] == 'interpolated').sum()})")
    print(new.groupby("zone")["datetime"].agg(["count", "min", "max"]).to_string())
    if a.dry_run:
        print("\nDry run - nothing written.")
        return

    backup = HISTORY.with_name(f"carbon_history_{now:%Y%m%d_%H%M}_backup.csv")
    shutil.copy(HISTORY, backup)
    new["datetime"] = new["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    h = history.copy()
    h["datetime"] = h["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    cols = list(history.columns)
    pd.concat([h, new[cols]], ignore_index=True).to_csv(HISTORY, index=False)
    print(f"\nBackup: {backup.name}\nUpdated: {HISTORY}")


if __name__ == "__main__":
    main()