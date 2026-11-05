#!/usr/bin/env python3
"""Load readings collected on the EC2 server into the Django database (run on the Mac).

    python scripts/import_ec2_carbon.py ~/Downloads/carbon_ec2_15min.csv --dry-run
    python scripts/import_ec2_carbon.py ~/Downloads/carbon_ec2_15min.csv

Existing (zone, timestamp) readings are never overwritten. Afterwards run
scripts/update_carbon_history.py as before.
"""
import argparse
import os
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZONES = {"IE", "DE", "FR"}


def read_rows(path):
    df = pd.read_csv(path)
    need = {"zone", "datetime", "carbon_intensity"}
    if not need <= set(df.columns):
        raise SystemExit(f"{path} is missing columns: {sorted(need - set(df.columns))}")
    df = df[df["zone"].isin(ZONES)].copy()
    df["timestamp"] = pd.to_datetime(df["datetime"], utc=True)
    df["updated_at"] = pd.to_datetime(df.get("updated_at"), utc=True, errors="coerce")
    df["carbon_intensity"] = pd.to_numeric(df["carbon_intensity"], errors="coerce")
    df["is_estimated"] = df.get("is_estimated", False).astype(str).str.lower().eq("true")
    df = df.dropna(subset=["timestamp", "carbon_intensity"])
    return df.drop_duplicates(["zone", "timestamp"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    df = read_rows(a.csv)

    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django
    django.setup()
    from data_collection.models import CarbonReading

    existing = set(CarbonReading.objects.values_list("zone", "timestamp"))
    new = [r for r in df.itertuples() if (r.zone, r.timestamp.to_pydatetime()) not in existing]
    print(f"CSV rows: {len(df)}   already in database: {len(df) - len(new)}   new: {len(new)}")
    print(df.groupby("zone")["timestamp"].agg(["min", "max", "count"]).to_string())
    if a.dry_run or not new:
        return
    CarbonReading.objects.bulk_create([
        CarbonReading(
            zone=r.zone, timestamp=r.timestamp.to_pydatetime(),
            carbon_intensity=Decimal(str(r.carbon_intensity)),
            updated_at=None if pd.isna(r.updated_at) else r.updated_at.to_pydatetime(),
            is_estimated=bool(r.is_estimated),
            temporal_granularity=getattr(r, "temporal_granularity", "15_minutes"),
            raw_response=None,
        ) for r in new
    ], ignore_conflicts=True)
    print(f"Imported {len(new)} readings.")


if __name__ == "__main__":
    main()
