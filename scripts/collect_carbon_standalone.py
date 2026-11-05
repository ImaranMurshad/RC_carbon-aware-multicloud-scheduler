#!/usr/bin/env python3
"""Always-on carbon collector for the EC2 server (no Django, no database).

Calls the same Electricity Maps endpoint as the Django collector for IE, DE, FR
every 15 minutes and APPENDS new readings to datasets/raw/carbon_ec2_15min.csv
(that folder is git-ignored). A reading is only written once per (zone, datetime).

The API key is read from the environment (systemd EnvironmentFile), never from git.

    python scripts/collect_carbon_standalone.py --once      # one pass, then exit
    python scripts/collect_carbon_standalone.py             # forever (systemd)
"""
import argparse
import csv
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = PROJECT_ROOT / "datasets" / "raw" / "carbon_ec2_15min.csv"
API_URL = "https://api.electricitymaps.com/v4/carbon-intensity/latest"
ZONES = ["IE", "DE", "FR"]
COLUMNS = ["collected_at_utc", "zone", "datetime", "updated_at",
           "carbon_intensity", "is_estimated", "temporal_granularity"]
STOP = threading.Event()


def log(msg):
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z  {msg}", flush=True)


def fetch(zone, api_key):
    r = requests.get(
        API_URL,
        headers={"auth-token": api_key},
        params={"zone": zone, "temporalGranularity": "15_minutes",
                "emissionFactorType": "lifecycle", "flowTraced": "true"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def load_seen(path):
    seen = set()
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                seen.add((row["zone"], row["datetime"]))
    return seen


def to_row(zone, data):
    if data.get("carbonIntensity") is None or not data.get("datetime"):
        raise ValueError("response has no carbonIntensity/datetime")
    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "zone": zone,
        "datetime": data["datetime"],
        "updated_at": data.get("updatedAt", ""),
        "carbon_intensity": data["carbonIntensity"],
        "is_estimated": data.get("isEstimated", False),
        "temporal_granularity": data.get("temporalGranularity", "15_minutes"),
    }


def one_pass(api_key, seen, path=OUT_FILE, fetcher=fetch):
    """Fetch every zone once; append new readings. Returns number of new rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    added = 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        for zone in ZONES:
            try:
                row = to_row(zone, fetcher(zone, api_key))
            except Exception as exc:  # keep going; the next pass retries
                log(f"{zone}: FAILED - {str(exc)[:120]}")
                continue
            key = (zone, row["datetime"])
            if key in seen:
                log(f"{zone}: {row['carbon_intensity']} (already stored)")
                continue
            w.writerow(row)
            f.flush()
            seen.add(key)
            added += 1
            log(f"{zone}: {row['carbon_intensity']} gCO2e/kWh NEW ({row['datetime']})")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=900, help="seconds between passes")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    key = os.getenv("ELECTRICITY_MAPS_API_KEY")
    if not key:
        sys.exit("ELECTRICITY_MAPS_API_KEY is not set (see the systemd EnvironmentFile).")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.set())

    seen = load_seen(OUT_FILE)
    log(f"collector started; {len(seen)} readings already in {OUT_FILE.name}")
    while not STOP.is_set():
        t0 = time.time()
        one_pass(key, seen)
        if a.once:
            break
        STOP.wait(max(5, a.interval - (time.time() - t0)))
    log("collector stopped")


if __name__ == "__main__":
    main()
