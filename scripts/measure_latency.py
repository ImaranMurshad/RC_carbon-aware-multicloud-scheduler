"""Measure real HTTPS round-trip latency from this machine to each cloud region.

Writes config/latency.json, replacing the placeholder values.
Uses public regional endpoints (no credentials needed).
Run from the project root:  python scripts/measure_latency.py
"""
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ENDPOINTS = {
    "AWS": "https://ec2.eu-west-1.amazonaws.com/ping",                      # Ireland
    "AZURE": "https://germanywestcentral.api.cognitive.microsoft.com/",     # Frankfurt
    "GCP": "https://europe-west9-run.googleapis.com/",                      # Paris
}
SAMPLES = 50
OUT = Path(__file__).resolve().parents[1] / "config" / "latency.json"


def measure(url):
    session = requests.Session()
    times = []
    for i in range(SAMPLES + 1):
        start = time.perf_counter()
        try:
            session.get(url, timeout=5)
        except requests.RequestException:
            continue
        elapsed = (time.perf_counter() - start) * 1000
        if i > 0:  # drop first call (TLS handshake)
            times.append(elapsed)
        time.sleep(0.1)
    return times


def main():
    result, detail = {}, {}
    for provider, url in ENDPOINTS.items():
        t = measure(url)
        if not t:
            raise SystemExit(f"No successful requests to {provider} ({url})")
        result[provider] = round(statistics.median(t), 1)
        detail[provider] = {
            "endpoint": url,
            "samples": len(t),
            "median_ms": result[provider],
            "p95_ms": round(sorted(t)[int(0.95 * (len(t) - 1))], 1),
        }
        print(f"{provider}: median {result[provider]} ms ({len(t)} samples)")

    OUT.write_text(json.dumps({
        "_description": "Measured HTTPS round-trip latency from the client machine.",
        "measured": True,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": result,
        "detail": detail,
    }, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()