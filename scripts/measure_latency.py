"""Measure network round-trip latency from this machine to each represented region.

Uses TCP connect time to each region's public API endpoint on port 443. This is
one network round trip with no TLS or server processing, so the three providers
are measured the same way. Also records (for information only) the TCP time to
the real AWS instance, which physically runs in Stockholm.

Writes config/latency.json.  Run from the project root:
    python scripts/measure_latency.py
"""
import json
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

# Each provider's cluster represents a city. To measure the network distance to
# that city in one consistent way, we connect to AWS regional API endpoints in
# the same cities (these are real, unicast, in-region servers).
# Azure and GCP public endpoints are routed to the nearest edge, so they cannot
# be used to measure distance to a specific region.
ENDPOINTS = {
    "AWS": ("ec2.eu-west-1.amazonaws.com", 443),       # Dublin   (represents AWS / IE)
    "AZURE": ("ec2.eu-central-1.amazonaws.com", 443),  # Frankfurt (represents Azure / DE)
    "GCP": ("ec2.eu-west-3.amazonaws.com", 443),       # Paris    (represents GCP / FR)
}
# Where the live AWS cluster actually runs (information only).
AWS_INSTANCE = ("16.170.150.47", 22)

SAMPLES = 60
OUT = Path(__file__).resolve().parents[1] / "config" / "latency.json"


def tcp_connect_ms(host, port):
    times = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=5):
                pass
        except OSError:
            continue
        times.append((time.perf_counter() - start) * 1000)
        time.sleep(0.1)
    return times


def summarise(times):
    times = sorted(times)
    return {
        "samples": len(times),
        "median_ms": round(statistics.median(times), 1),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))], 1),
    }


def main():
    latency, detail = {}, {}
    for provider, (host, port) in ENDPOINTS.items():
        t = tcp_connect_ms(host, port)
        if not t:
            raise SystemExit(f"No successful connections to {provider} ({host})")
        detail[provider] = {"endpoint": f"{host}:{port}", **summarise(t)}
        latency[provider] = detail[provider]["median_ms"]
        print(f"{provider}: median {latency[provider]} ms ({len(t)} samples)")

    info = {}
    t = tcp_connect_ms(*AWS_INSTANCE)
    if t:
        info = {"aws_instance_stockholm": summarise(t)}
        print(f"(info) real AWS instance in Stockholm: median {info['aws_instance_stockholm']['median_ms']} ms")

    OUT.write_text(json.dumps({
        "_description": "TCP connect round-trip time from the client machine to each represented region.",
        "measured": True,
        "method": "tcp_connect_443_to_aws_regional_endpoints_in_represented_cities",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": latency,
        "detail": detail,
        "info": info,
    }, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()