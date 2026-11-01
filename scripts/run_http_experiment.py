import concurrent.futures
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

TARGETS = {
    "Azure-Simulation": "http://127.0.0.1:8081",
    "GCP-Simulation": "http://127.0.0.1:8082",
}

REQUESTS_PER_CLUSTER = 1500
REQUEST_INTERVAL = 0.2
TIMEOUT = 10

OUTPUT = Path("results/telemetry/local_http_experiment.csv")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def send_request(environment, url, request_number):
    started = time.perf_counter()

    try:
        response = requests.get(url, timeout=TIMEOUT)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return {
            "timestamp_utc": utc_now(),
            "environment": environment,
            "request_number": request_number,
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response_time_ms": round(elapsed_ms, 3),
            "error": "",
        }

    except requests.RequestException as error:
        elapsed_ms = (time.perf_counter() - started) * 1000

        return {
            "timestamp_utc": utc_now(),
            "environment": environment,
            "request_number": request_number,
            "status_code": "",
            "success": False,
            "response_time_ms": round(elapsed_ms, 3),
            "error": str(error),
        }


def run_environment(environment, url):
    results = []

    print(f"\nStarting {environment}: {REQUESTS_PER_CLUSTER} requests")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for number in range(1, REQUESTS_PER_CLUSTER + 1):
            future = executor.submit(
                send_request, environment, url, number
            )
            results.append(future.result())

            if number % 100 == 0:
                print(f"{environment}: {number}/{REQUESTS_PER_CLUSTER}")

            time.sleep(REQUEST_INTERVAL)

    return results


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("Starting simultaneous HTTP experiment.")
    print("Both environments will receive 1,500 requests.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_environment, env, url): env
            for env, url in TARGETS.items()
        }

        all_results = []

        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

    fields = [
        "timestamp_utc",
        "environment",
        "request_number",
        "status_code",
        "success",
        "response_time_ms",
        "error",
    ]

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(
            all_results,
            key=lambda row: (row["environment"], row["request_number"])
        ))

    print("\nExperiment completed.")

    for environment in TARGETS:
        records = [
            row for row in all_results
            if row["environment"] == environment
        ]

        successful = sum(row["success"] for row in records)
        failed = len(records) - successful

        print(
            f"{environment}: "
            f"Total={len(records)}, "
            f"Successful={successful}, "
            f"Failed={failed}"
        )

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
