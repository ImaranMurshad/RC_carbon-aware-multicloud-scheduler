import concurrent.futures
import csv
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

CONTEXTS = {
    "Azure-Simulation": {
        "context": "kind-azure-sim",
        "url": "http://127.0.0.1:8081",
    },
    "GCP-Simulation": {
        "context": "kind-gcp-sim",
        "url": "http://127.0.0.1:8082",
    },
}

REQUESTS_PER_CLUSTER = 1500
REQUEST_INTERVAL = 0.2
METRICS_INTERVAL = 15
PRE_IDLE_SECONDS = 60
RECOVERY_SECONDS = 120
REQUEST_TIMEOUT = 10

OUTPUT_DIR = Path("results/telemetry")
HTTP_OUTPUT = OUTPUT_DIR / "local_http_experiment.csv"
METRICS_OUTPUT = OUTPUT_DIR / "local_active_telemetry.csv"

phase = {"value": "PRE_IDLE"}
stop_metrics = threading.Event()
phase_lock = threading.Lock()

http_results = []
telemetry_results = []


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def get_phase():
    with phase_lock:
        return phase["value"]


def set_phase(value):
    with phase_lock:
        phase["value"] = value
    print(f"\n========== PHASE: {value} ==========")


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def collect_cluster_metrics(environment, context):
    timestamp = utc_now()
    current_phase = get_phase()

    node_output = run_command([
        "kubectl", "--context", context,
        "top", "nodes", "--no-headers"
    ])

    pod_output = run_command([
        "kubectl", "--context", context,
        "top", "pods", "--no-headers"
    ])

    node = node_output.split()

    if len(node) < 5:
        raise ValueError(f"Unexpected node metrics: {node_output}")

    rows = []

    for line in pod_output.strip().splitlines():
        parts = line.split()

        if len(parts) < 3:
            continue

        rows.append({
            "timestamp_utc": timestamp,
            "phase": current_phase,
            "environment": environment,
            "node_name": node[0],
            "node_cpu": node[1],
            "node_cpu_percent": node[2],
            "node_memory": node[3],
            "node_memory_percent": node[4],
            "pod_name": parts[0],
            "pod_cpu": parts[1],
            "pod_memory": parts[2],
            "collection_error": "",
        })

    return rows


def telemetry_worker():
    print("\nTelemetry collection started.")

    while not stop_metrics.is_set():
        for environment, config in CONTEXTS.items():
            try:
                rows = collect_cluster_metrics(
                    environment, config["context"]
                )
                telemetry_results.extend(rows)

                for row in rows:
                    print(
                        f"[{row['phase']}] {environment} | "
                        f"{row['pod_name']} | "
                        f"CPU {row['pod_cpu']} | "
                        f"Memory {row['pod_memory']}"
                    )

            except Exception as error:
                telemetry_results.append({
                    "timestamp_utc": utc_now(),
                    "phase": get_phase(),
                    "environment": environment,
                    "node_name": "",
                    "node_cpu": "",
                    "node_cpu_percent": "",
                    "node_memory": "",
                    "node_memory_percent": "",
                    "pod_name": "",
                    "pod_cpu": "",
                    "pod_memory": "",
                    "collection_error": str(error),
                })
                print(f"Telemetry error for {environment}: {error}")

        stop_metrics.wait(METRICS_INTERVAL)


def send_http_requests(environment, url):
    results = []

    print(
        f"\nStarting {environment}: "
        f"{REQUESTS_PER_CLUSTER} HTTP requests"
    )

    session = requests.Session()

    for number in range(1, REQUESTS_PER_CLUSTER + 1):
        start_time = time.perf_counter()
        started_at = utc_now()

        try:
            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT,
            )

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

            results.append({
                "timestamp_utc": started_at,
                "environment": environment,
                "request_number": number,
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response_time_ms": round(elapsed_ms, 3),
                "error": "",
            })

        except requests.RequestException as error:
            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

            results.append({
                "timestamp_utc": started_at,
                "environment": environment,
                "request_number": number,
                "status_code": "",
                "success": False,
                "response_time_ms": round(elapsed_ms, 3),
                "error": str(error),
            })

        if number % 100 == 0:
            print(
                f"{environment}: "
                f"{number}/{REQUESTS_PER_CLUSTER}"
            )

        time.sleep(REQUEST_INTERVAL)

    session.close()
    return results


def save_csv(path, rows, fields):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Checking both HTTP endpoints...")

    for environment, config in CONTEXTS.items():
        response = requests.get(
            config["url"],
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        print(f"{environment}: HTTP {response.status_code}")

    print("\nStarting coordinated experiment.")
    print(f"Pre-idle: {PRE_IDLE_SECONDS} seconds")
    print(f"Active: {REQUESTS_PER_CLUSTER} requests per cluster")
    print(f"Recovery: {RECOVERY_SECONDS} seconds")
    print(f"Telemetry interval: {METRICS_INTERVAL} seconds")

    metrics_thread = threading.Thread(
        target=telemetry_worker,
        daemon=True,
    )
    metrics_thread.start()

    try:
        set_phase("PRE_IDLE")
        time.sleep(PRE_IDLE_SECONDS)

        set_phase("ACTIVE_HTTP")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2
        ) as executor:

            futures = {
                executor.submit(
                    send_http_requests,
                    environment,
                    config["url"],
                ): environment
                for environment, config in CONTEXTS.items()
            }

            for future in concurrent.futures.as_completed(futures):
                environment = futures[future]
                results = future.result()
                http_results.extend(results)
                print(f"{environment}: HTTP workload finished.")

        set_phase("RECOVERY")
        time.sleep(RECOVERY_SECONDS)

    finally:
        stop_metrics.set()
        metrics_thread.join()

    http_fields = [
        "timestamp_utc",
        "environment",
        "request_number",
        "status_code",
        "success",
        "response_time_ms",
        "error",
    ]

    metrics_fields = [
        "timestamp_utc",
        "phase",
        "environment",
        "node_name",
        "node_cpu",
        "node_cpu_percent",
        "node_memory",
        "node_memory_percent",
        "pod_name",
        "pod_cpu",
        "pod_memory",
        "collection_error",
    ]

    save_csv(
        HTTP_OUTPUT,
        sorted(
            http_results,
            key=lambda row: (
                row["environment"], row["request_number"]
            ),
        ),
        http_fields,
    )

    save_csv(
        METRICS_OUTPUT,
        telemetry_results,
        metrics_fields,
    )

    print("\n========== EXPERIMENT SUMMARY ==========")

    for environment in CONTEXTS:
        records = [
            row for row in http_results
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

    print(f"\nHTTP results: {HTTP_OUTPUT}")
    print(f"Telemetry results: {METRICS_OUTPUT}")
    print(f"Telemetry records: {len(telemetry_results)}")


if __name__ == "__main__":
    main()
