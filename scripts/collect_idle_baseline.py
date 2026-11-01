import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

CONTEXTS = {
    "Azure-Simulation": "kind-azure-sim",
    "GCP-Simulation": "kind-gcp-sim",
}

OUTPUT = Path("results/telemetry/local_idle_baseline.csv")
SAMPLES = 11
INTERVAL = 30


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def collect_metrics(environment, context):
    timestamp = datetime.now(timezone.utc).isoformat()

    node_output = run_command([
        "kubectl", "--context", context, "top", "nodes", "--no-headers"
    ])

    pod_output = run_command([
        "kubectl", "--context", context, "top", "pods", "--no-headers"
    ])

    node = node_output.split()
    rows = []

    for line in pod_output.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            rows.append({
                "timestamp_utc": timestamp,
                "environment": environment,
                "node_name": node[0],
                "node_cpu": node[1],
                "node_cpu_percent": node[2],
                "node_memory": node[3],
                "node_memory_percent": node[4],
                "pod_name": parts[0],
                "pod_cpu": parts[1],
                "pod_memory": parts[2],
            })

    return rows


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

fields = [
    "timestamp_utc", "environment", "node_name",
    "node_cpu", "node_cpu_percent",
    "node_memory", "node_memory_percent",
    "pod_name", "pod_cpu", "pod_memory"
]

with OUTPUT.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for sample in range(SAMPLES):
        print(f"\nCollecting sample {sample + 1}/{SAMPLES}")

        for environment, context in CONTEXTS.items():
            try:
                rows = collect_metrics(environment, context)
                writer.writerows(rows)

                for row in rows:
                    print(
                        f"{environment} | {row['pod_name']} | "
                        f"CPU: {row['pod_cpu']} | "
                        f"Memory: {row['pod_memory']}"
                    )

            except Exception as error:
                print(f"Error collecting {environment}: {error}")

        file.flush()

        if sample < SAMPLES - 1:
            time.sleep(INTERVAL)

print(f"\nCollection complete: {OUTPUT}")
