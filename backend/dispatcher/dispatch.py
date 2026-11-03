"""Dispatch a workload to a chosen cluster with kubectl and record the result.

Examples (run from the project root):
    python backend/dispatcher/dispatch.py --provider GCP --type cpu_bound --seconds 20
    python backend/dispatcher/dispatch.py --provider GCP --type web --dry-run
    python backend/dispatcher/dispatch.py --smoke-test --providers GCP AZURE
"""

import argparse
import csv
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))

import workloads  # noqa: E402
import run_scheduler_experiments_v2 as sim  # noqa: E402

PROJECT_ROOT = HERE.parents[1]
CLUSTERS_FILE = PROJECT_ROOT / "config" / "clusters.json"
LATENCY_FILE = PROJECT_ROOT / "config" / "latency.json"
OUT_FILE = PROJECT_ROOT / "results" / "live" / "live_runs.csv"

TYPES = ["web", "cpu_bound", "mem_bound", "background"]
# Resource request and SLA per type come from the simulation's workload mix.
SPECS = {}
for _w in sim.WORKLOAD_MIX:
    SPECS.setdefault(_w["type"], _w)

COLUMNS = [
    "run_id", "strategy", "provider", "context", "workload_type",
    "cpu_millicores", "memory_mb", "sla_ms", "requested_seconds",
    "submitted_at_utc", "total_seconds", "startup_seconds", "wall_seconds",
    "work_units", "throughput_per_s", "service_latency_p50_ms",
    "service_latency_p95_ms", "network_latency_ms", "response_latency_ms",
    "sla_breach", "peak_cpu_m", "peak_mem_mi", "status", "error",
]


def load_config():
    cfg = json.loads(CLUSTERS_FILE.read_text())
    lat = json.loads(LATENCY_FILE.read_text())["latency_ms"]
    return cfg, lat


def kubectl(context, *args, input_text=None, timeout=60):
    cmd = ["kubectl", "--context", context, *args]
    return subprocess.run(cmd, input=input_text, capture_output=True,
                          text=True, timeout=timeout)


def ensure_namespace(context, namespace):
    r = kubectl(context, "create", "namespace", namespace)
    if r.returncode != 0 and "AlreadyExists" not in r.stderr:
        raise RuntimeError(f"cannot create namespace: {r.stderr.strip()}")


def parse_cpu_m(text):
    return int(text[:-1]) if text.endswith("m") else int(float(text) * 1000)


def parse_mem_mi(text):
    if text.endswith("Mi"):
        return int(text[:-2])
    if text.endswith("Gi"):
        return int(float(text[:-2]) * 1024)
    if text.endswith("Ki"):
        return int(int(text[:-2]) / 1024)
    return 0


def run_workload(provider, wl_type, seconds=30, strategy="manual",
                 dry_run=False, extra_timeout=240):
    cfg, lat = load_config()
    context = cfg["providers"][provider]["context"]
    namespace = cfg["namespace"]
    spec = SPECS[wl_type]
    run_id = uuid.uuid4().hex[:8]
    name = f"wl-{wl_type.replace('_', '-')}-{run_id}"

    job = workloads.build_job(
        name, namespace, cfg["image"], wl_type, spec["cpu"], spec["mem"],
        seconds, labels={"provider": provider.lower(), "strategy": strategy[:40].replace("_", "-")},
    )
    if dry_run:
        shown = json.loads(json.dumps(job))
        c = shown["spec"]["template"]["spec"]["containers"][0]
        c["command"] = ["python", "-c", "<payload omitted>"]
        print(json.dumps(shown, indent=2))
        return None

    rec = {
        "run_id": run_id, "strategy": strategy, "provider": provider,
        "context": context, "workload_type": wl_type,
        "cpu_millicores": spec["cpu"], "memory_mb": spec["mem"],
        "sla_ms": spec["sla_ms"], "requested_seconds": seconds,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "network_latency_ms": lat[provider], "status": "failed", "error": "",
        "peak_cpu_m": 0, "peak_mem_mi": 0,
    }
    t0 = time.time()
    try:
        ensure_namespace(context, namespace)
        r = kubectl(context, "apply", "-f", "-", input_text=json.dumps(job))
        if r.returncode != 0:
            raise RuntimeError(f"apply failed: {r.stderr.strip()}")

        deadline = time.time() + seconds + extra_timeout
        while True:
            if time.time() > deadline:
                raise TimeoutError("job did not finish in time")
            st = kubectl(context, "get", "job", name, "-n", namespace, "-o",
                         "jsonpath={.status.succeeded},{.status.failed}")
            succeeded, _, failed = st.stdout.partition(",")
            if succeeded.strip() == "1":
                break
            if failed.strip() not in ("", "0"):
                raise RuntimeError("job failed (check: kubectl describe job)")
            top = kubectl(context, "top", "pod", "-n", namespace, "-l",
                          f"job-name={name}", "--no-headers")
            if top.returncode == 0 and top.stdout.split():
                parts = top.stdout.split()
                rec["peak_cpu_m"] = max(rec["peak_cpu_m"], parse_cpu_m(parts[1]))
                rec["peak_mem_mi"] = max(rec["peak_mem_mi"], parse_mem_mi(parts[2]))
            time.sleep(3)

        logs = kubectl(context, "logs", f"job/{name}", "-n", namespace)
        line = next((l for l in logs.stdout.splitlines()
                     if l.startswith("RESULT ")), None)
        if line is None:
            raise RuntimeError("no RESULT line in pod logs")
        res = json.loads(line[len("RESULT "):])
        rec.update({k: res.get(k, "") for k in (
            "wall_seconds", "work_units", "throughput_per_s",
            "service_latency_p50_ms", "service_latency_p95_ms")})
        rec["status"] = "ok"
    except Exception as exc:  # recorded, not hidden
        rec["error"] = str(exc)[:300]
    finally:
        rec["total_seconds"] = round(time.time() - t0, 2)
        if rec["status"] == "ok":
            rec["startup_seconds"] = round(
                rec["total_seconds"] - float(rec["wall_seconds"]), 2)
            svc = float(rec["service_latency_p50_ms"] or 0)
            rec["response_latency_ms"] = round(rec["network_latency_ms"] + svc, 2)
            rec["sla_breach"] = rec["response_latency_ms"] > rec["sla_ms"]
        kubectl(context, "delete", "job", name, "-n", namespace,
                "--ignore-not-found=true")
    save(rec)
    return rec


def save(rec):
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT_FILE.exists()
    with OUT_FILE.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(rec)


def show(rec):
    if rec is None:
        return
    keys = ["provider", "workload_type", "status", "total_seconds",
            "startup_seconds", "throughput_per_s", "response_latency_ms",
            "sla_breach", "peak_cpu_m", "peak_mem_mi", "error"]
    print("  " + "  ".join(f"{k}={rec.get(k, '')}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["AWS", "AZURE", "GCP"])
    ap.add_argument("--type", choices=TYPES)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run each workload type once on each provider")
    ap.add_argument("--providers", nargs="+", default=["AWS", "AZURE", "GCP"])
    a = ap.parse_args()

    if a.smoke_test:
        for p in a.providers:
            for t in TYPES:
                print(f"{p} / {t}")
                show(run_workload(p, t, a.seconds, "smoke_test"))
        print(f"\nResults appended to {OUT_FILE}")
        return
    if not (a.provider and a.type):
        ap.error("give --provider and --type (or use --smoke-test)")
    show(run_workload(a.provider, a.type, a.seconds, "manual", a.dry_run))


if __name__ == "__main__":
    main()