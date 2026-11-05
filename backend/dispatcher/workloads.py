"""Kubernetes Job definitions for the four thesis workload types.

Each workload is a one-shot Job that runs a small self-contained Python
program (PAYLOAD) and prints a single line:  RESULT {json}
No custom image is needed: only python:3.11-alpine.
"""

import json

# Inline program executed inside the container (python -c).
PAYLOAD = r'''
import os, time, json, hashlib, threading
import http.server, urllib.request

T = os.environ["WL_TYPE"]
DUR = float(os.environ["WL_SECONDS"])
MEM = int(os.environ.get("WL_ALLOC_MB", "0"))
def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except Exception:
        return None


def _cpu_seconds():
    """CPU time used by this container (cgroup v2, then v1)."""
    s = _read("/sys/fs/cgroup/cpu.stat")
    if s:
        for line in s.splitlines():
            if line.startswith("usage_usec"):
                return int(line.split()[1]) / 1e6
    s = _read("/sys/fs/cgroup/cpuacct/cpuacct.usage")
    if s:
        return int(s.strip()) / 1e9
    return None


def _mem_peak_mb():
    for path in ("/sys/fs/cgroup/memory.peak",
                 "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        s = _read(path)
        if s and s.strip().isdigit():
            return int(s.strip()) / (1024 * 1024)
    return None


cpu_start = _cpu_seconds()
start = time.time()
end = start + DUR
out = {"type": T, "target_seconds": DUR}

if T == "cpu_bound":
    n = 0
    while time.time() < end:
        s = 0
        for i in range(20000):
            s += i * i % 7
        n += 1
    out["work_units"] = n

elif T == "mem_bound":
    buf = bytearray(MEM * 1024 * 1024)
    for i in range(0, len(buf), 4096):
        buf[i] = 1
    n = 0
    step = 4096 * 16
    while time.time() < end:
        for i in range(0, len(buf), step):
            buf[i] = (buf[i] + 1) % 256
            n += 1
    out["work_units"] = n
    out["allocated_mb"] = MEM

elif T == "background":
    data = b"x" * 65536
    n = 0
    while time.time() < end:
        hashlib.sha256(data).digest()
        n += 1
        time.sleep(0.01)
    out["work_units"] = n

elif T == "web":
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"ok" * 256
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8080), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    lat = []
    while time.time() < end:
        t0 = time.perf_counter()
        urllib.request.urlopen("http://127.0.0.1:8080/").read()
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    out["work_units"] = len(lat)
    out["service_latency_p50_ms"] = round(lat[len(lat) // 2], 3)
    out["service_latency_p95_ms"] = round(lat[int(0.95 * (len(lat) - 1))], 3)
    srv.shutdown()
else:
    raise SystemExit("unknown workload type " + T)

wall = time.time() - start
cpu_end = _cpu_seconds()
cpu_s = None if (cpu_end is None or cpu_start is None) else cpu_end - cpu_start
mem_mb = _mem_peak_mb()
out["cpu_seconds"] = None if cpu_s is None else round(cpu_s, 3)
out["mem_peak_mb"] = None if mem_mb is None else round(mem_mb, 1)
out["wall_seconds"] = round(wall, 2)
out["throughput_per_s"] = round(out["work_units"] / wall, 2)
print("RESULT " + json.dumps(out), flush=True)
'''

# Fraction of the memory request that the memory-bound workload allocates.
ALLOC_FRACTION = 0.6


def build_job(name, namespace, image, wl_type, cpu_m, mem_mb, seconds,
              labels=None):
    """Return a Job manifest (a dict; JSON is valid YAML for kubectl)."""
    alloc = int(mem_mb * ALLOC_FRACTION) if wl_type == "mem_bound" else 0
    all_labels = {"app": "thesis-workload", "wl-type": wl_type.replace("_", "-")}
    all_labels.update(labels or {})
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": namespace, "labels": all_labels},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 600,
            "activeDeadlineSeconds": int(seconds) + 300,
            "template": {
                "metadata": {"labels": all_labels},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "workload",
                        "image": image,
                        "command": ["python", "-c", PAYLOAD],
                        "env": [
                            {"name": "WL_TYPE", "value": wl_type},
                            {"name": "WL_SECONDS", "value": str(seconds)},
                            {"name": "WL_ALLOC_MB", "value": str(alloc)},
                        ],
                        "resources": {
                            "requests": {"cpu": f"{cpu_m}m", "memory": f"{mem_mb}Mi"},
                            "limits": {"cpu": f"{cpu_m}m", "memory": f"{mem_mb}Mi"},
                        },
                    }],
                },
            },
        },
    }


def manifest_json(*args, **kwargs):
    return json.dumps(build_job(*args, **kwargs))