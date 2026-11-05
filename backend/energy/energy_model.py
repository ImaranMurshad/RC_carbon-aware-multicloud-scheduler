"""Energy and carbon model for one workload pod.

Uses the Cloud Carbon Footprint power model (config/energy_coefficients.json):

    watts_cpu = vCPU_requested * (min + utilisation * (max - min))
    watts_mem = GB_requested * 0.392
    energy_kWh = (watts_cpu + watts_mem) * hours * PUE / 1000
    gCO2e = energy_kWh * grid_intensity_gCO2_per_kWh

Utilisation is MEASURED inside the pod: container CPU seconds (cgroup) divided
by (requested vCPU x wall seconds). The measured utilisation profile of a
short run is assumed to hold for the one-hour scheduling interval.
If cgroup CPU time is missing, kubectl-top peak CPU is used and flagged.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COEFF_FILE = PROJECT_ROOT / "config" / "energy_coefficients.json"

_COEFF = None


def coefficients():
    global _COEFF
    if _COEFF is None:
        _COEFF = json.loads(COEFF_FILE.read_text())
    return _COEFF


def _num(x):
    """float(x), or None for None / '' / NaN / non-numeric."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def utilisation(cpu_request_m, wall_seconds, cpu_seconds=None, peak_cpu_m=None):
    """Return (utilisation 0..1, source). source: 'cgroup' | 'kubectl_top' | 'none'."""
    vcpu = cpu_request_m / 1000.0
    cpu_s, wall, peak = _num(cpu_seconds), _num(wall_seconds), _num(peak_cpu_m)
    if cpu_s is not None and wall and wall > 0:
        return min(max(cpu_s / (vcpu * wall), 0.0), 1.0), "cgroup"
    if peak and peak > 0:
        return min(peak / cpu_request_m, 1.0), "kubectl_top"
    return 0.0, "none"


def power_watts(provider, cpu_request_m, mem_request_mb, util):
    c = coefficients()
    p = c["providers"][provider]
    vcpu = cpu_request_m / 1000.0
    cpu_w = vcpu * (p["min_watts_per_vcpu"]
                    + util * (p["max_watts_per_vcpu"] - p["min_watts_per_vcpu"]))
    mem_w = (mem_request_mb / 1024.0) * c["memory_watts_per_gb"]
    return (cpu_w + mem_w) * p["pue"]


def energy_kwh(power_w, hours=1.0):
    return power_w * hours / 1000.0


def gco2e(kwh, intensity_g_per_kwh):
    return kwh * intensity_g_per_kwh
