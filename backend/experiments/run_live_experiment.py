"""Live experiment runner (run on the EC2 server, inside tmux).

Stage 1  profile : run every (provider, workload type) pair R times to measure
                   throughput, service latency, CPU time and memory.
                   12 pairs x R repeats, one job at a time.
Stage 2  subset  : at K evenly spaced historical decision hours, let EVERY
                   strategy place the full workload mix with that hour's real
                   carbon/price/latency data, and actually run each placed
                   workload on the chosen cluster.

Jobs run strictly one at a time (the host has 2 vCPUs shared by 3 clusters).
Both stages are resumable: re-run the same command and it carries on.

    python backend/experiments/run_live_experiment.py --stage profile --repeats 10 --seconds 30
    python backend/experiments/run_live_experiment.py --stage subset --origins 8 --seconds 20
    python backend/experiments/run_live_experiment.py --stage subset --origins 8 --dry-run
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dispatcher"))

import dispatch  # noqa: E402  (sets KUBECONFIG if needed)
import replay_common as rc  # noqa: E402

LIVE_RUNS = dispatch.OUT_FILE
MANIFEST = dispatch.PROJECT_ROOT / "results" / "live" / "live_subset_manifest.csv"
MANIFEST_COLS = ["run_id", "strategy", "decision_time", "target_time",
                 "workload_index", "workload_type", "provider",
                 "current_carbon", "actual_carbon", "status"]
TYPES = dispatch.TYPES


def eta(done, total, t0):
    if done == 0:
        return "?"
    left = (time.time() - t0) / done * (total - done)
    return f"{left / 60:.0f} min left"


def stage_profile(a):
    providers = list(dispatch.load_config()[0]["providers"])
    have = {}
    if LIVE_RUNS.exists():
        df = pd.read_csv(LIVE_RUNS)
        ok = df[(df["strategy"] == "profile") & (df["status"] == "ok")]
        have = ok.groupby(["provider", "workload_type"]).size().to_dict()
    plan = []
    for rep in range(a.repeats):          # interleave pairs: drift spreads evenly
        for p in providers:
            for t in TYPES:
                if have.get((p, t), 0) <= rep:
                    plan.append((p, t))
    print(f"profile: {len(plan)} jobs to run (~{len(plan) * (a.seconds + 9) / 60:.0f} min)")
    if a.dry_run:
        return
    run_plan(plan, lambda item: dispatch.run_workload(
        item[0], item[1], a.seconds, "profile"), a)


def run_plan(plan, fn, a):
    t0, fails = time.time(), 0
    for i, item in enumerate(plan, 1):
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print("time budget reached; re-run the same command to continue")
            return
        rec = fn(item)
        ok = rec is not None and rec["status"] == "ok"
        fails = 0 if ok else fails + 1
        print(f"[{i}/{len(plan)}] {item} -> {'ok' if ok else 'FAILED ' + str(rec and rec['error'])[:80]}"
              f"  ({eta(i, len(plan), t0)})", flush=True)
        if fails >= 5:
            print("5 failures in a row: stopping. Check the clusters.")
            return


def stage_subset(a):
    exp = rc.load_experiment()
    providers, dec = exp["providers"], exp["decisions"]
    idx = sorted({round(i * (len(dec) - 1) / max(a.origins - 1, 1))
                  for i in range(a.origins)})
    chosen = dec.iloc[idx]
    done = set()
    if MANIFEST.exists():
        m = pd.read_csv(MANIFEST)
        done = set(zip(m["strategy"], m["decision_time"].astype(str), m["workload_index"]))

    rng = random.Random(a.seed)
    plan = []  # (strategy, decision row, workload index, workload, provider)
    for d in chosen.itertuples(index=False):
        order = list(exp["strategies"])
        rng.shuffle(order)                 # no strategy always goes first
        for s in order:
            rr = {"i": int(d.prediction_time.hour)}   # fixed, reproducible RR start
            placed = rc.place_decision(s, d.prediction_time, providers,
                                       exp["forecasts"], exp["current"],
                                       rc.sim.WORKLOAD_MIX, rr)
            for n, wl, prov, _ in placed:
                if (s, str(d.prediction_time), n) not in done:
                    plan.append((s, d, n, wl, prov))
    runnable = sum(1 for p in plan if p[4] is not None)
    print(f"subset: {len(chosen)} decision hours x {len(exp['strategies'])} strategies; "
          f"{runnable} jobs left (~{runnable * (a.seconds + 9) / 60:.0f} min)")
    if a.dry_run:
        for p in plan[:12]:
            print("  ", p[0], str(p[1].prediction_time)[:16], p[3]["type"], "->", p[4])
        return

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    new = not MANIFEST.exists()
    fh = MANIFEST.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
    if new:
        w.writeheader()

    def do(item):
        s, d, n, wl, prov = item
        zone = providers[prov]["zone"] if prov else None
        row = {"strategy": s, "decision_time": str(d.prediction_time),
               "target_time": str(d.target_time), "workload_index": n,
               "workload_type": wl["type"], "provider": prov or "",
               "current_carbon": exp["current"].get((d.prediction_time, zone), "") if zone else "",
               "actual_carbon": exp["actual"].get((d.target_time, zone), "") if zone else ""}
        if prov is None:
            row.update(run_id="", status="rejected")
            w.writerow(row); fh.flush()
            return None
        rec = dispatch.run_workload(prov, wl["type"], a.seconds, s)
        row.update(run_id=rec["run_id"], status=rec["status"])
        w.writerow(row); fh.flush()
        return rec

    try:
        run_plan([p for p in plan if p[4] is not None], do, a)
    finally:
        fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["profile", "subset"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--origins", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    (stage_profile if a.stage == "profile" else stage_subset)(a)


if __name__ == "__main__":
    main()
