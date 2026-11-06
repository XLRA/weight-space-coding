"""Logs both GPUs every 30 s and writes a warning when one runs hot: python3 gpu_temps.py [--every SECONDS]

Every sample goes to /opt/wsp/runs/gpu-temps.csv: core temperature, how far that is below the card's own
slowdown point (tlimit; the 5080 slows itself at about 87 C, the 5090 at about 90 C), fan, power, SM clock,
use, memory, and whether the card is slowing itself for heat or braking for power.

A line goes to /opt/wsp/runs/gpu-temps-warnings.log when a card
- is within 6 C of its slowdown point,
- is slowing itself for heat, or its power brake is on,
- shows 0 % fan at 65 C or more,
or when nvidia-smi fails or a card is missing (a card dropping off the bus). Each condition is reported when
it starts, every 10 minutes while it lasts, and once when it clears. Nothing here changes the cards or the runs.
GeForce cards don't report their memory (GDDR7) temperature, so that can't be logged.
One copy runs at a time (lock /opt/wsp/runs/.gpu-temps.lock).
"""
import fcntl
import os
import subprocess
import sys
import time

RUNS = "/opt/wsp/runs"
CSV = f"{RUNS}/gpu-temps.csv"
WARN = f"{RUNS}/gpu-temps-warnings.log"
FIELDS = ["timestamp", "index", "temperature.gpu", "temperature.gpu.tlimit", "fan.speed", "power.draw",
          "power.limit", "clocks.sm", "utilization.gpu", "memory.used",
          "clocks_event_reasons.hw_thermal_slowdown", "clocks_event_reasons.sw_thermal_slowdown",
          "clocks_event_reasons.hw_power_brake_slowdown"]
CARDS = {"0": "RTX 5080", "1": "RTX 5090"}
REPEAT = 600


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def warn(msg):
    with open(WARN, "a") as f:
        f.write(f"{stamp()} {msg}\n")


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


def sample():
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode:
        raise RuntimeError((out.stderr or out.stdout).strip() or f"exit code {out.returncode}")
    return [dict(zip(FIELDS, (c.strip() for c in line.split(",")))) for line in out.stdout.strip().splitlines()]


def problems(d):
    t, head, fan = num(d["temperature.gpu"]), num(d["temperature.gpu.tlimit"]), num(d["fan.speed"])
    p = {}
    if head is not None and head <= 6:
        p["hot"] = f"core {t:.0f} C, {head:.0f} C below its slowdown point"
    if "Active" in (d["clocks_event_reasons.hw_thermal_slowdown"], d["clocks_event_reasons.sw_thermal_slowdown"]):
        p["thermal slowdown"] = f"slowing itself for heat at core {t:.0f} C"
    if d["clocks_event_reasons.hw_power_brake_slowdown"] == "Active":
        p["power brake"] = f"hardware power brake on at {d['power.draw']} W"
    if fan == 0 and t is not None and t >= 65:
        p["fan"] = f"fan at 0 % with core at {t:.0f} C"
    return p


def main():
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 30
    lock = open(f"{RUNS}/.gpu-temps.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another gpu_temps.py is running")
    if not os.path.exists(CSV):
        with open(CSV, "w") as f:
            f.write("card," + ",".join(FIELDS) + "\n")
    warn(f"logger started (pid {os.getpid()}, every {every} s)")
    active = {}  # (card, condition) -> time last reported
    while True:
        found = {}
        try:
            rows = sample()
            with open(CSV, "a") as f:
                for d in rows:
                    f.write(CARDS.get(d["index"], d["index"]) + "," + ",".join(d[k] for k in FIELDS) + "\n")
            for d in rows:
                for cond, msg in problems(d).items():
                    found[(CARDS.get(d["index"], d["index"]), cond)] = msg
            missing = sorted(set(CARDS) - {d["index"] for d in rows})
            if missing:
                found[("all", "missing")] = "nvidia-smi does not list " + ", ".join(CARDS[i] for i in missing)
        except Exception as e:  # noqa: BLE001 - any failure to read the cards is itself worth a warning
            found[("all", "nvidia-smi")] = f"nvidia-smi failed: {e}"
        t = time.time()
        for key, msg in found.items():
            if key not in active:
                warn(f"WARNING {key[0]}: {msg}")
                active[key] = t
            elif t - active[key] >= REPEAT:
                warn(f"STILL {key[0]}: {msg}")
                active[key] = t
        for key in [k for k in active if k not in found]:
            warn(f"OK {key[0]}: {key[1]} cleared")
            del active[key]
        time.sleep(every)


if __name__ == "__main__":
    main()
