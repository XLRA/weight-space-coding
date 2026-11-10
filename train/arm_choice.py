"""The plan's checkpoint rule for the GRPO arms, on the new tests (plans/training-phase.md, "Choosing the checkpoint";
Deviations, 2026-11-08 and 2026-11-10). Run on the host:
    python3 train/arm_choice.py ended <arm>             exit 0 once the run has logged done or has a stopped file
    python3 train/arm_choice.py ready <arm> <base>      exit 0 once it has ended and every checkpoint has a check
                                                        paired with <base> on the new tests
    python3 train/arm_choice.py choose <arm> <base> [--dry-run]
        prints the chosen checkpoint's folder, or nothing if none qualifies, and writes evals/<arm>/choice.json with
        the candidates. An existing choice.json is kept and read: an arm's checkpoint is chosen once. --dry-run writes
        nothing and prints the choice on stderr.
    python3 train/arm_choice.py last <arm> [<base>]     prints the arm's last checkpoint's folder (with a base, the
                                                        last one whose check is paired with it), or nothing
    python3 train/arm_choice.py panel-ok <check>        exit 0 unless the check's panel mean is more than 2 SE below its
                                                        base's, paired over the panel's questions (T's rule); 2 if the
                                                        check has no summary
The rule: the best validation pass rate on train/data2's tests (val2) among the checkpoints whose panel mean is no
more than 2.5 points below the arm's start. A tie goes to the earlier checkpoint. A checkpoint without a paired check
can't be judged and is left out. WSP_TRAIN points it at another train/ folder (for tests).
"""
import glob
import json
import os
import re
import sys
import time

T = os.environ.get("WSP_TRAIN", "/opt/wsp/train")
RULE = ("the best validation pass rate on train/data2's tests (val2) among the checkpoints whose panel mean is no "
        "more than 2.5 points below the base's; a tie goes to the earlier checkpoint; a checkpoint without a paired "
        "check is left out (plans/training-phase.md)")


def checkpoints(arm):
    """[(name, folder)] in step order."""
    ms = glob.glob(f"{T}/runs/{arm}/checkpoints/step_*/manifest.json")
    return sorted(((os.path.basename(os.path.dirname(m)), os.path.dirname(m)) for m in ms),
                  key=lambda c: int(c[0].split("_")[1]))


def ended(arm):
    if os.path.exists(f"{T}/runs/{arm}/stopped"):
        return True
    try:
        with open(f"{T}/runs/{arm}/train.log") as f:
            return any(re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d done", line.rstrip("\n")) for line in f)
    except OSError:
        return False


def summary(name):
    try:
        with open(f"{T}/evals/{name}/summary.json") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def paired(s, base):
    v = (s or {}).get("vs_base") or {}
    return v.get("base") == base and "val2" in s and "val2" in v


def candidate(arm, name, folder, base):
    s = summary(f"{arm}/{name}")
    if not paired(s, base):
        return {"checkpoint": name, "folder": folder, "checked": False}
    v = s["vs_base"]
    return {"checkpoint": name, "folder": folder, "checked": True, "val2_pass": s["val2"]["pass"],
            "val2_diff": v["val2"]["diff"], "val2_se": v["val2"]["se"], "val_pass_old_tests": s["val"]["pass"],
            "panel_mean": s["panel_mean"], "panel_mean_diff": v["panel_mean"]["diff"],
            "panel_mean_se": v["panel_mean"]["se"], "panel_within_2_5": v["panel_within_2_5"]}


def choose(arm, base, dry):
    path = f"{T}/evals/{arm}/choice.json"
    if os.path.exists(path):
        with open(path) as f:
            chosen = json.load(f)["chosen"]
        print(dict(checkpoints(arm)).get(chosen, "") if chosen else "")
        return
    cands = [candidate(arm, name, folder, base) for name, folder in checkpoints(arm)]
    ok = [c for c in cands if c["checked"] and c["panel_within_2_5"]]
    # max() keeps the first of equal values, and the candidates are in step order: a tie goes to the earlier one.
    best = max(ok, key=lambda c: c["val2_pass"], default=None)
    choice = {"arm": arm, "base": base, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "rule": RULE,
              "run_ended": ended(arm), "candidates": cands, "chosen": best["checkpoint"] if best else None}
    if dry:
        print(json.dumps(choice), file=sys.stderr)
    else:
        with open(path + ".tmp", "w") as f:
            json.dump(choice, f, indent=1)
        os.replace(path + ".tmp", path)
    print(best["folder"] if best else "")


def main():
    cmd, args = sys.argv[1], [a for a in sys.argv[2:] if a != "--dry-run"]
    if cmd == "ended":
        sys.exit(0 if ended(args[0]) else 1)
    if cmd == "ready":
        arm, base = args
        cs = checkpoints(arm)
        sys.exit(0 if ended(arm) and cs and all(paired(summary(f"{arm}/{n}"), base) for n, _ in cs) else 1)
    if cmd == "choose":
        choose(*args, dry="--dry-run" in sys.argv)
    elif cmd == "last":
        cs = [c for c in checkpoints(args[0]) if len(args) < 2 or paired(summary(f"{args[0]}/{c[0]}"), args[1])]
        print(cs[-1][1] if cs else "")
    elif cmd == "panel-ok":
        s = summary(args[0])
        if not s or not s.get("vs_base"):
            sys.exit(2)
        v = s["vs_base"]["panel_mean"]
        print(f"panel mean {s['panel_mean']:.1f}, {v['diff']:+.2f} (SE {v['se']:.2f}) against {s['vs_base']['base']}")
        sys.exit(1 if v["diff"] < -2 * v["se"] else 0)
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
