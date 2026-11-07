#!/bin/bash
# The RFT arms' held-out evaluations (plans/training-phase.md, "Choosing the checkpoint" and "Measurement"), on the
# 5090. Queued 2026-11-07 on the user's go-ahead to run between G-E and G-O: train/queue/after-G-E.sh calls it.
#   bash /opt/wsp/train/eval-rft-arms.sh [--dry-run]   (its log goes to stdout, which the GRPO queue appends to its log)
# 1. Each arm's checkpoint, by the rule fixed in the plan: the best validation pass rate among the checkpoints whose
#    panel mean is no more than 2.5 points below the arm's start (panel_within_2_5 in their checks). A tie goes to the
#    earlier checkpoint. The candidates and the choice go to train/evals/<arm>/choice.json.
# 2. On each chosen checkpoint: the five suites at presence penalty 1.5 (results/<arm>-pp15), then the code suite at
#    presence penalty 0 (results/<arm>-code-pp0), reusing the original model's answers (pilot-10-pp15, pilot-10).
# 3. E's code suite at presence penalty 0 (results/pilot-14-code-pp0). E was only evaluated at 1.5, and R-E and G-E are
#    paired with their start at 0 too.
# An arm is left out if, after up to 3 hours of waiting, its run has not logged done or its checkpoints are not all
# checked and paired with its start. Each evaluation is tried twice (evaluate.py resumes where it stopped).
# --dry-run: no waiting, no choice.json, no evaluations; it prints the choice and the commands. WSP_TRAIN points it at
# another train/ folder (for tests).
T=${WSP_TRAIN:-/opt/wsp/train}
EV=/opt/wsp/eval
PY=/opt/wsp/venv/bin/python
E=/opt/wsp/runs/pilot-14/submission
DRY=""; [ "$1" = --dry-run ] && DRY=1
log() { echo "$(date -u '+%F %T') eval-rft-arms: $*"; }

ready() {  # <arm> <base>: its run logged done, and each of its checkpoints has a check paired with the base
  WSP_TRAIN=$T $PY - "$@" <<'EOF'
import glob, json, os, re, sys
arm, base = sys.argv[1:]
T = os.environ["WSP_TRAIN"]
try:
    done = any(re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d done", l.rstrip("\n")) for l in open(f"{T}/runs/{arm}/train.log"))
except OSError:
    done = False
def paired(m):
    try:
        return json.load(open(f"{T}/evals/{arm}/{m.split('/')[-3]}/summary.json"))["vs_base"]["base"] == base
    except (OSError, ValueError, KeyError, TypeError):
        return False
ms = glob.glob(f"{T}/runs/{arm}/round_*/checkpoint/manifest.json")
sys.exit(0 if done and ms and all(paired(m) for m in ms) else 1)
EOF
}

choose() {  # <arm> <base>: writes evals/<arm>/choice.json (printed instead with --dry-run); prints the chosen folder
  WSP_TRAIN=$T DRY=$DRY $PY - "$@" <<'EOF'
import glob, json, os, sys, time
arm, base = sys.argv[1:]
T = os.environ["WSP_TRAIN"]
ms = sorted(glob.glob(f"{T}/runs/{arm}/round_*/checkpoint/manifest.json"), key=lambda m: int(m.split("/")[-3][6:]))
cands = []
for m in ms:
    name = m.split("/")[-3]
    s = json.load(open(f"{T}/evals/{arm}/{name}/summary.json"))
    v = s["vs_base"]
    cands.append({"checkpoint": name, "folder": os.path.dirname(m), "validation_pass": s["val"]["pass"],
                  "validation_diff": v["val"]["diff"], "validation_se": v["val"]["se"], "panel_mean": s["panel_mean"],
                  "panel_mean_diff": v["panel_mean"]["diff"], "panel_mean_se": v["panel_mean"]["se"],
                  "panel_within_2_5": v["panel_within_2_5"]})
# max() keeps the first of equal values, and the candidates are in round order: a tie goes to the earlier checkpoint.
best = max((c for c in cands if c["panel_within_2_5"]), key=lambda c: c["validation_pass"], default=None)
choice = {"arm": arm, "base": base, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
          "rule": "the best validation pass rate among the checkpoints whose panel mean is no more than 2.5 points "
                  "below the base's; a tie goes to the earlier checkpoint (plans/training-phase.md)",
          "candidates": cands, "chosen": best["checkpoint"] if best else None}
if os.environ.get("DRY"):
    print(json.dumps(choice), file=sys.stderr)
else:
    json.dump(choice, open(f"{T}/evals/{arm}/choice.json", "w"), indent=1)
print(best["folder"] if best else "")
EOF
}

evaluate() {  # <name> <submission> <evaluate.py options>: one held-out evaluation on the 5090, tried twice
  local name=$1 sub=$2; shift 2
  if [ -n "$DRY" ]; then log "would run: evaluate.py run $name --submission $sub $* --gpu 5090"; return 0; fi
  for attempt in 1 2; do
    log "$name: attempt $attempt, on $sub"
    if (cd $EV && $PY evaluate.py run "$name" --submission "$sub" "$@" --gpu 5090 >> "$EV/$name.out" 2>&1 < /dev/null); then
      log "$name: done (eval/results/$name/report.md)"; return 0
    fi
    log "$name: failed; last line: $(tail -1 "$EV/$name.out" | cut -c1-200)"
    sleep 60
  done
  return 1
}

log "start${DRY:+ (dry run)}"
if [ -z "$DRY" ]; then
  for i in $(seq 36); do  # up to 3 hours, for checks still running on the 5080
    ready R-E start-E && ready R-O start-O && break
    sleep 300
  done
fi
CHOSEN=()
for pair in R-E:start-E R-O:start-O; do
  arm=${pair%%:*}; base=${pair#*:}
  if ! ready $arm $base; then
    log "$arm: left out: its run has not logged done, or not all its checkpoints are checked and paired with $base"
    continue
  fi
  ckpt=$(choose $arm $base)
  if [ -z "$ckpt" ]; then
    log "$arm: no checkpoint's panel mean is within 2.5 points of $base's, so it has no held-out evaluation"
    continue
  fi
  log "$arm: $(basename "$(dirname "$ckpt")") chosen by the rule (evals/$arm/choice.json)"
  CHOSEN+=("$arm=$ckpt")
done
for c in "${CHOSEN[@]}"; do
  evaluate "${c%%=*}-pp15" "${c#*=}" --suites all --presence-penalty 1.5 --reuse-original pilot-10-pp15
done
for c in "${CHOSEN[@]}"; do
  evaluate "${c%%=*}-code-pp0" "${c#*=}" --suites code --reuse-original pilot-10
done
evaluate pilot-14-code-pp0 $E --suites code --reuse-original pilot-10
log "end"
