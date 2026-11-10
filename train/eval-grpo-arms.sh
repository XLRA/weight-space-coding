#!/bin/bash
# The GRPO arms' held-out evaluations, T's, and the removal's check (plans/training-phase.md, "Choosing the checkpoint"
# and "Measurement"; Deviations, 2026-11-10). Queued 2026-11-10 to run when G-E2 has ended: train/queue/after-G-E2.sh
# calls it, and grpo-queue-2026-11-08.sh appends its output to the queue's log.
#   bash /opt/wsp/train/eval-grpo-arms.sh [--dry-run]
# The evaluations run on the 5090 in this order, reusing O's answers (pilot-10-pp15 at presence penalty 1.5, pilot-10
# at 0):
# 1. G-O, on the checkpoint the rule chooses (train/arm_choice.py; evals/G-O/choice.json): the best val2 among the
#    checkpoints whose panel mean is no more than 2.5 points below start-O's, a tie going to the earlier one. The five
#    suites at presence penalty 1.5 (results/G-O-pp15), then the code suite at 0 (results/G-O-code-pp0).
# 2. G-E2, once its run has ended and its checkpoints are checked (it waits up to 6 hours): its checkpoint by the same
#    rule against start-E, then G-E2-pp15 and G-E2-code-pp0. Meanwhile the 5080 runs the removal: that checkpoint − ΔE
#    (train/combine.py; train/combined/G-E2-minus-E-<step>), checked and paired with the checkpoint
#    (evals/G-E2-minus-E/<step>/). If no checkpoint qualifies, G-E2 gets no held-out evaluation, and the removal uses
#    its last checked checkpoint.
# 3. T = G-O's checkpoint + ΔE (train/combined/T-<step>; checked on the 5080 and paired with that checkpoint, unless
#    already done): the five suites at 1.5 (results/T-pp15), unless its panel is clearly worse than the checkpoint's:
#    a panel mean more than 2 SE below, paired (arm_choice.py panel-ok).
# Each evaluation is tried twice (evaluate.py resumes where it stopped). --dry-run: no waiting, building, checks,
# choice.json or evaluations; it prints the choices and the commands. WSP_TRAIN points it at another train/ folder
# (for tests).
T=${WSP_TRAIN:-/opt/wsp/train}
CODE=$(dirname "$(readlink -f "$0")")
EV=/opt/wsp/eval
PY=/opt/wsp/venv/bin/python
DRY=""; [ "$1" = --dry-run ] && DRY=1
log() { echo "$(date -u '+%F %T') eval-grpo-arms: $*"; }
ac() { WSP_TRAIN=$T $PY "$CODE/arm_choice.py" "$@" ${DRY:+--dry-run}; }

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

both() {  # <arm> <checkpoint folder>: a trained arm's two evaluations
  evaluate "$1-pp15" "$2" --suites all --presence-penalty 1.5 --reuse-original pilot-10-pp15
  evaluate "$1-code-pp0" "$2" --suites code --reuse-original pilot-10
}

combined() {  # <name> <checkpoint folder> <+|->: train/combined/<name>-<step>, built unless it exists; prints its folder
  local out; out=$T/combined/$1-$(basename "$2")
  if [ ! -d "$out" ]; then
    if [ -n "$DRY" ]; then
      log "would run: combine.py $out $2 $3" >&2
    else
      log "building $out ($2 $3 ΔE)" >&2
      mkdir -p "$T/combined" && $PY "$CODE/combine.py" "$out" "$2" "$3" >&2 || return 1
    fi
  fi
  echo "$out"
}

check() {  # <check name> <folder> <base>: checked on the 5080 and paired with the base, unless already done
  [ -s "$T/evals/$1/summary.json" ] && return 0
  if [ -n "$DRY" ]; then log "would run: run-ckpt.sh 5080 $1 $2 $3"; return 0; fi
  log "$1: checking on the 5080, paired with $3"
  mkdir -p "$T/evals/${1%%/*}"
  bash "$CODE/run-ckpt.sh" 5080 "$1" "$2" "$3" >> "$T/evals/${1%%/*}/check.log" 2>&1 < /dev/null
}

removal() {  # <G-E2 checkpoint folder>: the checkpoint − ΔE, checked on the 5080 and paired with the checkpoint
  local step rf; step=$(basename "$1")
  rf=$(combined G-E2-minus-E "$1" -) || { log "removal: combine.py failed, so there is no removal check"; return 1; }
  check "G-E2-minus-E/$step" "$rf" "G-E2/$step" || { log "removal: the check failed (evals/G-E2-minus-E/$step/log)"; return 1; }
  [ -n "$DRY" ] || log "removal: $(tail -1 "$T/evals/G-E2-minus-E/check.log")"
}

log "start${DRY:+ (dry run)}"

# 1. G-O
GO=""
if ac ready G-O start-O; then
  GO=$(ac choose G-O start-O)
  if [ -n "$GO" ]; then
    log "G-O: $(basename "$GO") chosen by the rule (evals/G-O/choice.json)"
    both G-O "$GO"
  else
    log "G-O: no checkpoint's panel mean is within 2.5 points of start-O's, so it has no held-out evaluation"
  fi
else
  log "G-O: left out: its run has not ended, or not all its checkpoints are checked and paired with start-O"
fi

# 2. G-E2, with the removal on the 5080
if [ -z "$DRY" ]; then
  for i in $(seq 72); do  # up to 6 hours, for checks still running on the 5080
    ac ready G-E2 start-E && break
    ac ended G-E2 && ! pgrep -f "[w]atch-ckpts.sh G-E2 " > /dev/null && break  # its watcher gave up on a checkpoint
    sleep 300
  done
fi
RPID=""; R=""
if ! ac ended G-E2 || [ -z "$(ac last G-E2)" ]; then
  log "G-E2: left out: its run has not ended, or it has no checkpoint"
else
  ac ready G-E2 start-E ||
    log "G-E2: not every checkpoint has a check paired with start-E; the rule leaves those out (evals/G-E2/choice.json)"
  GE=$(ac choose G-E2 start-E)
  if [ -n "$GE" ]; then
    log "G-E2: $(basename "$GE") chosen by the rule (evals/G-E2/choice.json)"; R=$GE
  else
    R=$(ac last G-E2 start-E)
    if [ -n "$R" ]; then
      log "G-E2: no checkpoint's panel mean is within 2.5 points of start-E's, so it has no held-out evaluation; the removal uses its last checked checkpoint, $(basename "$R")"
    else
      log "G-E2: no checkpoint has a check paired with start-E, so it has no held-out evaluation and no removal check"
    fi
  fi
  if [ -n "$R" ]; then
    if [ -n "$DRY" ]; then removal "$R"; else removal "$R" & RPID=$!; fi
  fi
  [ -n "$GE" ] && both G-E2 "$GE"
fi

# 3. T
if [ -n "$GO" ]; then
  step=$(basename "$GO")
  TF=$(combined T "$GO" +) && check "T/$step" "$TF" "G-O/$step"
  if [ -n "$DRY" ] && [ ! -s "$T/evals/T/$step/summary.json" ]; then
    log "T: would decide by its panel once its check is done"
  else
    v=$(ac panel-ok "T/$step"); rc=$?
    case $rc in
      0) log "T: its panel is not clearly worse than G-O/$step's ($v), so it gets its held-out evaluation"
         evaluate T-pp15 "$TF" --suites all --presence-penalty 1.5 --reuse-original pilot-10-pp15;;
      1) log "T: its panel is clearly worse than G-O/$step's ($v: more than 2 SE below), so it gets no held-out evaluation";;
      *) log "T: its check has no summary (evals/T/$step/log), so it gets no held-out evaluation";;
    esac
  fi
fi

if [ -n "$RPID" ]; then
  wait $RPID
fi
log "end"
