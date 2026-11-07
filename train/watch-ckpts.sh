#!/bin/bash
# Checks every checkpoint of a training run as it appears (plans/training-phase.md, "Choosing the checkpoint"), on the
# 5080 by default, until the run has ended (logged "done", or has a "stopped" file saying why) and every checkpoint is
# checked. A failed check is tried again on the next pass, 5 minutes later, and given up after its second failure.
#   bash /opt/wsp/train/watch-ckpts.sh <arm> <base name> [gpu]
# e.g. watch-ckpts.sh G-E start-E      (results: train/evals/G-E/step_<k>/, paired with train/evals/start-E/)
# GRPO checkpoints are runs/<arm>/checkpoints/step_<k>/, RFT ones runs/<arm>/round_<r>/checkpoint/.
ARM=$1; BASE=$2; GPU=${3:-5080}
RUN=/opt/wsp/train/runs/$ARM
LOG=/opt/wsp/train/evals/$ARM/watch.log
mkdir -p /opt/wsp/train/evals/$ARM
echo "$(date -u '+%F %T') watching $RUN" >> "$LOG"
declare -A FAILS
while true; do
  # Read before the scan: a run saves its last checkpoint before it logs "done" or is marked stopped.
  ended=$(grep -c -E '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8} done$' "$RUN/train.log" 2>/dev/null)
  [ -e "$RUN/stopped" ] && ended=1
  pending=0
  for m in $(ls -d "$RUN"/checkpoints/step_*/manifest.json "$RUN"/round_*/checkpoint/manifest.json 2>/dev/null | sort -V); do
    d=$(dirname "$m")
    case "$d" in */checkpoints/*) name=$(basename "$d");; *) name=$(basename "$(dirname "$d")");; esac
    [ -s "/opt/wsp/train/evals/$ARM/$name/summary.json" ] && continue
    [ "${FAILS[$name]:-0}" -ge 2 ] && continue
    echo "$(date -u '+%F %T') checking $name" >> "$LOG"
    if bash /opt/wsp/train/run-ckpt.sh "$GPU" "$ARM/$name" "$d" "$BASE" >> "$LOG" 2>&1; then
      echo "$(date -u '+%F %T') checked $name" >> "$LOG"
    else
      FAILS[$name]=$(( ${FAILS[$name]:-0} + 1 ))
      echo "$(date -u '+%F %T') check of $name failed (${FAILS[$name]} of 2 tries); see evals/$ARM/$name/log" >> "$LOG"
      [ "${FAILS[$name]}" -lt 2 ] && pending=$((pending + 1))
    fi
  done
  [ "${ended:-0}" -gt 0 ] && [ $pending = 0 ] && break
  sleep 300
done
gave_up=""
for n in "${!FAILS[@]}"; do [ "${FAILS[$n]}" -ge 2 ] && gave_up="$gave_up $n"; done
echo "$(date -u '+%F %T') all checkpoints checked${gave_up:+, except$gave_up (failed twice)}" >> "$LOG"
