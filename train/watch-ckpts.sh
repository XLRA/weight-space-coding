#!/bin/bash
# Checks every checkpoint of a training run as it appears (plans/training-phase.md, "Choosing the checkpoint"), on the
# 5080 by default, until the run has logged "done" and every checkpoint is checked.
#   bash /opt/wsp/train/watch-ckpts.sh <arm> <base name> [gpu]
# e.g. watch-ckpts.sh G-E start-E      (results: train/evals/G-E/step_<k>/, paired with train/evals/start-E/)
# GRPO checkpoints are runs/<arm>/checkpoints/step_<k>/, RFT ones runs/<arm>/round_<r>/checkpoint/.
ARM=$1; BASE=$2; GPU=${3:-5080}
RUN=/opt/wsp/train/runs/$ARM
LOG=/opt/wsp/train/evals/$ARM/watch.log
mkdir -p /opt/wsp/train/evals/$ARM
echo "$(date -u '+%F %T') watching $RUN" >> "$LOG"
while true; do
  done_before=$(grep -c ' done$' "$RUN/train.log" 2>/dev/null)
  for m in $(ls -d "$RUN"/checkpoints/step_*/manifest.json "$RUN"/round_*/checkpoint/manifest.json 2>/dev/null | sort -V); do
    d=$(dirname "$m")
    case "$d" in */checkpoints/*) name=$(basename "$d");; *) name=$(basename "$(dirname "$d")");; esac
    [ -s "/opt/wsp/train/evals/$ARM/$name/summary.json" ] && continue
    echo "$(date -u '+%F %T') checking $name" >> "$LOG"
    if bash /opt/wsp/train/run-ckpt.sh "$GPU" "$ARM/$name" "$d" "$BASE" >> "$LOG" 2>&1; then
      echo "$(date -u '+%F %T') checked $name" >> "$LOG"
    else
      echo "$(date -u '+%F %T') check of $name failed; see evals/$ARM/$name/log" >> "$LOG"; exit 1
    fi
  done
  [ "${done_before:-0}" -gt 0 ] && break
  sleep 300
done
echo "$(date -u '+%F %T') all checkpoints checked" >> "$LOG"
