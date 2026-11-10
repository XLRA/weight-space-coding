#!/bin/bash
# Activation probes for one model (plans/training-phase.md, "Do the edits survive training?"; train/probes.py run).
#   bash /opt/wsp/train/run-probes.sh <5080|5090> <name> <weights folder on the host | original>
# e.g. run-probes.sh 5080 G-E2/step_50 /opt/wsp/train/runs/G-E2/checkpoints/step_50
# Holds the GPU's lock while it runs, so it waits for a check or an evaluation that has it. On the 5090, llama-swap
# is stopped meanwhile if it is running, as run-ckpt.sh does. Skipped when the model's result exists.
# Output: train/analysis/probes/<name, with / as ->.json and .npz; log: train/analysis/probes/log.
set -e
GPU=$1; NAME=$2; SRC=$3
G() { (cd /opt/wsp/harness && python3 -c "from gpus import GPUS; print(GPUS['$GPU']['$1'])"); }
DEV=$(G device); LOCK=/opt/wsp/runs/$(G lock)
OUT=/opt/wsp/train/analysis/probes
FILE=$(echo "$NAME" | tr '/' '-')
LOG=$OUT/log
[ -s "$OUT/texts.jsonl" ] || { echo "no $OUT/texts.jsonl: run probes.py build first" >&2; exit 2; }
if [ -s "$OUT/$FILE.json" ]; then echo "$NAME: probed already"; exit 0; fi
MOUNT=(); ARG=original
if [ "$SRC" != original ]; then
  [ -d "$SRC" ] || { echo "no folder $SRC" >&2; exit 2; }
  MOUNT=(-v "$SRC:/ckpt:ro"); ARG=/ckpt
fi

exec 9>>"$LOCK"
echo "$(date -u '+%F %T') $NAME: waiting for the $GPU" >> "$LOG"
flock 9
STOPPED=0; RC=0
if [ "$(G llama_swap)" = True ] && [ -n "$(docker ps -q --filter name=^llama-swap$)" ]; then
  echo "$(date -u '+%F %T') stopping llama-swap to free the $GPU" >> "$LOG"
  docker stop llama-swap > /dev/null; STOPPED=1
fi
echo "$(date -u '+%F %T') $NAME: probing $SRC" >> "$LOG"
docker run --rm --name "wsp-probes-$FILE" --network none --device "$DEV" --shm-size 4g -e PYTHONPATH=/train \
  -v /opt/wsp/models/qwen3.5-0.8b:/model:ro -v /opt/wsp/runs/pilot-14/submission:/edit:ro \
  -v /opt/wsp/train:/train "${MOUNT[@]}" wsp-train:1 \
  python3 /train/probes.py run "$NAME" "$ARG" "$SRC" >> "$LOG" 2>&1 || RC=$?
if [ $STOPPED = 1 ]; then docker start llama-swap > /dev/null && echo "$(date -u '+%F %T') llama-swap started again" >> "$LOG"; fi
flock -u 9; exec 9>&-
[ $RC = 0 ] || { echo "$NAME: probes failed ($RC), see $LOG" >&2; exit $RC; }
grep "^$NAME: " "$LOG" | tail -1 | cut -c1-200
