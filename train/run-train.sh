#!/bin/bash
# Runs a training script (train/grpo.py or train/rft.py) in wsp-train:1 on one local GPU, holding that GPU's lock
# (waits for it if busy).
#   bash /opt/wsp/train/run-train.sh <5080|5090> <grpo|rft> <arm> [--start <submission folder on the host>] [options]
# Log: train/runs/<arm>/train.log. No network. The model, the harness (wsp_ext.py) and the grader are read-only.
# On the 5090, llama-swap is stopped for the run if it is running, and started again afterwards, as driver.py does.
set -e
GPU=$1; KIND=$2; ARM=$3; shift 3
case "$KIND" in grpo|rft) ;; *) echo "kind must be grpo or rft" >&2; exit 2;; esac
START=""; REST=()
while [ $# -gt 0 ]; do
  if [ "$1" = --start ]; then START=$2; shift 2; else REST+=("$1"); shift; fi
done
G() { (cd /opt/wsp/harness && python3 -c "from gpus import GPUS; print(GPUS['$GPU']['$1'])"); }
DEV=$(G device); LOCK=/opt/wsp/runs/$(G lock); SWAP=$(G llama_swap)
LOG=/opt/wsp/train/runs/$ARM/train.log
mkdir -p /opt/wsp/train/runs/$ARM
# One line per launch (paper/collect.py reads the arm's start from it: inside the container it is always /start).
printf '{"time": "%s", "gpu": "%s", "kind": "%s", "start": "%s", "args": "%s"}\n' "$(date -u '+%F %T')" "$GPU" \
  "$KIND" "${START:-original}" "${REST[*]}" >> /opt/wsp/train/runs/$ARM/launches.jsonl
MOUNT_START=(); ARG_START=()
if [ -n "$START" ]; then MOUNT_START=(-v "$START:/start:ro"); ARG_START=(--start /start); fi

exec 9>>"$LOCK"
echo "$(date -u '+%F %T') waiting for the $GPU" >> "$LOG"
flock 9
STOPPED=0
if [ "$SWAP" = True ] && [ -n "$(docker ps -q --filter name=^llama-swap$)" ]; then
  echo "$(date -u '+%F %T') stopping llama-swap to free the $GPU" >> "$LOG"
  docker stop llama-swap > /dev/null; STOPPED=1
fi
trap 'if [ $STOPPED = 1 ]; then docker start llama-swap > /dev/null && echo "$(date -u "+%F %T") llama-swap started again" >> "$LOG"; fi' EXIT
docker run --rm --name "wsp-train-$ARM" --network none --device "$DEV" --shm-size 16g \
  -e PYTHONPATH=/harness:/train \
  -v /opt/wsp/models/qwen3.5-0.8b:/model:ro -v /opt/wsp/harness:/harness:ro \
  -v /opt/wsp/train:/train "${MOUNT_START[@]}" wsp-train:1 \
  python3 "/train/$KIND.py" "$ARM" "${ARG_START[@]}" "${REST[@]}" >> "$LOG" 2>&1
