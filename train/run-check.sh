#!/bin/bash
# Runs a pre-training check in wsp-train:1 on one local GPU, holding that GPU's lock (waits for it if busy).
#   bash /opt/wsp/train/run-check.sh <5080|5090> [logprobs|push] [start submission]
#   logprobs (the default): train/check_logprobs.py, log and results train/checks/logprobs-<gpu>.{log,json}
#   push: train/check_push.py, log and results train/checks/push-<gpu>.{log,json}
# No network; the model, the harness (for wsp_ext.py) and the grader are mounted read-only; train/ is writable.
set -e
GPU=$1; shift
CHECK=logprobs
if [ "$1" = logprobs ] || [ "$1" = push ]; then CHECK=$1; shift; fi
START=$1
DEV=$(cd /opt/wsp/harness && python3 -c "from gpus import GPUS; print(GPUS['$GPU']['device'])")
LOCK=/opt/wsp/runs/$(cd /opt/wsp/harness && python3 -c "from gpus import GPUS; print(GPUS['$GPU']['lock'])")
MEM=$( [ "$GPU" = 5080 ] && echo 0.40 || echo 0.30 )
NAME=wsp-train-check-$GPU; [ "$CHECK" = push ] && NAME=wsp-train-check-push-$GPU
mkdir -p /opt/wsp/train/checks
MOUNT_START=(); ARG_START=()
if [ -n "$START" ]; then MOUNT_START=(-v "$START:/start:ro"); ARG_START=(/start); fi
exec flock "$LOCK" docker run --rm --name "$NAME" --network none --device "$DEV" --shm-size 8g \
  -e WSP_GPU_MEMORY=$MEM -e PYTHONPATH=/harness:/train \
  -v /opt/wsp/models/qwen3.5-0.8b:/model:ro -v /opt/wsp/harness:/harness:ro -v /opt/wsp/train:/train \
  "${MOUNT_START[@]}" wsp-train:1 python3 "/train/check_$CHECK.py" "$GPU" "${ARG_START[@]}" \
  > "/opt/wsp/train/checks/$CHECK-$GPU.log" 2>&1
