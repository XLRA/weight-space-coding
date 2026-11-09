#!/bin/bash
# Checks one checkpoint (plans/training-phase.md, "Choosing the checkpoint"): the panel and the validation problems.
#   bash /opt/wsp/train/run-ckpt.sh <5080|5090> <name> <submission folder on the host | original> [<base name>]
# e.g. run-ckpt.sh 5080 G-E/step_25 /opt/wsp/train/runs/G-E/checkpoints/step_25 start-E
# 1. train/ckpt_eval.py generates the answers in wsp-train:1, holding the GPU's lock (waits for it if busy).
# 2. The panel is graded in the grading container exactly as harness/driver.py grade_panel does.
# 3. train/ckpt_grade.py grades the validation answers (no GPU) and writes summary.json, paired with <base name>.
# Results and log: train/evals/<name>/. Each stage is skipped if its output exists, so a rerun resumes.
# On the 5090, llama-swap is stopped while generating if it is running, and started again afterwards.
# The graded code runs as user 65533, apart from the GRPO trainer's 65534: a user's process limit counts across
# containers, and a check grading as 65534 crashed G-O's grader on 2026-11-09 (harness/grade.py, RUN_AS).
set -e
GPU=$1; NAME=$2; SRC=$3; BASE=$4
G() { (cd /opt/wsp/harness && python3 -c "from gpus import GPUS; print(GPUS['$GPU']['$1'])"); }
DEV=$(G device); LOCK=/opt/wsp/runs/$(G lock)
MEM=$( [ "$GPU" = 5080 ] && echo 0.65 || echo 0.85 )
OUT=/opt/wsp/train/evals/$NAME
LOG=$OUT/log
TAG=$(echo "$NAME" | tr '/' '-')
mkdir -p "$OUT"
MOUNT=(); ARG=original
if [ "$SRC" != original ]; then MOUNT=(-v "$SRC:/ckpt:ro"); ARG=/ckpt; fi
[ -z "$BASE" ] || [ -s "/opt/wsp/train/evals/$BASE/summary.json" ] || { echo "base $BASE has no summary.json" >&2; exit 2; }

if [ ! -s "$OUT/panel.raw.jsonl" ]; then
  exec 9>>"$LOCK"
  echo "$(date -u '+%F %T') waiting for the $GPU" >> "$LOG"
  flock 9
  STOPPED=0; RC=0
  if [ "$(G llama_swap)" = True ] && [ -n "$(docker ps -q --filter name=^llama-swap$)" ]; then
    echo "$(date -u '+%F %T') stopping llama-swap to free the $GPU" >> "$LOG"
    docker stop llama-swap > /dev/null; STOPPED=1
  fi
  docker run --rm --name "wsp-ckpt-$TAG" --network none --device "$DEV" --shm-size 8g \
    -e WSP_GPU_MEMORY=$MEM -e PYTHONPATH=/harness:/train \
    -v /opt/wsp/models/qwen3.5-0.8b:/model:ro -v /opt/wsp/harness:/harness:ro \
    -v /opt/wsp/panel/final/panel.jsonl:/panel/panel.jsonl:ro -v /opt/wsp/train:/train "${MOUNT[@]}" wsp-train:1 \
    python3 /train/ckpt_eval.py "$NAME" "$ARG" "$SRC" >> "$LOG" 2>&1 || RC=$?
  if [ $STOPPED = 1 ]; then docker start llama-swap > /dev/null && echo "$(date -u '+%F %T') llama-swap started again" >> "$LOG"; fi
  flock -u 9; exec 9>&-
  [ $RC = 0 ] || exit $RC
fi

if [ ! -s "$OUT/panel.jsonl" ]; then
  echo "$(date -u '+%F %T') grading the panel" >> "$LOG"
  docker run --rm --network none --read-only --tmpfs /tmp:rw,size=2g,mode=1777 \
    --user 65533:65533 --cap-drop ALL --security-opt no-new-privileges \
    --memory 16g --pids-limit 512 --cpus 12 -w /tmp \
    -e PYTHONPATH=/harness:/evalbin -v /opt/wsp/panel/final/panel.jsonl:/panel/panel.jsonl:ro \
    -v /opt/wsp/harness:/harness:ro -v /opt/wsp/eval/suites.py:/evalbin/suites.py:ro \
    -v /opt/wsp/eval/ifeval:/evalbin/ifeval:ro -v "$OUT":/work:ro \
    wsp-grade:1 python3 -W ignore /harness/grade_panel.py /work/panel.raw.jsonl > "$OUT/panel.jsonl.tmp" 2>> "$LOG"
  [ "$(wc -l < "$OUT/panel.jsonl.tmp")" = "$(wc -l < "$OUT/panel.raw.jsonl")" ] || { echo "panel grading incomplete" >> "$LOG"; exit 1; }
  mv "$OUT/panel.jsonl.tmp" "$OUT/panel.jsonl"
fi

echo "$(date -u '+%F %T') grading the validation answers" >> "$LOG"
docker run --rm --name "wsp-ckpt-grade-$TAG" --network none --pids-limit 1024 --cpus 12 \
  -e PYTHONPATH=/harness:/train -e WSP_GRADE_UID=65533 -v /opt/wsp/models/qwen3.5-0.8b:/model:ro -v /opt/wsp/harness:/harness:ro \
  -v /opt/wsp/train:/train wsp-train:1 python3 /train/ckpt_grade.py "$NAME" $BASE >> "$LOG" 2>&1
tail -1 "$LOG"
