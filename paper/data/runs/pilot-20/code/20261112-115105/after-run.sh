#!/bin/bash
# Usage: after-run.sh <run> <gpu: 5090|5080|any> <start submission dir>
# Waits for runs/<run> to end. If it finished with an edit that differs from <start submission dir>, evaluates
# it on all suites at presence penalty 1.5 on <gpu> (results/<run>-pp15, output in eval/<run>-pp15.out), reusing
# the unedited model's answers from results/pilot-10-pp15. Log: /opt/wsp/runs/after-pilot-11.log, shared with
# the other hand-offs. Start it right after the run's driver. With "any" (for a run on Colab), the evaluation
# waits for the first local GPU whose lock is free, and waits again if another job takes it first.
run=$1; gpu=$2; start=$3
log() { echo "$(date -u '+%F %T') [after-$run] $*" >> /opt/wsp/runs/after-pilot-11.log; }
if [ -z "$run" ] || [ -z "$gpu" ] || [ ! -d "$start" ]; then echo "usage: after-run.sh <run> <gpu> <start dir>"; exit 2; fi
trap '/opt/wsp/venv/bin/python /opt/wsp/paper/collect.py > /dev/null 2>&1' EXIT  # refresh the paper tables (README, "Paper data")
if [ "$gpu" = any ]; then where="the first free local GPU"; else where="the $gpu"; fi
log "waiting for $run (evaluation on $where when it ends)"
sleep 30
while pgrep -f "[d]river.py $run " > /dev/null; do sleep 60; done
st=/opt/wsp/runs/$run/state.json
if [ ! -f $st ]; then log "$run never started; nothing to evaluate"; exit 0; fi
status=$(/opt/wsp/venv/bin/python -c "import json; print(json.load(open('$st'))['status'])")
log "$run ended: $status"
if [ "$status" != "done" ]; then log "not done; not evaluated"; exit 1; fi
sums() { (cd "$1" && md5sum *.npy 2>/dev/null); }
if [ "$(sums "$start")" = "$(sums /opt/wsp/runs/$run/submission)" ]; then
  log "$run's edit is identical to its starting edit; not evaluated"; exit 0
fi
evaluate() {
  log "evaluating $run on the $1 ($run-pp15)"
  cd /opt/wsp/eval && /opt/wsp/venv/bin/python evaluate.py run $run-pp15 \
    --submission /opt/wsp/runs/$run/submission --suites all --presence-penalty 1.5 \
    --reuse-original pilot-10-pp15 --gpu $1 > /opt/wsp/eval/$run-pp15.out 2>&1 < /dev/null
}
free_gpu() {  # the first local GPU whose lock nobody holds
  for g in 5090:.gpu.lock 5080:.gpu-5080.lock; do
    if flock -n /opt/wsp/runs/${g#*:} true 2>/dev/null; then echo ${g%%:*}; return; fi
  done
}
if [ "$gpu" != any ]; then
  evaluate $gpu
  log "evaluation $run-pp15 ended with exit code $?"
  exit
fi
log "waiting for a free local GPU"
while true; do
  g=$(free_gpu)
  if [ -z "$g" ]; then sleep 120; continue; fi
  evaluate $g
  code=$?
  if [ $code -ne 0 ] && grep -q "another run or evaluation holds" /opt/wsp/eval/$run-pp15.out; then
    log "the $g was taken first; waiting again"; sleep 120; continue
  fi
  log "evaluation $run-pp15 on the $g ended with exit code $code"
  exit
done
