#!/bin/bash
# Usage: after-run.sh <run> <gpu: 5090|5080> <start submission dir>
# Waits for runs/<run> to end. If it finished with an edit that differs from <start submission dir>, evaluates
# it on all suites at presence penalty 1.5 on <gpu> (results/<run>-pp15, output in eval/<run>-pp15.out), reusing
# the unedited model's answers from results/pilot-10-pp15. Log: /opt/wsp/runs/after-pilot-11.log, shared with
# the other hand-offs. Start it right after the run's driver.
run=$1; gpu=$2; start=$3
log() { echo "$(date -u '+%F %T') [after-$run] $*" >> /opt/wsp/runs/after-pilot-11.log; }
if [ -z "$run" ] || [ -z "$gpu" ] || [ ! -d "$start" ]; then echo "usage: after-run.sh <run> <gpu> <start dir>"; exit 2; fi
log "waiting for $run (evaluation on the $gpu when it ends)"
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
log "evaluating $run on the $gpu ($run-pp15)"
cd /opt/wsp/eval && /opt/wsp/venv/bin/python evaluate.py run $run-pp15 \
  --submission /opt/wsp/runs/$run/submission --suites all --presence-penalty 1.5 \
  --reuse-original pilot-10-pp15 --gpu $gpu > /opt/wsp/eval/$run-pp15.out 2>&1 < /dev/null
log "evaluation $run-pp15 ended with exit code $?"
