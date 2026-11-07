#!/bin/bash
# Waits for pilot-13 (on the 5080) to end, then, if it finished with an edit that differs from pilot-10's,
# evaluates it on all suites at presence penalty 1.5 on the 5080, reusing the unedited model's answers from
# results/pilot-10-pp15. Log: /opt/wsp/runs/after-pilot-11.log (shared with the other hand-offs)
log() { echo "$(date -u '+%F %T') [after-pilot-13] $*" >> /opt/wsp/runs/after-pilot-11.log; }
log "waiting for pilot-13"
while pgrep -f "[d]river.py pilot-13 " > /dev/null; do sleep 60; done
st=/opt/wsp/runs/pilot-13/state.json
if [ ! -f $st ]; then log "pilot-13 never started; nothing to evaluate"; exit 0; fi
status=$(/opt/wsp/venv/bin/python -c "import json; print(json.load(open('$st'))['status'])")
log "pilot-13 ended: $status"
if [ "$status" != "done" ]; then log "not done; not evaluated"; exit 1; fi
sums() { (cd "$1" && md5sum *.npy 2>/dev/null); }
if [ "$(sums /opt/wsp/runs/pilot-10/submission)" = "$(sums /opt/wsp/runs/pilot-13/submission)" ]; then
  log "pilot-13's edit is identical to pilot-10's; not evaluated"; exit 0
fi
log "evaluating pilot-13 on the 5080 (pilot-13-pp15)"
cd /opt/wsp/eval && /opt/wsp/venv/bin/python evaluate.py run pilot-13-pp15 \
  --submission /opt/wsp/runs/pilot-13/submission --suites all --presence-penalty 1.5 \
  --reuse-original pilot-10-pp15 --gpu 5080 > /opt/wsp/eval/pilot-13-pp15.out 2>&1 < /dev/null
log "evaluation pilot-13-pp15 ended with exit code $?"
