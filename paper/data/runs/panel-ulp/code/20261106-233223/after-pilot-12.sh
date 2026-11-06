#!/bin/bash
# Waits for after-pilot-11.sh and pilot-12 to end, then, if pilot-12 finished with an edit that differs from
# pilot-10's, evaluates it on all suites at presence penalty 1.5, reusing the unedited model's answers from
# results/pilot-10-pp15. Log: /opt/wsp/runs/after-pilot-11.log (shared with after-pilot-11.sh)
log() { echo "$(date -u '+%F %T') [after-pilot-12] $*" >> /opt/wsp/runs/after-pilot-11.log; }
log "waiting for pilot-12"
while pgrep -f "[a]fter-pilot-11.sh" > /dev/null || pgrep -f "[d]river.py pilot-12 " > /dev/null; do sleep 60; done
st=/opt/wsp/runs/pilot-12/state.json
if [ ! -f $st ]; then log "pilot-12 never started; nothing to evaluate"; exit 0; fi
status=$(/opt/wsp/venv/bin/python -c "import json; print(json.load(open('$st'))['status'])")
log "pilot-12 ended: $status"
if [ "$status" != "done" ]; then log "not done; not evaluated"; exit 1; fi
sums() { (cd "$1" && md5sum *.npy 2>/dev/null); }
if [ "$(sums /opt/wsp/runs/pilot-10/submission)" = "$(sums /opt/wsp/runs/pilot-12/submission)" ]; then
  log "pilot-12's edit is identical to pilot-10's; not evaluated"; exit 0
fi
log "evaluating pilot-12 (pilot-12-pp15)"
sleep 30
cd /opt/wsp/eval && /opt/wsp/venv/bin/python evaluate.py run pilot-12-pp15 \
  --submission /opt/wsp/runs/pilot-12/submission --suites all --presence-penalty 1.5 \
  --reuse-original pilot-10-pp15 > /opt/wsp/eval/pilot-12-pp15.out 2>&1 < /dev/null
log "evaluation pilot-12-pp15 ended with exit code $?"
