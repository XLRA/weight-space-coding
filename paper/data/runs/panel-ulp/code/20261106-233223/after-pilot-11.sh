#!/bin/bash
# Waits for pilot-11 to end, then: if its edit is identical to pilot-10's, skips its evaluation (it would repeat
# results/pilot-10-pp15) and starts pilot-12 from pilot-10's edit; if the edit changed, evaluates it at presence
# penalty 1.5 instead and leaves pilot-12 for a person to start. Log: /opt/wsp/runs/after-pilot-11.log
log() { echo "$(date -u '+%F %T') $*" >> /opt/wsp/runs/after-pilot-11.log; }
log "waiting for pilot-11"
while pgrep -f "[d]river.py pilot-11 " > /dev/null; do sleep 60; done
status=$(/opt/wsp/venv/bin/python -c "import json; print(json.load(open('/opt/wsp/runs/pilot-11/state.json'))['status'])")
log "pilot-11 ended: $status"
if [ "$status" != "done" ]; then log "not done; nothing started"; exit 1; fi
sums() { (cd "$1" && md5sum *.npy 2>/dev/null); }
a=$(sums /opt/wsp/runs/pilot-10/submission); b=$(sums /opt/wsp/runs/pilot-11/submission)
if [ -n "$b" ] && [ "$a" = "$b" ]; then
  log "pilot-11's edit is identical to pilot-10's; evaluation skipped; starting pilot-12"
  sleep 30  # let llama-swap come back up and the GPU lock be released
  cd /opt/wsp/harness && exec /opt/wsp/venv/bin/python driver.py pilot-12 --passes 12 \
    --start-from /opt/wsp/runs/pilot-10/submission --brief briefs/pilot-12.md --presence-penalty 1.5 \
    > /opt/wsp/runs/pilot-12.out 2>&1 < /dev/null
else
  log "pilot-11's edit differs from pilot-10's; evaluating it (pilot-11-pp15); pilot-12 not started"
  sleep 30
  cd /opt/wsp/eval && exec /opt/wsp/venv/bin/python evaluate.py run pilot-11-pp15 \
    --submission /opt/wsp/runs/pilot-11/submission --suites all --presence-penalty 1.5 \
    --reuse-original pilot-10-pp15 > /opt/wsp/eval/pilot-11-pp15.out 2>&1 < /dev/null
fi
