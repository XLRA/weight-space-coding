#!/bin/bash
# Starts pilot-12 with 20 passes and re-arms its evaluation hand-off.
pkill -f "/opt/wsp/harness/after-pilot-12[.]sh"
sleep 1
echo "$(date -u '+%F %T') hand-off stopped by hand before launch; pilot-12 started by hand with --passes 20 (Max 5x plan)" >> /opt/wsp/runs/after-pilot-11.log
cd /opt/wsp/harness
nohup setsid /opt/wsp/venv/bin/python driver.py pilot-12 --passes 20 --start-from /opt/wsp/runs/pilot-10/submission \
  --brief briefs/pilot-12.md --presence-penalty 1.5 > /opt/wsp/runs/pilot-12.out 2>&1 < /dev/null &
sleep 5
nohup setsid /opt/wsp/harness/after-pilot-12.sh > /dev/null 2>&1 < /dev/null &
sleep 1
echo launched
