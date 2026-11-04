#!/bin/bash
# Starts pilot-13 on the 5080 and its evaluation hand-off.
cd /opt/wsp/harness
nohup setsid /opt/wsp/venv/bin/python driver.py pilot-13 --passes 20 --start-from /opt/wsp/runs/pilot-10/submission \
  --brief briefs/pilot-13.md --presence-penalty 1.5 --gpu 5080 --kernel-memory-gb 14 \
  > /opt/wsp/runs/pilot-13.out 2>&1 < /dev/null &
sleep 3
nohup setsid /opt/wsp/harness/after-pilot-13.sh > /dev/null 2>&1 < /dev/null &
sleep 1
echo launched
