#!/bin/bash
# Starts the GPU temperature logger (gpu_temps.py) in the background if it isn't running already.
# Log: /opt/wsp/runs/gpu-temps.csv; warnings: /opt/wsp/runs/gpu-temps-warnings.log. Run it again after a reboot.
if pgrep -f "/opt/wsp/harness/gpu_temps[.]py" > /dev/null; then echo "already running"; exit 0; fi
nohup setsid /opt/wsp/venv/bin/python /opt/wsp/harness/gpu_temps.py > /opt/wsp/runs/gpu-temps.out 2>&1 < /dev/null &
sleep 2
pgrep -f "/opt/wsp/harness/gpu_temps[.]py" > /dev/null && echo "started" || { echo "failed:"; cat /opt/wsp/runs/gpu-temps.out; }
