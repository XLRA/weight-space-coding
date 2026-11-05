#!/bin/bash
# Switch the running pilot-14 to 400k-token conversations (2026-11-05). Run on the server with nohup setsid.
# 1. Stops pilot-14's evaluation hand-off, so it can't see the driver gone and give up.
# 2. Waits for the next turn line in driver.log (not a final answer) and for the Claude call that follows it,
#    so no code block is running and no forward pass is in flight.
# 3. Kills the driver's process group with SIGKILL: the driver and its claude call end, while the kernel and
#    GPU server containers keep running.
# 4. Resumes with --handoff-every 0 --max-context 400000 (the new driver keeps that kernel and GPU server and
#    passes on the unanswered result), then restarts the hand-off.
set -u
R=/opt/wsp/runs/pilot-14
L=$R/driver.log
log() { echo "$(date -u '+%F %T') [restart-pilot-14] $*" >> /opt/wsp/runs/after-pilot-11.log; }
resume() {
  cd /opt/wsp/harness
  nohup setsid /opt/wsp/venv/bin/python driver.py pilot-14 --handoff-every 0 --max-context 400000 \
    >> /opt/wsp/runs/pilot-14.out 2>&1 < /dev/null &
  sleep 5
}
handoff() {
  nohup setsid bash /opt/wsp/harness/after-run.sh pilot-14 5090 /opt/wsp/runs/combo-12-13/submission \
    > /dev/null 2>&1 < /dev/null &
  log "evaluation hand-off restarted"
}
pkill -f "after-run.sh pilot-14 " && log "stopped the evaluation hand-off; waiting for a turn to end"
start=$(wc -l < $L)
deadline=$(( $(date +%s) + 3 * 3600 ))
while true; do
  pid=$(pgrep -f "[d]river.py pilot-14 " | head -1)
  if [ -z "$pid" ]; then log "pilot-14's driver ended by itself; not resumed"; handoff; exit 0; fi
  if [ "$(date +%s)" -gt "$deadline" ]; then log "no safe moment in 3 hours; left as it was"; handoff; exit 1; fi
  line=$(tail -n +$((start + 1)) $L | grep -E "^.{19} turn [0-9]+: " | grep -v "final answer" | tail -n 1)
  if [ -n "$line" ]; then
    pg=$(ps -o pgid= -p "$pid" | tr -d ' ')
    for i in $(seq 40); do  # up to 20 s for the Claude call that follows the turn line
      if [ "$(tail -n 1 $L)" != "$line" ]; then break; fi  # something else happened first (a request, a handoff)
      if pgrep -g "$pg" -f -- "-p --output-format json" > /dev/null; then
        if [ "$pg" != "$pid" ]; then log "driver $pid is not its group's leader ($pg); not touched"; handoff; exit 1; fi
        kill -KILL -- "-$pg"
        log "killed pilot-14's driver (pid $pid) after: ${line:20:60}"
        sleep 3
        pgrep -f "[d]river.py pilot-14 " > /dev/null && { log "the driver is still there; stopping"; exit 1; }
        resume
        if pgrep -f "[d]river.py pilot-14 " > /dev/null; then log "resumed pilot-14 with 400k-token conversations"
        else log "the resumed driver exited; see runs/pilot-14.out"; fi
        handoff
        exit 0
      fi
      sleep 0.5
    done
    start=$(wc -l < $L)
  fi
  sleep 1
done
