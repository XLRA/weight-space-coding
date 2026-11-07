#!/bin/bash
# One-off, 2026-11-07 (the user approved queuing R-O before shutting down for the night). Once pilot-19 and its
# evaluation have finished with the 5090, runs the RFT arm R-O (from the original model, the plan's recipe: rft.py's
# defaults) on the 5090, then checks R-O's round checkpoints on the 5080 as they appear. The checks are paired with
# start-O if its check has finished by then; if not, they run unpaired, and a rerun of run-ckpt.sh with start-O as
# the base redoes only the pairing (the validation grading).
#   setsid nohup bash /opt/wsp/train/overnight-2026-11-07.sh > /dev/null 2>&1 < /dev/null &
# Log: train/runs/overnight-2026-11-07.log
LOG=/opt/wsp/train/runs/overnight-2026-11-07.log
log() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }
log "waiting for pilot-19 and its evaluation to finish with the 5090"
while pgrep -f "[d]river.py pilot-19 " > /dev/null || pgrep -f "[a]fter-run.sh pilot-19 " > /dev/null; do sleep 60; done
log "pilot-19 and its evaluation are done: $(grep 'after-pilot-19' /opt/wsp/runs/after-pilot-11.log | tail -1)"
log "starting R-O on the 5090 (log: train/runs/R-O/train.log)"
setsid nohup bash /opt/wsp/train/run-train.sh 5090 rft R-O > /dev/null 2>&1 < /dev/null &
sleep 60
BASE=start-O
if [ ! -s /opt/wsp/train/evals/start-O/summary.json ]; then
  BASE=""; log "start-O's check has no summary.json yet, so R-O's checks run unpaired"
fi
log "watching R-O's checkpoints (checks on the 5080, base: ${BASE:-none}; log: train/evals/R-O/watch.log)"
setsid nohup bash /opt/wsp/train/watch-ckpts.sh R-O "$BASE" 5080 > /dev/null 2>&1 < /dev/null &
log "handed off"
