#!/bin/bash
# Day 1 of the training phase (plans/training-phase.md), 2026-11-07, once E was frozen as pilot-14's edit. On the 5090:
# E's starting check, then the RFT arm R-E, then R-O's second round. R-O's first round ended at 14:22 UTC in an
# out-of-memory error as vLLM woke up for round 2: the trainer's last autograd graph kept its parameters and
# gradients alive. rft.py now frees them, and R-O resumes at round 2. Each RFT run is started once more if it ends
# without logging "done" (rft.py resumes where it stopped).
# Checks: R-E's checkpoints on the 5080, paired with start-E, once start-E is checked. R-O's are checked by the
# watcher started overnight without a base (start-O's check was not graded yet), and are paired with start-O at the
# end (a rerun of run-ckpt.sh repeats only the validation grading).
#   setsid nohup bash /opt/wsp/train/day1-2026-11-07.sh > /dev/null 2>&1 < /dev/null &
# Log: train/runs/day1-2026-11-07.log
E=/opt/wsp/runs/pilot-14/submission
T=/opt/wsp/train
LOG=$T/runs/day1-2026-11-07.log
log() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }

rft() {  # <arm> [--start <folder>]: train/rft.py on the 5090, once more if it ends without "done"
  local arm=$1; shift
  for attempt in 1 2; do
    log "$arm: attempt $attempt on the 5090"
    bash $T/run-train.sh 5090 rft "$arm" "$@" > /dev/null 2>&1 < /dev/null
    if grep -q ' done$' $T/runs/$arm/train.log 2>/dev/null; then log "$arm: done"; return 0; fi
    log "$arm: ended without logging done; last line: $(grep -E '^[0-9]{4}-|Error' $T/runs/$arm/train.log | tail -1 | cut -c1-200)"
  done
  return 1
}

log "start-E's check on the 5090 (E = pilot-14's edit, $E)"
if bash $T/run-ckpt.sh 5090 start-E $E > /dev/null 2>&1 < /dev/null && [ -s $T/evals/start-E/summary.json ]; then
  log "start-E checked; watching R-E's checkpoints (checks on the 5080, paired with start-E)"
  setsid nohup bash $T/watch-ckpts.sh R-E start-E 5080 > /dev/null 2>&1 < /dev/null &
else
  log "start-E's check failed (train/evals/start-E/log); R-E's checkpoints are not watched"
fi
rft R-E --start $E
rft R-O
R_O_DONE=$?
log "the 5090 lane is done"

if [ $R_O_DONE != 0 ]; then log "R-O is not done, so its checkpoints are left to pair by hand"; exit 1; fi
log "waiting for the overnight watcher to check R-O's checkpoints"
while pgrep -f "[w]atch-ckpts.sh R-O " > /dev/null; do sleep 300; done
BASE=start-O; [ -s $T/evals/start-O/summary.json ] || BASE=""
for d in $T/runs/R-O/round_*/checkpoint; do
  name=R-O/$(basename "$(dirname "$d")")
  if bash $T/run-ckpt.sh 5080 "$name" "$d" $BASE > /dev/null 2>&1 < /dev/null; then
    log "$name checked, paired with ${BASE:-nothing}"
  else
    log "$name: check failed (train/evals/$name/log)"
  fi
done
log "all done"
