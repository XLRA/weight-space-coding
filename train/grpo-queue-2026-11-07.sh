#!/bin/bash
# The GRPO arms (plans/training-phase.md, timeline steps 3 and 4), queued 2026-11-07 behind day 1's RFT runs on the
# 5090: G-E (from E, pilot-14's edit), then G-O (from the original model), 150 steps each with the recipe's settings.
# Each arm's checkpoints are checked on the 5080 as they are saved, paired with its start's check.
#
# Fixed before either run:
# - G-E's first 20 steps are the smoke run. Nothing in a step depends on the step count, so they are G-E's own.
# - vLLM gets 0.60 of the 5090 (the default 0.30 leaves room for about 480k tokens of cache, too few for 256 answers
#   at once). If a run ends out of GPU memory, it and later runs go back to 0.30.
# - Each run gets a second attempt, which resumes from its last checkpoint. If G-E fails twice, the queue stops.
# - G-O's stop rule: once its step-50 check is graded, G-O stops unless its validation pass rate is above start-O's by
#   more than two standard errors.
# - If train/queue/after-G-E.sh exists when G-E ends, it runs before G-O (jobs added later, such as the RFT arms'
#   held-out evaluations).
# Log: train/runs/grpo-queue-2026-11-07.log. What is still queued: train/queue.json, which progress.py shows.
E=/opt/wsp/runs/pilot-14/submission
T=/opt/wsp/train
LOG=$T/runs/grpo-queue-2026-11-07.log
Q=$T/queue.json
MEM=0.60
DONE='^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8} done$'  # the trainer's last line (stricter than any line ending in done)
log() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }
queued() {  # queued [<what the first waits for> <name>...]: writes train/queue.json for the watcher
  local items=() after=$1; shift
  for a in "$@"; do
    case $a in
      G-E) items+=("{\"name\": \"G-E\", \"kind\": \"grpo\", \"gpu\": \"5090\", \"after\": \"$after\", \"args\": \" --start $E --gpu-memory $MEM\"}");;
      G-O) items+=("{\"name\": \"G-O\", \"kind\": \"grpo\", \"gpu\": \"5090\", \"after\": \"$after\", \"args\": \" --gpu-memory $MEM\"}");;
    esac
    after=$a
  done
  local IFS=,; echo "[${items[*]}]" > $Q.tmp && mv $Q.tmp $Q
}
grpo() {  # <arm> [--start <folder>]: train/grpo.py on the 5090; 0 done, 1 failed twice, 2 stopped by its rule
  local arm=$1; shift
  for attempt in 1 2; do
    [ -e $T/runs/$arm/stopped ] && return 2
    log "$arm: attempt $attempt on the 5090, vLLM at $MEM of its memory"
    bash $T/run-train.sh 5090 grpo "$arm" "$@" --gpu-memory $MEM > /dev/null 2>&1 < /dev/null
    if grep -q -E "$DONE" $T/runs/$arm/train.log 2>/dev/null; then log "$arm: done"; return 0; fi
    [ -e $T/runs/$arm/stopped ] && return 2
    log "$arm: ended without logging done; last line: $(grep -E '^[0-9]{4}-|Error' $T/runs/$arm/train.log | tail -1 | cut -c1-200)"
    if [ $MEM != 0.30 ] && tail -c 300000 $T/runs/$arm/train.log | grep -q -E "OutOfMemoryError|CUDA out of memory|Error: out of memory"; then
      MEM=0.30; log "$arm ran out of GPU memory: vLLM gets the default 0.30 from now on"
    fi
  done
  return 1
}
step50_rule() {  # G-O's stop rule (plans/training-phase.md, "Risks")
  local s=$T/evals/G-O/step_50/summary.json
  while [ ! -s $s ]; do
    if grep -q -E "$DONE" $T/runs/G-O/train.log 2>/dev/null || [ -e $T/runs/G-O/stopped ]; then return; fi
    sleep 300
  done
  local v
  v=$(python3 -c "
import json, sys
v = json.load(open(sys.argv[1]))['vs_base']['val']
print('keep' if v['diff'] > 2 * v['se'] else 'stop', f\"validation {v['diff']:+.2f} (SE {v['se']:.2f}) against start-O at step 50\")" $s)
  if [ "${v%% *}" = stop ]; then
    echo "the step-50 rule: ${v#* }" > $T/runs/G-O/stopped
    log "G-O: stopping it, by the step-50 rule: ${v#* }"
    docker stop wsp-train-G-O > /dev/null 2>&1
  else
    log "G-O: the step-50 rule keeps it running: ${v#* }"
  fi
}

log "queued: G-E, then G-O, once day 1's RFT runs leave the 5090"
queued "the RFT runs" G-E G-O
while ! grep -q "the 5090 lane is done" $T/runs/day1-2026-11-07.log 2>/dev/null && pgrep -f "[d]ay1-2026-11-07.sh" > /dev/null; do
  sleep 120
done

log "G-E starts (from E, $E); its checkpoints are checked on the 5080 against start-E"
queued G-E G-O
setsid nohup bash $T/watch-ckpts.sh G-E start-E 5080 > /dev/null 2>&1 < /dev/null &
grpo G-E --start $E
if [ $? != 0 ]; then
  echo "failed twice; see train.log" > $T/runs/G-E/stopped
  queued
  log "G-E failed twice, so G-O does not start; the queue stops here"
  exit 1
fi

if [ -s $T/queue/after-G-E.sh ]; then
  log "running queue/after-G-E.sh"
  bash $T/queue/after-G-E.sh >> "$LOG" 2>&1
  log "queue/after-G-E.sh ended with exit code $?"
fi

log "G-O starts (from the original model); its checkpoints are checked on the 5080 against start-O"
queued
setsid nohup bash $T/watch-ckpts.sh G-O start-O 5080 > /dev/null 2>&1 < /dev/null &
step50_rule &
grpo G-O
case $? in
  0) ;;
  2) log "G-O: stopped ($(cat $T/runs/G-O/stopped))";;
  *) echo "failed twice; see train.log" > $T/runs/G-O/stopped; log "G-O failed twice";;
esac
log "the GRPO queue is done: the 5090 is free and llama-swap runs again"
