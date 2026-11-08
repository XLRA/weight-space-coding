#!/bin/bash
# The GRPO arms on train/data2's tests (plans/training-phase.md, "Deviations", 2026-11-08), queued 2026-11-08 on the
# user's go-ahead. grpo-queue-2026-11-07.sh ran G-E on train/data, where printing an APPS statement's example output
# passed most problems, and G-E learned to. That queue still runs the RFT arms' held-out evaluations, then skips G-O,
# which runs/G-O/stopped holds. Once that queue has ended, this one runs on the 5090:
# - G-O (from the original model), then G-E2 (G-E again: from E, pilot-14's edit), 150 steps each with the recipe's
#   settings and --data /train/data2/train.jsonl, the usable training problems (train/build_tests.py). G-O goes
#   first: it has not run at all, and T (G-O plus E's delta) needs it.
# - Each arm's checkpoints are checked on the 5080 as they are saved, paired with its start's check. ckpt_grade.py also
#   grades them on data2's tests (val2), and the rules read that from now on.
# - G-O's stop rule: once its step-50 check is graded, G-O stops unless its validation pass rate on data2's tests
#   (vs_base.val2) is above start-O's by more than two standard errors.
# - vLLM gets 0.60 of the 5090. If a run ends out of GPU memory, it and later runs go back to 0.30.
# - Each run gets a second attempt, which resumes from its last checkpoint. If G-O fails twice, the queue stops.
# - If train/queue/after-<arm>.sh exists when an arm ends, it runs next (jobs added later).
# Log: train/runs/grpo-queue-2026-11-08.log. What is still queued: train/queue.json, which progress.py shows.
E=/opt/wsp/runs/pilot-14/submission
T=/opt/wsp/train
LOG=$T/runs/grpo-queue-2026-11-08.log
Q=$T/queue.json
DATA=/train/data2/train.jsonl
MEM=0.60
DONE='^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8} done$'  # the trainer's last line (stricter than any line ending in done)
log() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }
queued() {  # queued [<what the first waits for> <name>...]: writes train/queue.json for the watcher
  local items=() after=$1; shift
  for a in "$@"; do
    case $a in
      G-O) items+=("{\"name\": \"G-O\", \"kind\": \"grpo\", \"gpu\": \"5090\", \"after\": \"$after\", \"args\": \" --data $DATA --gpu-memory $MEM\"}");;
      G-E2) items+=("{\"name\": \"G-E2\", \"kind\": \"grpo\", \"gpu\": \"5090\", \"after\": \"$after\", \"args\": \" --start $E --data $DATA --gpu-memory $MEM\"}");;
    esac
    after=$a
  done
  local IFS=,; echo "[${items[*]}]" > $Q.tmp && mv $Q.tmp $Q
}
grpo() {  # <arm> [options]: train/grpo.py on the 5090; 0 done, 1 failed twice, 2 stopped by its rule
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
step50_rule() {  # G-O's stop rule (plans/training-phase.md, "Risks"), on data2's tests
  local s=$T/evals/G-O/step_50/summary.json
  while [ ! -s $s ]; do
    if grep -q -E "$DONE" $T/runs/G-O/train.log 2>/dev/null || [ -e $T/runs/G-O/stopped ]; then return; fi
    sleep 300
  done
  local v
  v=$(python3 -c "
import json, sys
v = json.load(open(sys.argv[1]))['vs_base'].get('val2')
if v is None:
    print('keep: step_50/summary.json has no validation on the new tests, so the rule cannot be applied')
else:
    print('keep' if v['diff'] > 2 * v['se'] else 'stop', f\"validation on the new tests {v['diff']:+.2f} (SE {v['se']:.2f}) against start-O at step 50\")" $s)
  if [ "${v%% *}" = stop ]; then
    echo "the step-50 rule: ${v#* }" > $T/runs/G-O/stopped
    log "G-O: stopping it, by the step-50 rule: ${v#* }"
    docker stop wsp-train-G-O > /dev/null 2>&1
  else
    log "G-O: the step-50 rule keeps it running: ${v#* }"
  fi
}
hook() {  # train/queue/after-<arm>.sh, if there is one
  if [ -s $T/queue/after-$1.sh ]; then
    log "running queue/after-$1.sh"
    bash $T/queue/after-$1.sh >> "$LOG" 2>&1
    log "queue/after-$1.sh ended with exit code $?"
  fi
}

log "queued: G-O, then G-E2, on $DATA, once grpo-queue-2026-11-07.sh has ended"
queued "the RFT arms' held-out evaluations" G-O G-E2
while pgrep -f "[g]rpo-queue-2026-11-07.sh" > /dev/null || pgrep -f "[w]atch-ckpts.sh G-O " > /dev/null; do
  sleep 60
done
log "grpo-queue-2026-11-07.sh has ended"
if [ -e $T/runs/G-O/train.log ]; then
  log "G-O has run already (runs/G-O/train.log), so this queue stops"; queued; exit 1
fi
[ -e $T/runs/G-O/stopped ] && mv $T/runs/G-O/stopped $T/runs/G-O/held-2026-11-08.txt
if [ ! -s $T/data2/train.jsonl ]; then log "no train/data2/train.jsonl, so this queue stops"; queued; exit 1; fi
for b in start-O start-E; do
  [ -s $T/evals/$b/val2.jsonl ] || { log "$b has no val2.jsonl: grading it"; bash $T/regrade-val2.sh $b; }
done

log "G-O starts (from the original model, on $DATA); its checkpoints are checked on the 5080 against start-O"
queued G-O G-E2
setsid nohup bash $T/watch-ckpts.sh G-O start-O 5080 > /dev/null 2>&1 < /dev/null &
step50_rule &
grpo G-O --data $DATA
case $? in
  0) ;;
  2) log "G-O: stopped ($(cat $T/runs/G-O/stopped))";;
  *) echo "failed twice; see train.log" > $T/runs/G-O/stopped; queued
     log "G-O failed twice, so G-E2 does not start; the queue stops here"; exit 1;;
esac
hook G-O

log "G-E2 starts (from E, $E, on $DATA); its checkpoints are checked on the 5080 against start-E"
queued
setsid nohup bash $T/watch-ckpts.sh G-E2 start-E 5080 > /dev/null 2>&1 < /dev/null &
grpo G-E2 --start $E --data $DATA
case $? in
  0) ;;
  2) log "G-E2: stopped ($(cat $T/runs/G-E2/stopped))";;
  *) echo "failed twice; see train.log" > $T/runs/G-E2/stopped; log "G-E2 failed twice";;
esac
hook G-E2
log "the GRPO queue is done: the 5090 is free and llama-swap runs again"
