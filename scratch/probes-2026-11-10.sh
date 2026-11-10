#!/bin/bash
# The first activation probes (train/probes.py), 2026-11-10: every model with weights so far, on the 5080, E first.
# Each run waits for the 5080's lock, so G-E2's checkpoint checks go first when they are due. Stops if E fails.
# Then the summary (train/analysis/probes/summary.md). Log: train/analysis/probes/batch.log.
T=/opt/wsp/train
LOG=$T/analysis/probes/batch.log
log() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }
probe() { bash $T/run-probes.sh 5080 "$1" "$2" >> "$LOG" 2>&1 < /dev/null; }
log "start"
probe E /opt/wsp/runs/pilot-14/submission || { log "E failed: stopping"; exit 1; }
probe O original
probe T-step_100 $T/combined/T-step_100
for n in G-O/step_100 G-E2/step_25 G-E2/step_50; do probe $n $T/runs/${n%%/*}/checkpoints/${n#*/}; done
for n in R-O/round_1 R-O/round_2 R-E/round_1 R-E/round_2; do probe $n $T/runs/${n%%/*}/${n#*/}/checkpoint; done
for s in 25 50 75 125 150; do probe G-O/step_$s $T/runs/G-O/checkpoints/step_$s; done
for s in 25 50 75 100 125 150; do probe G-E/step_$s $T/runs/G-E/checkpoints/step_$s; done
WSP_TRAIN=$T /opt/wsp/venv/bin/python $T/probes.py summary > /dev/null 2>> "$LOG"
log "end: $(ls $T/analysis/probes/*.json | grep -vc index) models probed"
