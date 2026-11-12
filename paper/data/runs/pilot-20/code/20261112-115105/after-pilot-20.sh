#!/bin/bash
# After pilot-20 (plans/edit-after-training.md, "Evaluation, fixed before the run"). Start it right after the driver
# and after-run.sh (scratch/launch-pilot-20.sh does). It waits for after-run.sh, which evaluates pilot-20 on the five
# suites at presence penalty 1.5 (results/pilot-20-pp15) if the run ends with weights that differ from G-O's. Then it
# pairs that with G-O-pp15 (results/pilot-20-pp15/vs-G-O.md), evaluates the code suite at presence penalty 0 on the
# 5090 (results/pilot-20-code-pp0), and pairs that with G-O-code-pp0 (results/pilot-20-code-pp0/vs-G-O.md).
# Log: /opt/wsp/runs/after-pilot-11.log, shared with the other hand-offs.
EV=/opt/wsp/eval
PY=/opt/wsp/venv/bin/python
log() { echo "$(date -u '+%F %T') [after-pilot-20] $*" >> /opt/wsp/runs/after-pilot-11.log; }
trap '$PY /opt/wsp/paper/collect.py > /dev/null 2>&1' EXIT  # refresh the paper tables
log "waiting for after-run.sh pilot-20"
sleep 60
while pgrep -f "[a]fter-run.sh pilot-20 " > /dev/null; do sleep 60; done
if [ ! -f $EV/results/pilot-20-pp15/report.md ]; then
  log "results/pilot-20-pp15 has no report (no evaluation, or it failed); nothing more to do"; exit 0
fi
cd $EV || exit 1
$PY compare.py results/pilot-20-pp15 results/G-O-pp15 > results/pilot-20-pp15/vs-G-O.md 2>&1
log "pilot-20-pp15 paired with G-O-pp15: $(tail -1 results/pilot-20-pp15/vs-G-O.md)"
log "evaluating pilot-20 on the code suite at presence penalty 0 on the 5090 (pilot-20-code-pp0)"
$PY evaluate.py run pilot-20-code-pp0 --submission /opt/wsp/runs/pilot-20/submission --suites code \
  --reuse-original pilot-10 --gpu 5090 > $EV/pilot-20-code-pp0.out 2>&1 < /dev/null
code=$?
log "evaluation pilot-20-code-pp0 ended with exit code $code"
if [ $code -eq 0 ]; then
  $PY compare.py results/pilot-20-code-pp0 results/G-O-code-pp0 > results/pilot-20-code-pp0/vs-G-O.md 2>&1
  log "pilot-20-code-pp0 paired with G-O-code-pp0: $(grep '^| code | all' results/pilot-20-code-pp0/vs-G-O.md)"
fi
