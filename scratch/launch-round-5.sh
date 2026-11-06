#!/bin/bash
# Round 5: pilot-19 consolidates round 4 (pilot-16 stopper, pilot-17 copy, pilot-18 identifiers) from runs/start-r4,
# with the panel (--panel), on the 5090, with its evaluation hand-off.
# - The candidates (each run's changed tensors and report, and their sum) are built into work/candidates/ by
#   scratch/build_candidates.py; the log is runs/pilot-19/candidates.log.
# - The panel's start answers come from the null check runs/panel-null (same start edit, 5090, presence penalty 1.5),
#   so pilot-19's panel passes generate only its edit.
# - The 5090's lock was held for this run by a placeholder (process group in scratch/hold-5090.pgid), so that
#   pilot-18's evaluation took the 5080; it is released right before the driver starts.
set -e
S=/opt/wsp/runs/start-r4
R=/opt/wsp/runs/pilot-19
test ! -e $R || { echo "runs/pilot-19 already exists"; exit 1; }
for r in pilot-16 pilot-17 pilot-18; do
  grep -q '"status": "done"' /opt/wsp/runs/$r/state.json || { echo "$r is not done"; exit 1; }
done
cmp -s <(cd $S/submission && md5sum *) <(cd /opt/wsp/runs/pilot-14/submission && md5sum *) || { echo "start-r4 differs from pilot-14"; exit 1; }
grep -q '"start_from": "/opt/wsp/runs/start-r4/submission"' /opt/wsp/runs/panel-null/config.json || { echo "panel-null used another start"; exit 1; }
grep -q '"status": "done"' /opt/wsp/runs/panel-null/state.json || { echo "panel-null is not done"; exit 1; }
test "$(wc -l < /opt/wsp/runs/panel-null/panel/base_start.jsonl)" -eq 1000 || { echo "panel-null has no complete start answers"; exit 1; }
grep -q "{{IDENTIFIERS}}\|{{PANEL_BASE_NOTE}}\|{{PANEL_NOISE}}" /opt/wsp/harness/briefs/pilot-19.md && { echo "the brief still has placeholders"; exit 1; }
mkdir -p $R/work/candidates $R/panel
docker run --rm --network none -v /opt/wsp/models/qwen3.5-0.8b:/model:ro -v $S/submission:/start:ro \
  -v /opt/wsp/runs:/runs:ro -v /opt/wsp/scratch:/scratch:ro -v $R/work/candidates:/out wsp-runtime:1 \
  python3 -W ignore /scratch/build_candidates.py stopper=/runs/pilot-16/submission copy=/runs/pilot-17/submission \
  identifiers=/runs/pilot-18/submission > $R/candidates.log 2>&1
cat $R/candidates.log
cp /opt/wsp/runs/panel-null/panel/base_start.jsonl $R/panel/base_start.jsonl
echo "consolidating round 4 with the panel" > $R/label.txt
if [ -s /opt/wsp/scratch/hold-5090.pgid ]; then
  kill -- -"$(cat /opt/wsp/scratch/hold-5090.pgid)" 2>/dev/null || true
  rm -f /opt/wsp/scratch/hold-5090.pgid
  sleep 1
fi
cd /opt/wsp/harness
nohup setsid /opt/wsp/venv/bin/python driver.py pilot-19 --passes 24 --start-from $S/submission \
  --brief briefs/pilot-19.md --presence-penalty 1.5 --panel > /opt/wsp/runs/pilot-19.out 2>&1 < /dev/null &
sleep 3
nohup setsid bash /opt/wsp/harness/after-run.sh pilot-19 any $S/submission > /dev/null 2>&1 < /dev/null &
sleep 20
pgrep -af "driver.py pilot-19|after-run.sh pilot-19" | grep -v pgrep | cut -c1-160
tail -n 3 /opt/wsp/runs/pilot-19.out
echo launched
