#!/bin/bash
# pilot-20 (plans/edit-after-training.md): an editing session that starts from the trained model G-O (GRPO from O,
# step 100), with the held-back practice set (practice/val2, train/held_back_practice.py), E for reference in
# work/E/, and the panel, on the 5090. Then its evaluation hand-offs: after-run.sh (pilot-20-pp15) and
# after-pilot-20.sh (the pairing with G-O, and pilot-20-code-pp0).
set -e
S=/opt/wsp/train/runs/G-O/checkpoints/step_100
R=/opt/wsp/runs/pilot-20
test ! -e $R || { echo "runs/pilot-20 already exists"; exit 1; }
cmp -s <(cd $S && md5sum *.npy) <(cd /opt/wsp/eval/results/G-O-pp15/submission && md5sum *.npy) \
  || { echo "the start differs from the weights G-O-pp15 evaluated"; exit 1; }
test "$(ls $S/*.npy | wc -l)" -eq 318 || { echo "the start does not hold 318 tensors"; exit 1; }
test "$(cat /opt/wsp/practice/val2/*.jsonl | wc -l)" -eq 220 || { echo "practice/val2 does not hold 220 problems"; exit 1; }
grep -o "{{[A-Z_]*}}" /opt/wsp/harness/briefs/pilot-20.md | grep -qv "{{N_PASSES}}" && { echo "the brief has another placeholder"; exit 1; }
for f in G-O-pp15 G-O-code-pp0; do test -f /opt/wsp/eval/results/$f/report.md || { echo "results/$f is missing"; exit 1; }; done
flock -n /opt/wsp/runs/.gpu.lock true || { echo "the 5090 is busy"; exit 1; }
mkdir -p $R/work/E
cp /opt/wsp/runs/pilot-14/submission/*.npy $R/work/E/
cp /opt/wsp/runs/pilot-14/final_answer.md $R/work/E/report.md
echo "editing the trained model G-O" > $R/label.txt
cd /opt/wsp/harness
nohup setsid /opt/wsp/venv/bin/python driver.py pilot-20 --passes 24 --start-from $S --practice /opt/wsp/practice/val2 \
  --brief briefs/pilot-20.md --presence-penalty 1.5 --panel > /opt/wsp/runs/pilot-20.out 2>&1 < /dev/null &
sleep 3
nohup setsid bash /opt/wsp/harness/after-run.sh pilot-20 5090 $S > /dev/null 2>&1 < /dev/null &
nohup setsid bash /opt/wsp/harness/after-pilot-20.sh > /dev/null 2>&1 < /dev/null &
sleep 20
tail -3 /opt/wsp/runs/pilot-20.out
