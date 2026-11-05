#!/bin/bash
# Round 3: pilot-14 (5090, answers that restart after </think>) and pilot-15 (5080, whole-answer instruction
# formats), both from the combined edit runs/combo-12-13/submission, each with its evaluation hand-off.
set -e
C=/opt/wsp/runs/combo-12-13
if [ ! -f $C/final_answer.md ]; then
  { cat /opt/wsp/scratch/combo-header.md; echo "# Run A"; echo; cat /opt/wsp/runs/pilot-12/final_answer.md; echo;
    echo "---"; echo; echo "# Run B"; echo; cat /opt/wsp/runs/pilot-13/final_answer.md; } > $C/final_answer.md
fi
for r in pilot-14 pilot-15; do
  test ! -e /opt/wsp/runs/$r || { echo "runs/$r already exists"; exit 1; }
done
echo "answers that restart after the thinking closes" > /tmp/label-14
echo "instructions that constrain the whole answer" > /tmp/label-15
cd /opt/wsp/harness
nohup setsid /opt/wsp/venv/bin/python driver.py pilot-14 --passes 20 --start-from $C/submission \
  --brief briefs/pilot-14.md --presence-penalty 1.5 > /opt/wsp/runs/pilot-14.out 2>&1 < /dev/null &
sleep 3
nohup setsid bash /opt/wsp/harness/after-run.sh pilot-14 5090 $C/submission > /dev/null 2>&1 < /dev/null &
nohup setsid /opt/wsp/venv/bin/python driver.py pilot-15 --passes 20 --start-from $C/submission \
  --brief briefs/pilot-15.md --presence-penalty 1.5 --gpu 5080 --kernel-memory-gb 14 \
  > /opt/wsp/runs/pilot-15.out 2>&1 < /dev/null &
sleep 3
nohup setsid bash /opt/wsp/harness/after-run.sh pilot-15 5080 $C/submission > /dev/null 2>&1 < /dev/null &
sleep 20
for r in 14 15; do
  [ -d /opt/wsp/runs/pilot-$r ] && mv /tmp/label-$r /opt/wsp/runs/pilot-$r/label.txt
done
pgrep -af "driver.py pilot-1[45]|after-run.sh" | grep -v pgrep | cut -c1-140
echo launched
