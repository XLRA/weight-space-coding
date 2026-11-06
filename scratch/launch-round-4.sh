#!/bin/bash
# Round 4: pilot-16 (5090, turns that end with no answer, MLP tensors only) and pilot-17 (5080, copying text
# from the prompt under presence penalty, attention tensors only), both from pilot-14's final edit, each with
# its evaluation hand-off. The context handoff default (400k) applies.
set -e
S=/opt/wsp/runs/start-r4
if [ ! -d $S/submission ]; then
  mkdir -p $S
  cp -a /opt/wsp/runs/pilot-14/submission $S/submission
fi
if [ ! -f $S/final_answer.md ]; then
  { cat /opt/wsp/scratch/r4-header.md; cat /opt/wsp/runs/pilot-14/final_answer.md; echo; echo "---"; echo;
    cat /opt/wsp/runs/combo-12-13/final_answer.md; } > $S/final_answer.md
fi
cmp -s <(cd $S/submission && md5sum *) <(cd /opt/wsp/runs/pilot-14/submission && md5sum *) || { echo "start-r4 differs from pilot-14"; exit 1; }
for r in pilot-16 pilot-17; do
  test ! -e /opt/wsp/runs/$r || { echo "runs/$r already exists"; exit 1; }
done
echo "turns that end with no answer (MLP only)" > /tmp/label-16
echo "copying text from the prompt (attention only)" > /tmp/label-17
cd /opt/wsp/harness
nohup setsid /opt/wsp/venv/bin/python driver.py pilot-16 --passes 20 --start-from $S/submission \
  --brief briefs/pilot-16.md --presence-penalty 1.5 > /opt/wsp/runs/pilot-16.out 2>&1 < /dev/null &
sleep 3
nohup setsid bash /opt/wsp/harness/after-run.sh pilot-16 5090 $S/submission > /dev/null 2>&1 < /dev/null &
nohup setsid /opt/wsp/venv/bin/python driver.py pilot-17 --passes 20 --start-from $S/submission \
  --brief briefs/pilot-17.md --presence-penalty 1.5 --gpu 5080 --kernel-memory-gb 14 \
  > /opt/wsp/runs/pilot-17.out 2>&1 < /dev/null &
sleep 3
nohup setsid bash /opt/wsp/harness/after-run.sh pilot-17 5080 $S/submission > /dev/null 2>&1 < /dev/null &
sleep 20
for r in 16 17; do
  [ -d /opt/wsp/runs/pilot-$r ] && mv /tmp/label-$r /opt/wsp/runs/pilot-$r/label.txt
done
pgrep -af "driver.py pilot-1[67]|after-run.sh" | grep -v pgrep | cut -c1-140
echo launched
