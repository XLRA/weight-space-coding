#!/bin/bash
# pilot-18: the code answer's interface and identifiers under presence penalty, on a Colab A100 (--gpu colab), from
# pilot-14's final edit (runs/start-r4, as round 4). 12 passes, about 23-27 of the ~40 Colab compute units. Its
# evaluation hand-off waits for the first free local GPU. The Colab notebook is opened by hand when the driver
# logs "waiting for the Colab worker" (at the first forward pass); see README, Colab runner.
set -e
S=/opt/wsp/runs/start-r4
test -d $S/submission && test -f $S/final_answer.md || { echo "runs/start-r4 is missing"; exit 1; }
cmp -s <(cd $S/submission && md5sum *) <(cd /opt/wsp/runs/pilot-14/submission && md5sum *) || { echo "start-r4 differs from pilot-14"; exit 1; }
test ! -e /opt/wsp/runs/pilot-18 || { echo "runs/pilot-18 already exists"; exit 1; }
test -s /opt/wsp/secrets/colab/key || { echo "no Colab relay key"; exit 1; }
echo "the code answer's interface and identifiers (Colab A100)" > /tmp/label-18
cd /opt/wsp/harness
nohup setsid /opt/wsp/venv/bin/python driver.py pilot-18 --passes 12 --start-from $S/submission \
  --brief briefs/pilot-18.md --presence-penalty 1.5 --gpu colab --kernel-memory-gb 14 --gpu-idle-minutes 45 \
  > /opt/wsp/runs/pilot-18.out 2>&1 < /dev/null &
sleep 3
nohup setsid bash /opt/wsp/harness/after-run.sh pilot-18 any $S/submission > /dev/null 2>&1 < /dev/null &
sleep 20
[ -d /opt/wsp/runs/pilot-18 ] && mv /tmp/label-18 /opt/wsp/runs/pilot-18/label.txt
pgrep -af "driver.py pilot-18|after-run.sh pilot-18" | grep -v pgrep | cut -c1-140
echo launched
