#!/bin/bash
# Panel noise floor: the starting edit with one weight moved by one bf16 step, against the cached start answers
# (runs/panel-null). Releases the 5090 hold for pilot-19 while it runs and puts it back afterwards.
R=/opt/wsp/runs/panel-ulp
test ! -e $R || { echo "$R exists"; exit 1; }
mkdir -p $R/panel
cp /opt/wsp/runs/panel-null/panel/base_start.jsonl $R/panel/base_start.jsonl
echo "panel noise floor: the start with one weight moved by one bf16 step (replay)" > $R/label.txt
if [ -s /opt/wsp/scratch/hold-5090.pgid ]; then
  kill -- -"$(cat /opt/wsp/scratch/hold-5090.pgid)" 2>/dev/null || true
  rm -f /opt/wsp/scratch/hold-5090.pgid
  sleep 1
fi
cd /opt/wsp/harness
/opt/wsp/venv/bin/python driver.py panel-ulp --backend replay --replay /opt/wsp/scratch/panel_ulp_replies.json \
  --passes 1 --kernel-seconds 120 --panel --gpu 5090 --start-from /opt/wsp/runs/start-r4/submission \
  --presence-penalty 1.5 > /opt/wsp/runs/panel-ulp.out 2>&1
echo "driver exit $?" >> /opt/wsp/runs/panel-ulp.out
setsid bash -c 'echo $$ > /opt/wsp/scratch/hold-5090.pgid; exec flock /opt/wsp/runs/.gpu.lock sleep 43200' \
  < /dev/null > /dev/null 2>&1 &
sleep 2
echo "hold re-armed: $(cat /opt/wsp/scratch/hold-5090.pgid)" >> /opt/wsp/runs/panel-ulp.out
