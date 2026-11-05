#!/bin/bash
# Builds the combined edit (pilot-12's answer-ending neuron + pilot-13's function-name gates, both on top of
# pilot-10's edit) and evaluates it on the 5090 like the single runs.
set -e
S=/opt/wsp/runs/combo-12-13/submission
mkdir -p "$S"
cp -p /opt/wsp/runs/pilot-12/submission/*.npy "$S/"
for f in /opt/wsp/runs/pilot-13/submission/*.npy; do
  b=$(basename "$f")
  if [ -e "$S/$b" ]; then
    cmp -s "$f" "$S/$b" || { echo "conflict: $b differs between pilot-12 and pilot-13"; exit 1; }
  else
    cp -p "$f" "$S/"
  fi
done
/opt/wsp/venv/bin/python - <<'EOF'
import json
m = {}
for r in ("pilot-12", "pilot-13"):
    for k, v in json.load(open(f"/opt/wsp/runs/{r}/submission/manifest.json")).items():
        assert m.get(k, v) == v, k
        m[k] = v
json.dump(m, open("/opt/wsp/runs/combo-12-13/submission/manifest.json", "w"), indent=1)
print(len(m), "tensors in the combined manifest")
EOF
ls "$S"
echo "$(date -u '+%F %T') [combo-12-13] built from pilot-12 + pilot-13; evaluating on the 5090 (combo-12-13-pp15)" >> /opt/wsp/runs/after-pilot-11.log
cd /opt/wsp/eval
nohup setsid /opt/wsp/venv/bin/python evaluate.py run combo-12-13-pp15 --submission "$S" --suites all \
  --presence-penalty 1.5 --reuse-original pilot-10-pp15 > /opt/wsp/eval/combo-12-13-pp15.out 2>&1 < /dev/null &
sleep 5
pgrep -af "evaluate.py run combo-12-13-pp15" | cut -c1-120
echo launched
