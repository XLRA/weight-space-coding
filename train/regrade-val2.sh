#!/bin/bash
# Grades every saved check's validation answers on train/data2's tests (2026-11-08, plans/training-phase.md,
# "Deviations"): train/ckpt_grade.py writes val2.jsonl and adds val2 to the check's summary.json, pairing it with the
# base its summary already names. Checks without a base (the starts) go first. In wsp-train:1 with no GPU and no
# network, as run-ckpt.sh grades.
#   bash /opt/wsp/train/regrade-val2.sh [<check name>...]      (default: every train/evals/**/summary.json)
# Log: train/evals/regrade-val2.log. A check that already has val2.jsonl is only summarised again.
T=/opt/wsp/train
LOG=$T/evals/regrade-val2.log
[ -s $T/data2/val.jsonl ] || { echo "no train/data2/val.jsonl yet (train/make-tests.sh)" >&2; exit 2; }
/opt/wsp/venv/bin/python - "$@" <<'EOF' > /tmp/regrade-val2.list
import glob, json, os, re, sys
E = "/opt/wsp/train/evals"
names = sys.argv[1:] or [os.path.relpath(os.path.dirname(p), E) for p in glob.glob(f"{E}/**/summary.json", recursive=True)]
key = lambda n: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", n)]
rows = [(n, json.load(open(f"{E}/{n}/summary.json")).get("vs_base", {}).get("base", "")) for n in names]
for n, base in sorted(rows, key=lambda r: (bool(r[1]), key(r[0]))):
    print(n, base)
EOF
fails=0
while read -r name base; do
  echo "$(date -u '+%F %T') $name${base:+ (base $base)}" >> "$LOG"
  docker run --rm --name "wsp-regrade-$(echo "$name" | tr '/' '-')" --network none --pids-limit 1024 --cpus 8 \
    -e PYTHONPATH=/harness:/train -v /opt/wsp/models/qwen3.5-0.8b:/model:ro -v /opt/wsp/harness:/harness:ro \
    -v $T:/train wsp-train:1 python3 /train/ckpt_grade.py "$name" $base >> "$LOG" 2>&1 < /dev/null \
    || { fails=$((fails + 1)); echo "$(date -u '+%F %T') $name failed" >> "$LOG"; }
done < /tmp/regrade-val2.list
echo "$(date -u '+%F %T') done, $fails failed" >> "$LOG"
[ $fails = 0 ]
