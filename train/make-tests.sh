#!/bin/bash
# Build train/data2 (train/build_tests.py): prep on the host, make_tests.py in the grading container (no network,
# read-only, uid 65534), assemble on the host. About 15 minutes on 8 cores.
set -euo pipefail
T=/opt/wsp/train
W=$T/tests/work
/opt/wsp/venv/bin/python $T/build_tests.py prep
echo "$(date -u '+%F %T') making tests" >&2
nice -n 10 docker run --rm --network none --read-only --tmpfs /tmp:rw,size=2g,mode=1777 \
  --user 65534:65534 --cap-drop ALL --security-opt no-new-privileges \
  --memory 12g --pids-limit 512 --cpus 8 -w /tmp -e WORKERS=8 \
  -v /opt/wsp/harness:/harness:ro -v $T/make_tests.py:/train/make_tests.py:ro -v $W:/work:ro \
  wsp-grade:1 python3 -W ignore /train/make_tests.py /work/in.jsonl > $W/out.jsonl.tmp 2> $W/err.log
[ "$(wc -l < $W/out.jsonl.tmp)" = "$(wc -l < $W/in.jsonl)" ] || { echo "make_tests incomplete" >&2; exit 1; }
mv $W/out.jsonl.tmp $W/out.jsonl
echo "$(date -u '+%F %T') assembling" >&2
/opt/wsp/venv/bin/python $T/build_tests.py assemble
