#!/bin/bash
# End-to-end test of the Colab path without a GPU:  test-colab.sh [https://<the tunnel's hostname for the relay>]
# - the driver: run e2e-colab, the fake Claude CLI's 14 scripted turns (2 passes), --gpu colab, GPU idle 1 min (so
#   the fake usage-limit wait releases the GPU and the relay tells the worker);
# - the worker: in a container with the fake GPU server, started again 20 s after it exits, as if the notebook were
#   run again. With no argument it reaches the relay directly on the tunnel's Docker network
#   (http://wsp-colab-relay:8790); with the tunnel's hostname it goes out to the internet and back in through the
#   tunnel, as the notebook does.
# Afterwards: runs/e2e-colab.out, runs/e2e-colab/driver.log, docker logs wsp-colab-test-worker.
# Clean up: docker rm -f wsp-colab-test-worker
set -e
R=e2e-colab
URL=${1:-http://wsp-colab-relay:8790}
test -s /opt/wsp/secrets/colab/key || { echo "no relay key"; exit 1; }
if [ -e /opt/wsp/runs/$R ]; then mv /opt/wsp/runs/$R /opt/wsp/runs/$R-$(date +%H%M); fi
mv /opt/wsp/runs/$R.out /opt/wsp/runs/$R-$(date +%H%M).out 2>/dev/null || true
docker rm -f wsp-colab-test-worker > /dev/null 2>&1 || true
docker run -d --name wsp-colab-test-worker --network compose_default --memory 6g -e FAKE_SECONDS=20 \
  -v /opt/wsp/harness:/harness:ro -v /opt/wsp/secrets/colab:/colab:ro \
  wsp-runtime:1 sh -c "while true; do python3 -u /harness/tests/colab_local_worker.py '$URL';
                       echo \"worker exited (\$?); starting again in 20 s\"; sleep 20; done"
cd /opt/wsp/harness
nohup setsid /opt/wsp/venv/bin/python driver.py $R --backend claude-code --claude-bin /opt/wsp/harness/tests/fake_claude.py \
  --passes 2 --handoff-every 7 --kernel-seconds 20 --gpu-idle-minutes 1 --gpu colab --kernel-memory-gb 8 \
  > /opt/wsp/runs/$R.out 2>&1 < /dev/null &
echo "started; the worker uses $URL"
