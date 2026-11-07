"""Runs the Colab worker (colab_worker.py) on the server for a test, with the stand-in GPU server (fake_gpu_server.py),
so no GPU is needed. It reaches the relay directly on the tunnel's Docker network, without the tunnel:

    docker run --rm -d --name wsp-colab-test-worker --network compose_default --memory 4g \\
        -v /opt/wsp/harness:/harness:ro -v /opt/wsp/secrets/colab:/colab:ro wsp-runtime:1 \\
        python3 -u /harness/tests/colab_local_worker.py [http://wsp-colab-relay:8790]

Instead of shutting a runtime down, it exits, with the reason.
"""
import sys

sys.path.insert(0, "/harness")
import colab_worker  # noqa: E402

url = sys.argv[1] if len(sys.argv) > 1 else "http://wsp-colab-relay:8790"
key = open("/colab/key").read().strip()
w = colab_worker.Worker(lambda: colab_worker.Relay(url, key), root="/content/wsp",
                        server_script="/harness/tests/fake_gpu_server.py",
                        on_idle=lambda why: sys.exit(f"the worker would stop its runtime now: {why}"))
w.loop()
