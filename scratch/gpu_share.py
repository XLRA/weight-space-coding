import calendar
import glob
import json
import sys
import time

for r in sys.argv[1:]:
    w = 0.0
    for f in glob.glob(f"/opt/wsp/runs/{r}/requests/fwd_*.result.json"):
        w += json.load(open(f)).get("wall_seconds", 0)
    log = open(f"/opt/wsp/runs/{r}/driver.log").read().splitlines()
    t = lambda l: calendar.timegm(time.strptime(l[:19], "%Y-%m-%d %H:%M:%S"))
    span = t(log[-1]) - t(log[0])
    st = json.load(open(f"/opt/wsp/runs/{r}/state.json"))
    tok = st["tokens"]
    print(f"{r}: {span / 3600:.1f} h, forward passes {w / 3600:.2f} h ({100 * w / span:.0f}%), "
          f"passes {st['passes_used']}, output tokens {tok['output_tokens']:,}, "
          f"cache reads {tok['cache_read_input_tokens']:,}, limit waits {st.get('limit_waits')}")
