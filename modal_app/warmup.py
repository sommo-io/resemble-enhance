"""Pre-build memory snapshots right after a deploy, so real requests don't pay for it.

Modal keeps a snapshot per worker type (2-3 per GPU type) and builds each one lazily on the first
container that lands on that worker type, which adds ~30-40 s to that request. This starts several
containers in parallel so they spread over worker types and build the snapshots up front. Modal
doesn't guarantee every worker type gets hit, so an occasional slow first start can still happen.

    python modal_app/warmup.py [app_name] [containers]
"""

import concurrent.futures as cf
import sys
import time

import modal

app_name = sys.argv[1] if len(sys.argv) > 1 else "resemble-enhance"
containers = int(sys.argv[2]) if len(sys.argv) > 2 else 5

enhancer = modal.Cls.from_name(app_name, "Enhancer")()


def one(i: int) -> str:
    t0 = time.time()
    out = enhancer.warmup.remote(hold_seconds=20)
    return f"  #{i + 1}: {time.time() - t0:5.1f} s  task {out['task']}  on {out['device']}"


print(f"warming {containers} containers of {app_name}...")
with cf.ThreadPoolExecutor(containers) as pool:
    for line in pool.map(one, range(containers)):
        print(line)
