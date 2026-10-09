#!/usr/bin/env python3
"""Explain why a session has the number of trials it has. Read-only.

    python tools/diagnose_session.py "C:/.../sessions/sS00_n1_20261001_174831"

Uses the article's own segmentation (analysis_scripts/s26_common.py). Prints how long the session
is, what the metadata says was planned, how many grasp and rest commands were logged, the gaps between
consecutive grasp commands (a missing command shows up as a gap of about twice the normal one), and
where the last trial ends compared with the end of the recording.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", type=Path, help="one session folder (the one that contains raw.bin)")
    a = ap.parse_args(argv)
    os.environ.setdefault("EMG8_DATASET", "s26-01")
    os.environ.setdefault("EMG8_OUT", tempfile.mkdtemp(prefix="diagnose_"))
    sys.path.insert(0, str(ROOT / "analysis_scripts"))
    import numpy as np
    import s26_common as S
    from sweeps import phase_commands

    x = S.load26(a.session)
    meta = x.s.meta
    timing = meta.get("timing", {})
    cmds, echoes = phase_commands(x.s)
    grasp = [t for t, p in cmds if p == "grasp"]
    rest = [t for t, p in cmds if p == "rest"]
    labels = [(round(float((x.f(e["wall"]) - x.s.t0_us) / 1e6), 2), e["event"][1], e["event"][2])
              for e in x.s.events if e.get("event", [None])[0] == "label"]
    print(f"session            {a.session.name}")
    print(f"duration           {x.end_s:.1f} s ({x.end_s / 60:.2f} min)   clock-fit residual {x.resid_ms:.1f} ms")
    print(f"planned (metadata) trials={meta.get('trials')}  grasps={meta.get('grasps')}  reps={meta.get('reps')}  "
          f"order={meta.get('order')}  seed={meta.get('seed')}")
    print(f"                   timing={timing}")
    print(f"phase commands     {len(grasp)} grasp, {len(rest)} rest, {len(echoes)} polling echoes ignored")
    print(f"labels logged      {len(labels)}   (first few: {labels[:4]})")
    print(f"trials found       {len(x.trials)}   (a complete session has {meta.get('trials', 42)})   "
          f"grasp identity source: {x.order_source}")
    if grasp:
        gaps = np.diff(grasp)
        typical = float(np.median(gaps))
        print(f"\nfirst grasp at {grasp[0]:.1f} s, last grasp at {grasp[-1]:.1f} s, "
              f"recording ends at {x.end_s:.1f} s")
        print(f"gap between consecutive grasp commands: median {typical:.1f} s "
              f"(a block break is longer; a missing command shows as about twice the median)")
        odd = [(k + 1, round(float(g), 1)) for k, g in enumerate(gaps) if g > 1.6 * typical and g < 2.5 * typical]
        print(f"gaps of about 2x the median (between trial n and n+1): {odd if odd else 'none'}")
        long_ = [(k + 1, round(float(g), 1)) for k, g in enumerate(gaps) if g >= 2.5 * typical]
        print(f"long gaps (block breaks are expected every {timing.get('break_every', 6)} trials): "
              f"{[(n, g) for n, g in long_]}")
        tail = x.end_s - grasp[-1]
        print(f"time from the last grasp command to the end of the recording: {tail:.1f} s "
              f"(a full trial needs about {timing.get('hold_s', 5) + timing.get('rest_s', 7)} s)")
    others = [e["event"] for e in x.s.events if e.get("event", [None])[0] in ("paused", "device_error", "recording")]
    print(f"\nrecording / pause / error events: {others[:12]}")
    print("\ntrials found:")
    for tr in x.trials:
        print(f"  {tr.index + 1:2d}  grasp {tr.grasp:3d} rep {tr.rep}  "
              f"t_grasp {tr.t_grasp:7.1f}  t_rest {tr.t_rest:7.1f}  next {tr.t_next:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
