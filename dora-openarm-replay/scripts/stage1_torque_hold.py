#!/usr/bin/env python
"""STAGE 1 — first torque-on smoke test for the ORCABOT 6-motor arm.

Enables torque and HOLDS the current pose (start pose in the config == the measured
current pose, so smooth_move current->current = no commanded motion), holds for a
few seconds while printing position drift, then disables. NO Quest, NO IK, NO dora.

  * Keep the E-STOP in hand. Torque comes on the instant start() is called.
  * The arm should become stiff and stay put. If ANY joint creeps/oscillates,
    hit the e-stop; we retune gains before going further.
  * Ctrl-C at any time -> jumps to the finally block -> stop() -> torque off.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import openarm_driver


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="orcabot_arm_driver.yaml (6 motors)")
    ap.add_argument("--side", default="left_arm", choices=("left_arm", "right_arm"))
    ap.add_argument("--hold", type=float, default=4.0, help="seconds to hold before releasing")
    a = ap.parse_args()

    cfg = openarm_driver.Config(a.config)
    arm = openarm_driver.SingleArmDriver(a.side, cfg)   # construct = read-only, no torque yet
    # Belt-and-suspenders: the stock driver hardcodes num_mit_motors to the V1
    # count (7/8); this arm has 6. We patched the driver to derive it, but if the
    # venv is reinstalled that patch is lost — so pin the real count here too.
    arm.num_mit_motors = len(cfg.get_send_ids())

    p0 = np.asarray(arm.fetch_position(refresh=True), dtype=float)
    # Pin the driver's start/stop pose to the LIVE reading so start()'s smooth_move
    # is current->current (truly zero motion). The config's baked pose goes stale
    # the instant the arm is back-driven with torque off — that staleness is what
    # made the previous run swing ~49 deg on j3 at start.
    hold = [float(x) for x in p0]
    arm.config._config["start"]["moves"][0]["position"][a.side] = hold
    arm.config._config["stop"]["moves"][0]["position"][a.side] = hold
    print(f"[stage1] pre-start measured pose (rad): {[round(float(x),4) for x in p0]}", flush=True)
    print("[stage1] hold pose pinned to live reading -> start() should not move.", flush=True)
    print("[stage1] >>> enabling TORQUE now (holds current pose). E-STOP READY. <<<", flush=True)

    try:
        arm.start()   # enable_all() torque ON, then smooth_move current->current (no motion)
        print("[stage1] torque ON, holding. Watching for drift...", flush=True)
        t_end = time.time() + a.hold
        while time.time() < t_end:
            p = np.asarray(arm.fetch_position(refresh=True), dtype=float)
            drift = np.max(np.abs(p - p0))
            print(f"[stage1]  pose={[round(float(x),4) for x in p]}  max|drift|={drift:.4f} rad",
                  flush=True)
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[stage1] Ctrl-C — releasing.", flush=True)
    finally:
        print("[stage1] stopping (torque OFF)...", flush=True)
        try:
            arm.stop()
        except Exception as e:
            print(f"[stage1] stop() error: {e}", flush=True)
        print("[stage1] done.", flush=True)


if __name__ == "__main__":
    main()
