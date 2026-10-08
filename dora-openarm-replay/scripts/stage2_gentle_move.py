#!/usr/bin/env python
"""STAGE 2 — gentle single-joint move for the ORCABOT 6-motor arm.

After Stage 1 proves torque-hold is stable, this commands ONE small, slow move
(default: base-yaw J1 by +0.15 rad ~= 8.6 deg) and returns, via the driver's
own smooth_move interpolation. Verifies: position tracking, correct direction,
and gains under motion. NO Quest, NO IK, NO dora.

  * E-STOP in hand. Clear the arm's swept volume for the chosen joint.
  * ~0.001 rad per 50 Hz step (well under the 0.2 rad delta backstop).
  * Ctrl-C -> finally -> stop() -> torque off.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import openarm_driver


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--side", default="left_arm", choices=("left_arm", "right_arm"))
    ap.add_argument("--joint", type=int, default=0, help="joint index 0..5 to nudge (default 0 = J1 base yaw)")
    ap.add_argument("--delta", type=float, default=0.15, help="radians to move then undo (default 0.15)")
    ap.add_argument("--secs", type=float, default=3.0, help="seconds each way")
    a = ap.parse_args()

    cfg = openarm_driver.Config(a.config)
    arm = openarm_driver.SingleArmDriver(a.side, cfg)
    # Driver hardcodes num_mit_motors to the V1 count (7/8); this arm has 6.
    arm.num_mit_motors = len(cfg.get_send_ids())

    p0 = np.asarray(arm.fetch_position(refresh=True), dtype=float)
    # Pin start/stop pose to the live reading: torque-on holds in place, and on
    # exit the arm returns to this same safe pose before disabling.
    hold = [float(x) for x in p0]
    arm.config._config["start"]["moves"][0]["position"][a.side] = hold
    arm.config._config["stop"]["moves"][0]["position"][a.side] = hold
    print(f"[stage2] measured pose (rad): {[round(float(x),4) for x in p0]}", flush=True)
    print(f"[stage2] plan: J{a.joint+1} {a.delta:+.3f} rad over {a.secs}s, hold 1s, return.", flush=True)
    print("[stage2] >>> enabling TORQUE now. E-STOP READY. <<<", flush=True)

    try:
        arm.start()   # torque ON, hold current (start pose == current)
        target = p0.copy()
        target[a.joint] = p0[a.joint] + a.delta
        print(f"[stage2] moving to {[round(float(x),4) for x in target]} ...", flush=True)
        arm.smooth_move(target, hz=50, duration=a.secs)
        p = np.asarray(arm.fetch_position(refresh=True), dtype=float)
        print(f"[stage2] reached {[round(float(x),4) for x in p]} "
              f"(J{a.joint+1} err {p[a.joint]-target[a.joint]:+.4f} rad)", flush=True)
        time.sleep(1.0)
        print("[stage2] returning to start...", flush=True)
        arm.smooth_move(p0, hz=50, duration=a.secs)
        p = np.asarray(arm.fetch_position(refresh=True), dtype=float)
        print(f"[stage2] back at {[round(float(x),4) for x in p]}", flush=True)
    except KeyboardInterrupt:
        print("\n[stage2] Ctrl-C — releasing.", flush=True)
    except Exception as e:
        print(f"[stage2] ERROR: {e}", flush=True)
    finally:
        print("[stage2] stopping (torque OFF)...", flush=True)
        try:
            arm.stop()
        except Exception as e:
            print(f"[stage2] stop() error: {e}", flush=True)
        print("[stage2] done.", flush=True)


if __name__ == "__main__":
    main()
