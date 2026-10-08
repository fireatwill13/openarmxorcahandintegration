#!/usr/bin/env python
"""Dora sink node: drive the PHYSICAL ORCA hand from quest_live_source's fingers_right,
and (optionally, with --wrist) the ORCA wrist joint from the IK's j7_wrist.

Hardware twin of ``mujoco_viewer_hand.py`` — consumes the same ``fingers_right``
struct ``{names, q}`` (16 finger joint angles, RADIANS) and writes to ``orca_core``
(Feetech servos on /dev/ttyACM0).

NAME TRANSLATION: ``fingers_right`` carries URDF/model joint names; orca_core wants
its own ids (``thumb_cmc``, ``index_mcp``, ...). Bridge = inverse of
``semantic_to_model_joints(side)`` (verified: all 16 map; only ``wrist`` is left over).

UNITS: ``fingers_right`` and the IK output are RADIANS (quest_live_source np.deg2rad's
the retargeter, and the MuJoCo model is radians). orca_core ``set_joint_positions``
and the config ROMs are DEGREES — so we convert rad->deg here or nothing moves.

WRIST (--wrist): the ORCA ``wrist`` joint (joint7) is driven from the IK's
``position_left`` = ``[j1..j6, j7_wrist, gripper]`` (index 6). IK j7 is radians in the
model convention; the ORCA wrist is degrees, ROM [-65, 35]. --wrist-scale (use -1 to
flip sign) and --wrist-offset-deg map one convention onto the other without a code edit.
Without --wrist the wrist is omitted and holds the neutral set at startup (fingers-only).

SAFETY: hand must be CALIBRATED. --no-torque = dry run (prints targets, no motion). On
enable the servos snap to neutral once; hand power switch in reach. POSITION mode is
rigid (fingers push hard into stops/obstacles) — keep the hand clear. Ctrl-C -> torque off.
"""
from __future__ import annotations

import argparse
import math

import dora
import numpy as np
import pyarrow as pa


def _extract_qpos(value) -> np.ndarray:
    """Parse the IK ``position_left`` (struct{"qpos": list<float32>}), same as arm6_from_ik."""
    if pa.types.is_struct(value.type):
        value = value.field("qpos")[0].values
    return np.array(value, dtype=np.float32)


def _run(a: argparse.Namespace) -> None:
    from orca_core import OrcaHand, OrcaJointPositions
    from orca_retarget.mapping import semantic_to_model_joints

    # model (URDF) joint name -> orca_core joint id
    model_to_orca = {v: k for k, v in semantic_to_model_joints(a.side).items()}

    hand = OrcaHand(a.orca_config)
    ok, msg = hand.connect()
    if not ok:
        raise RuntimeError(f"OrcaHand connect failed: {msg}")
    if not hand.is_calibrated():
        raise RuntimeError("ORCA hand is NOT calibrated — refusing to move.")
    print(f"[orca-follower:{a.side}] connected: {msg}", flush=True)

    if not a.no_torque:
        if a.max_current is not None:
            hand.set_max_current(a.max_current)   # gentle first runs
        hand.enable_torque()
        hand.set_neutral_position()               # known open pose; wrist lands at neutral
        # set_neutral_position() restores the config control_mode (current_based_position),
        # which needs a goal-current budget to move — so streamed set_joint_positions didn't
        # track. Put the joints in POSITION mode (what set_neutral used to reach neutral).
        # TODO: for compliant teleop, use current_based_position + set_max_current instead.
        hand.set_control_mode("position")
        print("[orca-follower] torque ON, POSITION mode, at neutral — streaming.", flush=True)
    else:
        print("[orca-follower] DRY RUN (--no-torque): printing targets, no motion.", flush=True)

    if a.wrist:
        print(f"[orca-follower] wrist tracking ON from IK j7 "
              f"(scale={a.wrist_scale}, offset={a.wrist_offset_deg} deg, ROM clamp [-65,35]).", flush=True)

    node = dora.Node()
    n_sent = 0
    wrist_deg = None          # latest wrist target (deg); None until the first IK sample
    wrist_input = f"position_{a.side}"   # IK output carrying this side's j7_wrist
    try:
        for event in node:
            if event["type"] != "INPUT":
                continue
            eid = event["id"]

            # ---- IK j7_wrist -> ORCA wrist (only when --wrist) ----
            if eid == wrist_input:
                if a.wrist:
                    q = _extract_qpos(event["value"])
                    if q.shape[0] >= 7:
                        # IK j7 is RADIANS (model convention); ORCA wrist is DEGREES.
                        wrist_deg = a.wrist_scale * math.degrees(float(q[6])) + a.wrist_offset_deg
                continue

            if eid != a.fingers_input:
                continue

            # ---- fingers ----
            rec = event["value"].to_pylist()[0]
            targets = {}
            for n, v in zip(rec["names"], rec["q"]):
                oj = model_to_orca.get(n)
                if oj is not None:
                    targets[oj] = math.degrees(float(v))   # rad -> deg (orca_core is degrees)
            if a.wrist and wrist_deg is not None:
                targets["wrist"] = wrist_deg                # set_joint_positions clamps to ROM [-65,35]

            if a.no_torque:
                print(f"[orca-follower] (dry) wrist={wrist_deg} {targets}", flush=True)
                continue
            hand.set_joint_positions(OrcaJointPositions.from_dict(targets))
            n_sent += 1
            if n_sent % 30 == 0:                    # ~1 Hz heartbeat so live mode isn't silent
                print(f"[orca-follower] sent {n_sent} cmds "
                      f"(index_mcp={targets.get('index_mcp', float('nan')):.1f} deg, "
                      f"wrist={targets.get('wrist', float('nan')):.1f} deg)", flush=True)
    finally:
        print("[orca-follower] stopping (torque OFF)...", flush=True)
        try:
            hand.disable_torque()
            hand.disconnect()
        except Exception as exc:  # best-effort cleanup
            print(f"[orca-follower] cleanup error: {exc}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="ORCA hand follower (dora sink)")
    p.add_argument("--orca-config", required=True,
                   help="orca_core hand config yaml (calibration + motor/joint map)")
    p.add_argument("--side", choices=("left", "right"), default="left")
    p.add_argument("--max-current", type=float, default=None,
                   help="cap servo current (A) for gentle first runs")
    p.add_argument("--no-torque", action="store_true",
                   help="dry run: print targets, never enable torque / move")
    p.add_argument("--wrist", action="store_true",
                   help="also drive the ORCA wrist from the IK j7_wrist (position_left[6])")
    p.add_argument("--wrist-scale", type=float, default=1.0,
                   help="multiply deg(j7) — use -1.0 to flip the wrist direction")
    p.add_argument("--wrist-offset-deg", type=float, default=0.0,
                   help="offset (deg) added to the wrist target")
    p.add_argument("--fingers-input", default="fingers_right",
                   help="dora input id carrying this hand's fingers (bimanual: fingers_left / fingers_right)")
    _run(p.parse_args())


if __name__ == "__main__":
    main()
