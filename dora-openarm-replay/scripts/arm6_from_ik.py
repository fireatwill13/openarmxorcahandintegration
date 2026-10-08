# Copyright 2026 AprilRoboticsAI
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Dora adapter: orcabot IK `position_left` (8) -> 6-motor follower `move_position`.

The orcabot_yaw IK solves a 7-joint LEFT chain (OpenArm J1-6 + the ORCA wrist-flex
renamed to joint7) and appends a gripper slot, so `position_left` is length 8:

    [j1, j2, j3, j4, j5, j6, j7_wrist, gripper]

On the ORCABOT hardware only J1-6 are CAN motors (can1). Joint7 (wrist-flex) lives
on the ORCA hand's Feetech bus and there is no gripper. This node forwards ONLY the
six CAN joints, so the 6-motor follower's alignment math (element-wise against a
6-value fetch_position) stays length-consistent. Drop-in between `ik` and the
`dora-openarm` follower.
"""

from __future__ import annotations

import argparse
import time

import dora
import numpy as np
import pyarrow as pa

QPOS_TYPE = pa.struct({"qpos": pa.list_(pa.float32())})


def _extract(value: pa.Array, key: str) -> np.ndarray:
    if pa.types.is_struct(value.type):
        value = value.field(key)[0].values
    return np.array(value, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", choices=("left", "right"), default="left",
                    help="which IK output to forward: position_left (default) / position_right")
    args = ap.parse_args()
    expected = f"position_{args.side}"   # this arm's IK output name

    node = dora.Node()
    n = 0
    for event in node:
        if event["type"] != "INPUT" or event["id"] != expected:
            continue
        q = _extract(event["value"], "qpos")
        if q.shape[0] < 6:
            print(f"[arm6] expected >=6 qpos, got {q.shape}; skipping", flush=True)
            continue
        six = np.ascontiguousarray(q[:6], dtype=np.float32)
        node.send_output(
            "move_position",
            pa.array([{"qpos": six}], type=QPOS_TYPE),
            {"timestamp": time.time_ns()},
        )
        n += 1
        if n % 90 == 0:  # ~ every 3s at 30 Hz
            print(f"[arm6] forwarded {n} cmds; last J1-6="
                  f"{[round(float(x), 3) for x in six]}", flush=True)


if __name__ == "__main__":
    main()
