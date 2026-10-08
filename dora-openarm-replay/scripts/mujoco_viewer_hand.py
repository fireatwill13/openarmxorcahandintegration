# Copyright 2026 AprilRoboticsAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dora sink node: MuJoCo viewer with arm AND finger qpos inputs.

April-local extension of the stock enactic viewer, which can only write
7 arm joints + gripper per side (JointResolver contract). This node adds a
``fingers_right`` input: a self-describing struct {names, q} whose joint
names are resolved against the model once and written directly to qpos.
Kinematic display only (mj_forward, no physics stepping) — same as the
stock viewer's default mode.
"""

import argparse
import time

import dora
import mujoco
import mujoco.viewer
import numpy as np
from openarm_mujoco.v2 import JointResolver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--keyframe", default="home")
    parser.add_argument(
        "--viewer", action=argparse.BooleanOptionalAction, default=True,
        help="--no-viewer runs headless (graph debugging/CI)",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        print(f"Warning: keyframe '{args.keyframe}' not found, using defaults.")
    mujoco.mj_forward(model, data)

    resolver = JointResolver(model)
    finger_adr: dict[str, int] | None = None  # resolved from the first message

    viewer_handle = None
    if args.viewer:
        viewer_handle = mujoco.viewer.launch_passive(model, data)

    node = dora.Node()
    n_fingers = 0
    last_sync = 0.0
    for event in node:
        if event["type"] != "INPUT":
            continue
        eid = event["id"]
        if eid in ("position_right", "position_left"):
            value = event["value"]
            # float32[8], or struct with "qpos" (dora-openarm-ik) /
            # "new_position" (followers) field
            first = value.to_pylist()[0]
            if isinstance(first, dict):
                arr = first.get("qpos", first.get("new_position"))
                values = np.asarray(arr, dtype=np.float64)
            else:
                values = np.asarray(value.to_pylist(), dtype=np.float64)
            resolver.set_qpos(data.qpos, values, eid.removeprefix("position_"))
        elif eid == "fingers_right":
            msg = event["value"][0]
            names = [str(n) for n in msg["names"]]
            q = np.asarray([float(x) for x in msg["q"]], dtype=np.float64)
            if finger_adr is None:
                finger_adr = {}
                for name in names:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid < 0:
                        raise ValueError(f"finger joint '{name}' not in model")
                    finger_adr[name] = int(model.jnt_qposadr[jid])
                print(f"[viewer] resolved {len(finger_adr)} finger joints",
                      flush=True)
            for name, val in zip(names, q):
                data.qpos[finger_adr[name]] = val
            n_fingers += 1
            if not args.viewer and n_fingers % 200 == 0:
                print(f"[viewer] fingers frame {n_fingers}, "
                      f"mean |q| = {np.abs(q).mean():.3f} rad", flush=True)
        else:
            continue

        mujoco.mj_forward(model, data)
        if viewer_handle is not None:
            now = time.monotonic()
            if now - last_sync >= 1 / 60:
                if not viewer_handle.is_running():
                    break
                viewer_handle.sync()
                last_sync = now

    if viewer_handle is not None:
        viewer_handle.close()


if __name__ == "__main__":
    main()
