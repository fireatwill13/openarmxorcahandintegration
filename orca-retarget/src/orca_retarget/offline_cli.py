# Copyright 2026 AprilRoboticsAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline finger retargeting: MCAP hand keypoints -> .fingers npz sidecar.

Reads /hand/{side}/fused_keypoints from a DataInfra recording, retargets each
frame with the adaptive_analytical backend, and writes a sidecar consumed by
the dora replay pipeline (dora-openarm-replay):

  t_ns          (N,)   absolute MCAP log times [ns]
  q             (N,16) finger joint positions [rad]
  joint_names   (16,)  orcabot_yaw model joint names (wrist excluded)

Run (in this package's venv — torch/pinocchio/nlopt live here, NOT in the
dora venv):
  uv run python -m orca_retarget.offline_cli \
    --mcap ~/based/DataInfra/recordings/session_2026-07-22_15-47-57_glove1_raw.fused.mcap
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from mcap.reader import make_reader

from orca_retarget.bridge import FingerRetargeter
from orca_retarget.mapping import semantic_to_model_joints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--side", default="right", choices=("left", "right"))
    parser.add_argument("--out", default=None,
                        help="default: <mcap basename>.fingers_<side>.npz")
    args = parser.parse_args()

    out_path = args.out or args.mcap.replace(".mcap", f".fingers_{args.side}.npz")
    topic = f"/hand/{args.side}/fused_keypoints"

    rt = FingerRetargeter(args.side)
    mapping = semantic_to_model_joints(args.side)
    semantic_order = list(mapping.keys())
    model_names = [mapping[k] for k in semantic_order]

    t_ns, rows, n_cal = [], [], 0
    with open(args.mcap, "rb") as f:
        reader = make_reader(f)
        for _, _, msg in reader.iter_messages(topics=[topic]):
            kp = np.array(json.loads(msg.data)["data"])
            joints = rt.retarget(kp)
            if joints is None:
                n_cal += 1
                continue
            rows.append([np.deg2rad(joints[k]) for k in semantic_order])
            t_ns.append(msg.log_time)

    if not rows:
        raise SystemExit(f"no retargeted frames from {topic}")
    q = np.array(rows, dtype=np.float32)
    np.savez_compressed(
        out_path,
        t_ns=np.array(t_ns, dtype=np.int64),
        q=q,
        joint_names=np.array(model_names),
        meta=json.dumps({
            "mcap": args.mcap, "topic": topic, "side": args.side,
            "backend": "adaptive_analytical", "calibration_frames": n_cal,
            "units": "radians",
        }),
    )
    dur = (t_ns[-1] - t_ns[0]) / 1e9
    print(f"[retarget] {len(rows)} frames over {dur:.1f}s "
          f"({n_cal} calibration frames skipped)")
    print(f"[retarget] q range [{q.min():.2f}, {q.max():.2f}] rad")
    print(f"[retarget] wrote {out_path}")


if __name__ == "__main__":
    main()
