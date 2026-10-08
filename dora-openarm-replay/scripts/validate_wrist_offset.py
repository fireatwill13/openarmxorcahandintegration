# Copyright 2026 AprilRoboticsAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline full-clip IK validation of a wrist-replay frame offset.

Replays an MCAP wrist trajectory through openarm_control.Kinematics with the
exact solver parameters of a replay dataflow yaml (no dora, no viewer) and
reports position/orientation tracking error over the whole clip. Use this
before trusting any new --rot-offset-euler or model: single-pose convergence
tests hide trajectory infeasibility (memory lesson, 2026-07-21).

Example (orcabot_yaw scene, right carpals frame):
  uv run python scripts/validate_wrist_offset.py \
    --mcap ~/based/DataInfra/recordings/session_2026-07-22_15-47-57_glove1_raw.fused.mcap \
    --xml models/orcabot_yaw/scene.xml \
    --frame-right orcahand_right_R-Carpals_8d1f1041 \
    --rot-offset-euler -50.307 85.797 156.518
"""

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcap_wrist_replay import load_samples  # noqa: E402

from openarm_control.config import ArmSetup  # noqa: E402
from openarm_control.kinematics import IKParams, Kinematics  # noqa: E402
from openarm_control.poses import read_ee_pose  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mcap", required=True)
parser.add_argument("--topic", default="/hand/right/fused_wrist_pose_world")
parser.add_argument("--start-sec", type=float, default=0.0)
parser.add_argument("--end-sec", type=float, default=float("inf"))
parser.add_argument(
    "--calibration", default=os.path.join(REPO, "config", "frame_calibration.json")
)
parser.add_argument("--rot-offset-euler", nargs=3, type=float, required=True)
parser.add_argument("--xml", default=os.path.join(REPO, "models/orcabot_yaw/scene.xml"))
parser.add_argument("--frame-right", default="orcahand_right_R-Carpals_8d1f1041")
parser.add_argument("--frame-type-right", default="body")
parser.add_argument("--gripper", type=float, default=0.0)
# IK params mirroring dataflow yaml args:
parser.add_argument("--max-iters", type=int, default=10)
parser.add_argument("--dt", type=float, default=0.1)
parser.add_argument("--damping", type=float, default=0.1)
parser.add_argument("--posture-cost", type=float, default=0.01)
parser.add_argument("--lm-damping", type=float, default=0.01)
parser.add_argument("--orientation-cost", type=float, default=1.0)
parser.add_argument("--pre-roll-cycles", type=int, default=150)
args = parser.parse_args()

samples, _ = load_samples(args)
print(f"[validate] {len(samples)} samples, {samples[-1][0]:.1f}s clip")

setup = ArmSetup.from_args(
    xml=args.xml,
    mode="right",
    frame_right=args.frame_right,
    frame_type_right=args.frame_type_right,
    frame_left=args.frame_right,  # unused in right mode
    frame_type_left=args.frame_type_right,
    keyframe="home",
)
kin = Kinematics(
    setup,
    IKParams(
        max_iters=args.max_iters,
        dt=args.dt,
        damping=args.damping,
        posture_cost=args.posture_cost,
        lm_damping=args.lm_damping,
        orientation_cost=args.orientation_cost,
    ),
)

# pre-roll: settle on the first pose (mirrors the replayer's hold)
for _ in range(args.pre_roll_cycles):
    kin.set_target("right", samples[0][1][:7])
    kin.solve()

pos_err, ori_err, fails = [], [], 0
for rel_t, pose8 in samples:
    kin.set_target("right", pose8[:7])
    values16 = kin.solve()
    if values16 is None:
        fails += 1
        continue
    # solve() only advances the solver's internal mink Configuration; apply
    # the returned driver values to setup.data (as the viewer node does)
    # before FK evaluation.
    setup.joint_resolver.set_qpos(setup.data.qpos, values16[:8], "right")
    mujoco.mj_forward(setup.model, setup.data)
    ee = read_ee_pose(setup.data, setup.frame_ids["right"], setup.frame_types["right"])
    pos_err.append(np.linalg.norm(ee[:3] - pose8[:3]) * 1e3)
    q_t = Rotation.from_quat([*pose8[4:7], pose8[3]])  # xyzw
    q_e = Rotation.from_quat([*ee[4:7], ee[3]])
    ori_err.append(np.rad2deg((q_t.inv() * q_e).magnitude()))

pos_err, ori_err = np.array(pos_err), np.array(ori_err)
t = np.array([s[0] for s in samples])
print(
    f"[validate] position error mm: mean {pos_err.mean():.1f}  "
    f"median {np.median(pos_err):.1f}  p95 {np.percentile(pos_err, 95):.1f}  "
    f"max {pos_err.max():.1f}"
)
print(
    f"[validate] orientation error deg: mean {ori_err.mean():.2f}  "
    f"median {np.median(ori_err):.2f}  p95 {np.percentile(ori_err, 95):.2f}  "
    f"max {ori_err.max():.2f}"
)
if fails:
    print(f"[validate] WARNING: {fails} solve failures")
bad = pos_err > 30
if bad.any():
    # report contiguous bad segments
    idx = np.where(bad)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    segs = np.split(idx, splits + 1)
    print(f"[validate] segments with >30mm position error:")
    for s in segs:
        print(
            f"  t={t[s[0]]:.1f}-{t[s[-1]]:.1f}s  "
            f"max {pos_err[s].max():.0f}mm"
        )
