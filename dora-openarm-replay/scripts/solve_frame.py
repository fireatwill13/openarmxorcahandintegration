"""Solve the world(AprilTag) -> robot-body rigid transform from touched corners.

Inputs:
  - ../config/corner_captures.json (from corner_capture.py): measured right-arm
    joint vectors, one per corner touch, in a consistent order. Repeated
    4-corner passes are averaged.
  - --tag-size S: the physical edge length (metres) between the ADJACENT corners
    you touched. Measure it on the real tag.

Pipeline:
  1. mujoco forward kinematics on each joint vector -> openarm_right_hand_tcp
     position in the MJCF base = robot body frame.  -> body points B (4x3)
  2. Known corner coords in the tag frame (origin = tag centre, +x right,
     +y up, +z out of the face), matching the touch order below. -> world W (4x3)
  3. Umeyama / Kabsch (no scale) best-fit R,t with  B ~= R @ W + t.
     T_tag_to_body maps a point in the tag frame into the robot body frame.

Touch order must match TAG_CORNERS_UNIT below (default: BL, BR, TR, TL).

Usage (from repo root):
    uv run python scripts/solve_frame.py --tag-size 0.10
"""

import argparse
import json
import os

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENE = os.path.join(HERE, "..", "..", "openarm_mujoco_v1", "v1", "scene.xml")
DEFAULT_CAPTURES = os.path.join(HERE, "..", "config", "corner_captures.json")
DEFAULT_OUT = os.path.join(HERE, "..", "config", "frame_calibration.json")
RIGHT_QPOS_SLICE = slice(9, 16)  # 7 right-arm joints in the bimanual MJCF
TCP_BODY = "openarm_right_hand_tcp"

# Corner coords in the tag frame, as unit multiples of (S/2). Touch order = this
# order. Change both together if you prefer a different convention.
TAG_CORNERS_UNIT = np.array(
    [
        [-1.0, -1.0, 0.0],  # 1: bottom-left
        [+1.0, -1.0, 0.0],  # 2: bottom-right
        [+1.0, +1.0, 0.0],  # 3: top-right
        [-1.0, +1.0, 0.0],  # 4: top-left
    ]
)


def fk_right_tcp(model, data, joints7: np.ndarray) -> np.ndarray:
    """Forward-kinematics: right-arm joint angles -> TCP position in body frame."""
    data.qpos[:] = 0.0
    data.qpos[RIGHT_QPOS_SLICE] = joints7[:7]
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TCP_BODY)
    return data.xpos[bid].copy()


def umeyama_no_scale(W: np.ndarray, B: np.ndarray):
    """Best-fit R,t with B ~= R @ W + t (Kabsch, reflection-corrected)."""
    wc, bc = W.mean(0), B.mean(0)
    Wc, Bc = W - wc, B - bc
    H = Wc.T @ Bc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = bc - R @ wc
    return R, t


def load_body_points(captures_path: str, scene_path: str) -> np.ndarray:
    with open(captures_path) as f:
        data = json.load(f)
    caps = np.asarray(data["captures"], dtype=float)
    if caps.ndim != 2 or caps.shape[0] < 4 or caps.shape[0] % 4 != 0:
        raise SystemExit(
            f"need a positive multiple of 4 captures, got {caps.shape[0]}"
        )
    passes = caps.shape[0] // 4
    caps = caps.reshape(passes, 4, -1).mean(axis=0)  # average repeated passes
    model = mujoco.MjModel.from_xml_path(scene_path)
    mjdata = mujoco.MjData(model)
    B = np.array([fk_right_tcp(model, mjdata, c) for c in caps])
    return B, passes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-size", type=float, required=True,
                    help="edge length [m] between adjacent touched corners")
    ap.add_argument("--captures", default=DEFAULT_CAPTURES)
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    B, passes = load_body_points(args.captures, args.scene)
    W = TAG_CORNERS_UNIT * (args.tag_size / 2.0)

    R, t = umeyama_no_scale(W, B)
    B_pred = (R @ W.T).T + t
    err = np.linalg.norm(B - B_pred, axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))

    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    T_inv = np.linalg.inv(T)

    np.set_printoptions(precision=5, suppress=True)
    print(f"passes averaged: {passes}   tag-size: {args.tag_size} m\n")
    print("body-frame corner points (m):\n", B, "\n")
    print("T_tag_to_body (world->body):\n", T, "\n")
    print("per-corner residual (mm):", np.round(err * 1e3, 2))
    print(f"RMS residual: {rms * 1e3:.2f} mm")
    if rms > 0.01:
        print("  !! >10 mm: check touch accuracy, tag-size, corner order, joint zeros")

    with open(args.out, "w") as f:
        json.dump(
            {
                "tag_size_m": args.tag_size,
                "corner_order": "BL,BR,TR,TL",
                "passes_averaged": passes,
                "rms_residual_m": rms,
                "per_corner_residual_m": err.tolist(),
                "T_tag_to_body": T.tolist(),
                "T_body_to_tag": T_inv.tolist(),
            },
            f,
            indent=2,
        )
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
