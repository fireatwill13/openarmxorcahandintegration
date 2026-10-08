# Copyright 2026 AprilRoboticsAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dora source node: LIVE Quest hand-tracking -> orcabot_yaw arm + hand teleop.

Live drop-in replacement for mcap_wrist_replay.py. Instead of replaying a
recorded wrist trajectory + a precomputed finger sidecar, this reads Meta Quest
hand tracking over WebXR (orca_teleop's QuestTelemetryBridge, port 8765) and
emits the SAME two streams, so it plugs straight into the yaw-hands graph with
`ik` and `mujoco_viewer_hand.py` unchanged:

  - pose_right    : struct{"pose":[x,y,z, qw,qx,qy,qz, gripper]}  -> arm IK target
  - fingers_right : struct{"names":[...], "q":[...16 rad...]}     -> hand viewer

No controllers: one bare, tracked hand drives BOTH the arm (its wrist pose ->
arm IK) and the fingers (its landmarks -> finger retargeter).

The finger path mirrors orca-retarget/offline_cli EXACTLY — FingerRetargeter ->
{semantic: degrees} -> semantic_to_model_joints names + np.deg2rad — only the
source differs (live WebXR landmarks vs DataInfra MCAP keypoints). We feed the
retargeter MediaPipe-ordered landmarks; FingerRetargeter.retarget() internally
reindexes DataInfra->MediaPipe, so we inverse-permute first (identity round-trip)
to reuse it byte-for-byte without touching bridge.py.

=============================================================================
TODO(hardware): the WRIST->arm mapping (--frame-offset / --rot-fix-euler /
--position-scale) needs tuning for the Quest-wrist -> orcabot R-Carpals frame.
The replayer's rot-offset (-50.307 85.797 156.518) was calibrated for the
DataInfra tracker frame, which is NOT the WebXR wrist frame — so expect to
re-tune, same as the controller receiver's frame args were tuned for v1.
Defaults below are the v1 controller starting guess, not a solved offset.
=============================================================================
"""

from __future__ import annotations

import argparse
import time

import dora
import numpy as np
import pyarrow as pa
from scipy.spatial.transform import Rotation

POSE_TYPE = pa.struct({"pose": pa.list_(pa.float32())})
FINGERS_TYPE = pa.struct(
    {"names": pa.list_(pa.string()), "q": pa.list_(pa.float32())}
)

_DEFAULT_QUEST_PORT = 8765


class WristCalibrator:
    """Quest wrist pose (robot-FLU frame) -> arm EE target, via a fixed R_fix
    calibrated against the robot's HOME EE pose.

    Convention matches the replay path (`R_eef = R_wrist * R_offset`):
        R_ee(t) = R_wrist(t) * R_fix,   R_fix = R_wrist_cal^-1 * R_ee_home
        p_ee(t) = p_ee_home + scale * (p_wrist(t) - p_wrist_cal)

    R_fix and p_wrist_cal are captured ONCE, on the first frame — so hold your
    right hand in a comfortable neutral pose at startup: that pose becomes
    "robot home". Thereafter world-frame wrist rotations/translations transfer
    1:1 to the EE. (Assumes the IK origin-frame is world, i.e. openarm_body_link0
    is at identity — verified for orcabot_yaw.)
    """

    def __init__(self, r_ee_home: Rotation, p_ee_home: np.ndarray,
                 position_scale: float = 1.0, max_reach: float = 0.25):
        self._r_ee_home = r_ee_home
        self._p_ee_home = np.asarray(p_ee_home, dtype=np.float64)
        self._scale = position_scale
        self._max_reach = max_reach          # m; clamp EE within this radius of home
        self._r_fix = None
        self._p_wrist_cal = None

    @property
    def calibrated(self) -> bool:
        return self._r_fix is not None

    def process(self, wrist_matrix) -> np.ndarray:
        m = np.asarray(wrist_matrix, dtype=np.float64).reshape(4, 4)
        p_w = m[:3, 3].copy()
        r_w = Rotation.from_matrix(m[:3, :3])
        if self._r_fix is None:                      # one-shot calibration
            self._r_fix = r_w.inv() * self._r_ee_home
            self._p_wrist_cal = p_w
        r_ee = r_w * self._r_fix                      # R_ee = R_wrist * R_fix
        p_ee = self._p_ee_home + self._scale * (p_w - self._p_wrist_cal)
        if self._max_reach and self._max_reach > 0.0:   # keep inside the workspace so
            off = p_ee - self._p_ee_home                 # you can't drive the arm into
            d = float(np.linalg.norm(off))               # its reach-boundary singularity
            if d > self._max_reach:
                p_ee = self._p_ee_home + off * (self._max_reach / d)
        q = r_ee.as_quat()  # x, y, z, w
        return np.array([p_ee[0], p_ee[1], p_ee[2], q[3], q[0], q[1], q[2]],
                        dtype=np.float32)


def _run(args: argparse.Namespace) -> None:
    # Heavy deps loaded lazily (torch/pinocchio/nlopt/aiohttp) — keep import
    # failures out of module load so `dora build` errors are legible.
    from orca_retarget.bridge import DATAINFRA_TO_MEDIAPIPE, FingerRetargeter
    from orca_retarget.mapping import semantic_to_model_joints
    from orca_teleop.ingress.metaquest.bridge import QuestTelemetryBridge
    from orca_teleop.ingress.metaquest.landmarks import retargeter_landmarks_from_webxr

    side = args.hand

    # --- finger retargeter (same construction as offline_cli) ----------------
    rt = FingerRetargeter(side)
    mapping = semantic_to_model_joints(side)          # {semantic_short: model_name}
    semantic_order = list(mapping.keys())
    model_names = [mapping[k] for k in semantic_order]
    inv_perm = np.argsort(DATAINFRA_TO_MEDIAPIPE)     # mediapipe-order -> datainfra-order

    # --- wrist -> arm EE: calibrate R_fix from the robot's HOME EE pose ------
    import mujoco
    model = mujoco.MjModel.from_xml_path(args.xml)
    mjdata = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, mjdata, kid)
    mujoco.mj_forward(model, mjdata)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.ee_body)
    if bid < 0:
        raise ValueError(f"--ee-body {args.ee_body!r} not found in {args.xml}")
    p_ee_home = mjdata.xpos[bid].copy()
    q_wxyz = mjdata.xquat[bid].copy()
    r_ee_home = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])  # wxyz->xyzw
    print(f"[quest-live] home EE '{args.ee_body}': p={p_ee_home} quat_wxyz={q_wxyz}", flush=True)
    wrist_proc = WristCalibrator(r_ee_home, p_ee_home, args.position_scale, args.max_reach)

    # --- Quest WebXR ingress -------------------------------------------------
    # QuestTelemetryBridge takes (host, port, ssl_context=None). For a local
    # run over `adb reverse` (http://localhost:8765 = secure context on Quest)
    # no TLS is needed; build an ssl_context only if a cert/key were passed.
    ssl_context = None
    if args.ssl_cert and args.ssl_key:
        import ssl
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(certfile=args.ssl_cert, keyfile=args.ssl_key)
    bridge = QuestTelemetryBridge(
        host=args.quest_host, port=args.quest_port, ssl_context=ssl_context)
    bridge.start()
    print(f"[quest-live] WebXR page at {bridge.url} — open it in Quest Browser, "
          f"tap 'Start hand tracking' (needs HTTPS or "
          f"`adb reverse tcp:{args.quest_port} tcp:{args.quest_port}`).", flush=True)

    node = dora.Node()
    last_seq = None
    calibrating_logged = False
    ticks = samples = sent = 0

    try:
        for event in node:
            if event["type"] != "INPUT" or event["id"] != "tick":
                continue
            ticks += 1
            sample = bridge.state.get_hand_sample(side)
            if ticks % 60 == 0:  # heartbeat (~2s at 30 Hz) so "is it arriving?" is visible
                if samples == 0:
                    print(f"[quest-live] tick {ticks}: NO hand frames from the Quest yet — "
                          "the headset must be WORN (tape the proximity sensor), your RIGHT "
                          "hand in the cameras' view, and controllers down.", flush=True)
                else:
                    print(f"[quest-live] tick {ticks}: frames={samples} sent={sent} — streaming.",
                          flush=True)
            if sample is None or sample.sequence_id == last_seq:
                continue
            samples += 1
            last_seq = sample.sequence_id
            ts = {"timestamp": time.time_ns()}

            # ---- arm: wrist pose -> pose_right ----
            if sample.wrist_matrix is not None:
                pose7 = wrist_proc.process(sample.wrist_matrix)
                pose8 = np.concatenate([pose7, [np.float32(args.gripper)]])
                node.send_output(
                    "pose_right", pa.array([{"pose": pose8}], type=POSE_TYPE), ts)

            # ---- hand: landmarks -> fingers_right ----
            mp21 = retargeter_landmarks_from_webxr(sample.landmarks, side)  # (21,3) mp order
            joints = rt.retarget(mp21[inv_perm])   # rt re-applies DATAINFRA_TO_MEDIAPIPE -> mp21
            if joints is None:
                if not calibrating_logged:
                    print("[quest-live] retargeter calibrating — hold your hand "
                          "still for ~0.5s...", flush=True)
                    calibrating_logged = True
                continue
            q = np.array([np.deg2rad(joints[k]) for k in semantic_order],
                         dtype=np.float32)
            node.send_output(
                "fingers_right",
                pa.array([{"names": model_names, "q": q}], type=FINGERS_TYPE), ts)
            sent += 1
    finally:
        bridge.stop()


def _run_both(args: argparse.Namespace) -> None:
    """--hand both: emit pose_left/right + fingers_left/right from ONE WebXR server.

    The bridge already receives both hands (hands=['left','right']); we build the
    per-hand machinery (retargeter + mapping + wrist-home) for each side and fan out.
    The single-hand _run above is intentionally left untouched.
    """
    from orca_retarget.bridge import DATAINFRA_TO_MEDIAPIPE, FingerRetargeter
    from orca_retarget.mapping import semantic_to_model_joints
    from orca_teleop.ingress.metaquest.bridge import QuestTelemetryBridge
    from orca_teleop.ingress.metaquest.landmarks import retargeter_landmarks_from_webxr
    import mujoco

    inv_perm = np.argsort(DATAINFRA_TO_MEDIAPIPE)
    model = mujoco.MjModel.from_xml_path(args.xml)
    mjdata = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, mjdata, kid)
    mujoco.mj_forward(model, mjdata)

    ee_bodies = {"left": args.ee_body_left, "right": args.ee_body_right}
    channels = []
    for side in ("left", "right"):
        eb = ee_bodies[side]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eb)
        if bid < 0:
            raise ValueError(f"ee-body {eb!r} ({side}) not found in {args.xml}")
        q_wxyz = mjdata.xquat[bid].copy()
        r_ee_home = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        wrist_proc = WristCalibrator(r_ee_home, mjdata.xpos[bid].copy(),
                                     args.position_scale, args.max_reach)
        mapping = semantic_to_model_joints(side)
        print(f"[quest-live] home EE '{eb}' ({side})", flush=True)
        channels.append({
            "side": side,
            "rt": FingerRetargeter(side),
            "semantic_order": list(mapping.keys()),
            "model_names": [mapping[k] for k in mapping.keys()],
            "wrist_proc": wrist_proc,
            "last_seq": None,
            "calib": False,
        })

    ssl_context = None
    if args.ssl_cert and args.ssl_key:
        import ssl
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(certfile=args.ssl_cert, keyfile=args.ssl_key)
    bridge = QuestTelemetryBridge(host=args.quest_host, port=args.quest_port,
                                  ssl_context=ssl_context)
    bridge.start()
    print(f"[quest-live] BIMANUAL — WebXR page at {bridge.url}; open in Quest Browser, "
          f"'Start hand tracking' (adb reverse tcp:{args.quest_port} tcp:{args.quest_port}). "
          f"BOTH hands in view.", flush=True)

    node = dora.Node()
    ticks = 0
    samples = {"left": 0, "right": 0}
    sent = {"left": 0, "right": 0}
    try:
        for event in node:
            if event["type"] != "INPUT" or event["id"] != "tick":
                continue
            ticks += 1
            if ticks % 60 == 0:
                print(f"[quest-live] tick {ticks}: "
                      f"L frames={samples['left']} sent={sent['left']} | "
                      f"R frames={samples['right']} sent={sent['right']} "
                      f"(0 => wear headset, both hands in view, controllers down)", flush=True)
            ts = {"timestamp": time.time_ns()}
            for c in channels:
                side = c["side"]
                sample = bridge.state.get_hand_sample(side)
                if sample is None or sample.sequence_id == c["last_seq"]:
                    continue
                samples[side] += 1
                c["last_seq"] = sample.sequence_id
                if sample.wrist_matrix is not None:
                    pose7 = c["wrist_proc"].process(sample.wrist_matrix)
                    pose8 = np.concatenate([pose7, [np.float32(args.gripper)]])
                    node.send_output(f"pose_{side}",
                                     pa.array([{"pose": pose8}], type=POSE_TYPE), ts)
                mp21 = retargeter_landmarks_from_webxr(sample.landmarks, side)
                joints = c["rt"].retarget(mp21[inv_perm])
                if joints is None:
                    if not c["calib"]:
                        print(f"[quest-live] retargeter calibrating ({side}) — hold still ~0.5s",
                              flush=True)
                        c["calib"] = True
                    continue
                q = np.array([np.deg2rad(joints[k]) for k in c["semantic_order"]], dtype=np.float32)
                node.send_output(f"fingers_{side}",
                                 pa.array([{"names": c["model_names"], "q": q}], type=FINGERS_TYPE), ts)
                sent[side] += 1
    finally:
        bridge.stop()


def main() -> None:
    import logging
    # Surface orca_teleop's bridge INFO logs ("WebSocket opened", "telemetry
    # packet N: hands=[...]") — without this they're below the default level and
    # invisible, so you can't tell if the Quest is actually connecting/streaming.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s | %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Live Quest hand-tracking source (arm + fingers)")
    p.add_argument("--hand", choices=("left", "right", "both"), default="right")
    p.add_argument("--quest-host", default="0.0.0.0")
    p.add_argument("--quest-port", type=int, default=_DEFAULT_QUEST_PORT)
    p.add_argument("--ssl-cert", default=None)
    p.add_argument("--ssl-key", default=None)
    p.add_argument("--gripper", type=float, default=0.0,
                   help="fixed value for the pose's 8th (gripper) slot; orcabot's "
                        "hand replaces the gripper, so keep 0")
    # R_fix is now calibrated from the model's HOME EE pose (no guessed offsets).
    p.add_argument("--xml", default="../models/orcabot_yaw/scene.xml",
                   help="MJCF scene, for FK of the home EE pose R_fix calibrates against")
    p.add_argument("--keyframe", default="home")
    p.add_argument("--ee-body", default="orcahand_right_R-Carpals_8d1f1041",
                   help="EE body for the home pose (match the IK --frame-right)")
    p.add_argument("--ee-body-left", default="orcahand_left_L-Carpals_719fff8c",
                   help="left EE body for --hand both")
    p.add_argument("--ee-body-right", default="orcahand_right_R-Carpals_8d1f1041",
                   help="right EE body for --hand both")
    p.add_argument("--position-scale", type=float, default=1.0)
    p.add_argument("--max-reach", type=float, default=0.25,
                   help="clamp the EE target within this radius (m) of home so you "
                        "can't drive the arm into its reach-boundary singularity")
    args = p.parse_args()
    (_run_both if args.hand == "both" else _run)(args)


if __name__ == "__main__":
    main()
