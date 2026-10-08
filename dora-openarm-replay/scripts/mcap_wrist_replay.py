# Copyright 2026 AprilRoboticsAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dora source node: replay a recorded wrist trajectory from MCAP as IK targets.

Reads Pose messages (4x4 matrix "T", wrist tracker in the recording's world
frame) from an MCAP file, maps them into the robot base frame via an AprilTag
frame calibration (T_tag_to_body), removes the fixed eef->tracker rotation
offset, and publishes `pose_right` targets ([x y z qw qx qy qz gripper]) with
the recording's original pacing.
"""

import argparse
import json
import os
import time

import dora
import numpy as np
import pyarrow as pa
from mcap.reader import make_reader
from scipy.spatial.transform import Rotation

POSE_TYPE = pa.struct({"pose": pa.list_(pa.float32())})
FINGERS_TYPE = pa.struct(
    {"names": pa.list_(pa.string()), "q": pa.list_(pa.float32())}
)


def load_finger_events(path: str, t0_abs_ns: int, clip_end_s: float):
    """Load a .fingers_<side>.npz sidecar (see orca-retarget offline_cli) and
    return [(rel_t, names, q16)] aligned to the wrist clip's time base."""
    data = np.load(path, allow_pickle=False)
    names = [str(n) for n in data["joint_names"]]
    events = []
    for t_ns, q in zip(data["t_ns"], data["q"]):
        rel = (int(t_ns) - t0_abs_ns) / 1e9
        if 0.0 <= rel <= clip_end_s:
            events.append((rel, names, q.astype(np.float32)))
    return events


def load_samples(args) -> tuple[list[tuple[float, np.ndarray]], int]:
    cal = json.load(open(args.calibration))
    T_base_world = np.array(cal["T_tag_to_body"])
    r_eef_tracker = Rotation.from_euler("xyz", args.rot_offset_euler, degrees=True)

    samples = []
    with open(args.mcap, "rb") as f:
        reader = make_reader(f)
        t0 = reader.get_summary().statistics.message_start_time
        lo = t0 + int(args.start_sec * 1e9)
        hi = None if args.end_sec == float("inf") else t0 + int(args.end_sec * 1e9)
        for _, _, msg in reader.iter_messages(
            topics=[args.topic], start_time=lo, end_time=hi
        ):
            T_base_wrist = T_base_world @ np.array(json.loads(msg.data)["T"])
            p = T_base_wrist[:3, 3]
            # Offset direction validated empirically (IK converges to <30mm/0.4deg
            # with the forward composition; the inverse collapses to ~800mm).
            r_eef = Rotation.from_matrix(T_base_wrist[:3, :3]) * r_eef_tracker
            q = r_eef.as_quat()  # x, y, z, w
            pose8 = np.array(
                [*p, q[3], q[0], q[1], q[2], args.gripper], dtype=np.float32
            )
            samples.append((msg.log_time, pose8))

    if not samples:
        raise SystemExit(
            f"No messages on {args.topic} in [{args.start_sec}, {args.end_sec}]s"
        )
    first = samples[0][0]
    return [((t - first) / 1e9, pose) for t, pose in samples], first


def main() -> None:
    # Per-run inputs default from environment variables so the dataflow yaml
    # stays machine-independent:
    #   REPLAY_MCAP=/path/to/file.mcap REPLAY_START_SEC=30 REPLAY_END_SEC=40 \
    #     dora run config/dataflow-mujoco-v1-replay.yaml --uv
    # Explicit CLI args in the yaml still take precedence over the env.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", default=os.environ.get("REPLAY_MCAP"))
    parser.add_argument(
        "--topic",
        default=os.environ.get("REPLAY_TOPIC", "/hand/right/fused_wrist_pose_world"),
    )
    parser.add_argument(
        "--start-sec", type=float, default=float(os.environ.get("REPLAY_START_SEC", 0.0))
    )
    parser.add_argument(
        "--end-sec", type=float, default=float(os.environ.get("REPLAY_END_SEC", "inf"))
    )
    parser.add_argument(
        "--calibration",
        default="frame_calibration.json",
        help="JSON with T_tag_to_body (recording world -> robot base)",
    )
    parser.add_argument(
        "--rot-offset-euler",
        nargs=3,
        type=float,
        default=[90.0, 0.0, 45.0],
        metavar=("RX", "RY", "RZ"),
        help="Fixed robot-eef -> tracker rotation, extrinsic xyz degrees",
    )
    parser.add_argument("--gripper", type=float, default=0.022)
    parser.add_argument(
        "--pre-roll-sec",
        type=float,
        default=float(os.environ.get("REPLAY_PRE_ROLL_SEC", 1.5)),
        help="Hold the first pose this long before playback so the arm "
        "(sim IK, or the real follower's creep alignment) settles on it",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restart the clip when it ends (default: on)",
    )
    parser.add_argument(
        "--fingers",
        default=os.environ.get("REPLAY_FINGERS", ""),
        help="Optional .fingers_<side>.npz sidecar (orca-retarget offline_cli); "
        "publishes retargeted finger joints on fingers_right",
    )
    args = parser.parse_args()
    if not args.mcap:
        raise SystemExit("--mcap or REPLAY_MCAP is required")

    samples, t0_abs = load_samples(args)
    print(
        f"[replay] {len(samples)} samples, {samples[-1][0]:.1f}s clip from "
        f"{args.topic} [{args.start_sec}..{args.end_sec}]s",
        flush=True,
    )

    finger_events = []
    if args.fingers:
        finger_events = load_finger_events(args.fingers, t0_abs, samples[-1][0])
        print(f"[replay] {len(finger_events)} finger frames from {args.fingers}",
              flush=True)

    # merged timeline: wrist poses and finger frames at their own timestamps
    events = [(t, "pose", p) for t, p in samples] + [
        (t, "fingers", (n, q)) for t, n, q in finger_events
    ]
    events.sort(key=lambda e: e[0])

    node = dora.Node()

    def send_fingers(names, q):
        node.send_output(
            "fingers_right",
            pa.array([{"names": names, "q": q}], type=FINGERS_TYPE),
            {"timestamp": time.time_ns()},
        )

    while True:
        # Pre-roll: hold the first pose so the incremental IK (and, on the
        # real robot, the follower's creep alignment) settles on it.
        if finger_events:
            send_fingers(finger_events[0][1], finger_events[0][2])
        for _ in range(int(args.pre_roll_sec * 100)):
            node.send_output(
                "pose_right",
                pa.array([{"pose": samples[0][1]}], type=POSE_TYPE),
                {"timestamp": time.time_ns()},
            )
            time.sleep(0.01)
        t_start = time.monotonic()
        for rel_t, kind, payload in events:
            delay = t_start + rel_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if kind == "pose":
                node.send_output(
                    "pose_right",
                    pa.array([{"pose": payload}], type=POSE_TYPE),
                    {"timestamp": time.time_ns()},
                )
            else:
                send_fingers(payload[0], payload[1])
        print("[replay] clip done", flush=True)
        if not args.loop:
            break
        time.sleep(2.0)


if __name__ == "__main__":
    main()
