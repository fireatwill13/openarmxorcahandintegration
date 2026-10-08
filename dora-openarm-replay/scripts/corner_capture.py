"""Calibration capture node: log the right arm's measured joint positions when
the right-A button is pressed.

Used for world(AprilTag)->robot-body frame calibration: teleop the right
gripper tip to each of the 4 tag corners and press A at each. The follower owns
the CAN bus, so we read the MEASURED joint vector from follower-right/state
(its `qpos` field, model space, 8 values = 7 arm joints + gripper) rather than
touching the driver directly.

Captures are appended to ../config/corner_captures.json (resolved from this
file's location, so independent of the node's working directory). Do full
4-corner passes in a consistent order; solve_frame.py averages repeated passes.
A is edge-triggered (one capture per press).

Local workspace node (April Robotics) — not part of upstream enactic.
"""

import json
import os

import numpy as np
import pyarrow as pa  # noqa: F401  (kept for parity with sibling nodes)

import dora

OUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "corner_captures.json")
)


def main() -> None:
    node = dora.Node()
    latest_qpos = None
    captures = []
    a_prev = False

    for event in node:
        if event["type"] != "INPUT":
            continue

        event_id = event["id"]
        if event_id == "state":
            # follower state: length-1 StructArray with a "qpos" list field.
            latest_qpos = np.asarray(
                event["value"].field("qpos")[0].as_py(), dtype=float
            )
        elif event_id == "capture":
            a_now = bool(event["value"][0].as_py())
            if a_now and not a_prev and latest_qpos is not None:
                captures.append(latest_qpos.tolist())
                with open(OUT_PATH, "w") as f:
                    json.dump(
                        {"unit": "rad", "space": "model", "captures": captures},
                        f,
                        indent=2,
                    )
                print(
                    f"[corner-capture] #{len(captures)} saved: "
                    f"{np.array2string(latest_qpos, precision=4)} -> {OUT_PATH}",
                    flush=True,
                )
            a_prev = a_now


if __name__ == "__main__":
    main()
