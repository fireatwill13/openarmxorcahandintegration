# Copyright 2026 AprilRoboticsAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Landmarks -> ORCA finger joints, shared by offline replay and live teleop.

Wraps orca_teleop's adaptive_analytical retargeter with the two conventions
our stack needs:

- DataInfra fused_keypoints use a custom 21-keypoint ordering (wrist=20,
  fingers tip->mcp); the retargeter expects MediaPipe ordering (wrist=0,
  fingers mcp->tip). ``DATAINFRA_TO_MEDIAPIPE`` reindexes.
- Output is the semantic OrcaJointPositions; downstream consumers map those
  names onto a robot model's joint names (see mapping.py). The wrist entry is
  dropped: in the orcabot_yaw embodiment the wrist-flex joint is owned by the
  arm IK (openarm_*_joint7), never by the finger retargeter.
"""

from __future__ import annotations

import os

import numpy as np

from orca_teleop.retargeting.retargeter import Retargeter, TargetPose

# Vendored semantic-name v2 URDFs (retargeter needs right_thumb_cmc-style
# joint names; the orcahand_description repo URDFs carry mesh-hash names).
# No companion MJCF is needed for v2: all 17 `ref` offsets in the upstream
# v2 MJCF are 0.0 (URDF zero == physical zero), so the loader's warning is
# harmless. (v1 is different — nonzero refs — do not reuse this shortcut.)
_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "assets")

# DataInfra skeleton (datainfra.fusion.skeleton.KP_NAMES) -> MediaPipe index:
# mediapipe[i] = datainfra[DATAINFRA_TO_MEDIAPIPE[i]]
DATAINFRA_TO_MEDIAPIPE = np.array(
    [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]
)


def _bundled_hand_model(side: str) -> str:
    """orca_core ships the v2 hand model configs inside the package."""
    import orca_core

    return os.path.join(
        os.path.dirname(os.path.abspath(orca_core.__file__)),
        "models", "v2", f"orcahand_{side}", "config.yaml",
    )


class FingerRetargeter:
    """Adaptive-analytical retargeter for one hand, DataInfra convention in.

    ``retarget()`` returns ``None`` during the auto-scale calibration phase
    (first ~30 frames) and a ``{semantic_joint_name: degrees}`` dict after,
    with the wrist entry removed.
    """

    def __init__(self, side: str, *, urdf_path: str | None = None,
                 model_path: str | None = None, config_path: str | None = None):
        assert side in ("left", "right")
        self.side = side
        self._rt = Retargeter.from_paths(
            backend="adaptive_analytical",
            model_path=model_path or _bundled_hand_model(side),
            urdf_path=urdf_path
            or os.path.join(_ASSETS, f"orcahand_{side}_semantic_v2.urdf"),
            config_path=config_path,
        )

    def retarget(self, kp_datainfra: np.ndarray) -> dict[str, float] | None:
        kp = np.asarray(kp_datainfra, dtype=float)
        if kp.shape != (21, 3):
            raise ValueError(f"expected (21,3) keypoints, got {kp.shape}")
        out = self._rt.retarget(
            TargetPose(joint_positions=kp[DATAINFRA_TO_MEDIAPIPE], source="mediapipe")
        )
        if out is None:
            return None
        joints = dict(out.items()) if hasattr(out, "items") else dict(out)
        joints.pop("wrist", None)  # arm IK owns the wrist-flex joint
        return joints

    def reset(self) -> None:
        self._rt.reset()
