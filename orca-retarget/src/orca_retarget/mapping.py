# Copyright 2026 AprilRoboticsAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Semantic ORCA joint names -> orcabot_yaw model joint names.

The retargeter emits OrcaJointPositions keyed by semantic names (thumb_cmc,
index_pip, ...). The orcabot_yaw MJCF/URDF carries the upstream mesh-hash
joint names prefixed per side (orcahand_right_P-AP_..._to_R-Carpals_...).
The vendored semantic URDFs are document-order-preserving renames of the
upstream orcahand_description v2 URDFs, so the mapping is built by pairing
revolute joints in document order — and VERIFIED by requiring identical
axis and limits on every pair (fails loud on any drift between the files).

The wrist is excluded: in orcabot_yaw it is openarm_{side}_joint7, owned by
the arm IK.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "assets")
_ORCAHAND_DESC = os.environ.get(
    "ORCAHAND_DESCRIPTION_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), "orcahand_description"),
)


def _revolute_joints(urdf_path: str) -> list[dict]:
    out = []
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        if j.get("type") != "revolute":
            continue
        lim = j.find("limit")
        axis = j.find("axis")
        out.append({
            "name": j.get("name"),
            "axis": np.array([float(x) for x in axis.get("xyz").split()]),
            "lower": float(lim.get("lower")),
            "upper": float(lim.get("upper")),
        })
    return out


def semantic_to_model_joints(side: str) -> dict[str, str]:
    """{semantic_name_without_side_prefix: orcabot_yaw_joint_name}, wrist excluded."""
    semantic = _revolute_joints(
        os.path.join(_ASSETS, f"orcahand_{side}_semantic_v2.urdf"))
    upstream = _revolute_joints(
        os.path.join(_ORCAHAND_DESC, "v2", "models", "urdf", f"orcahand_{side}.urdf"))
    if len(semantic) != len(upstream):
        raise RuntimeError(
            f"joint count mismatch: semantic {len(semantic)} vs upstream {len(upstream)}")

    mapping: dict[str, str] = {}
    for s, u in zip(semantic, upstream):
        if not (np.allclose(s["axis"], u["axis"], atol=1e-9)
                and abs(s["lower"] - u["lower"]) < 1e-9
                and abs(s["upper"] - u["upper"]) < 1e-9):
            raise RuntimeError(
                f"joint pairing mismatch: {s['name']} vs {u['name']} "
                "(axis/limits differ — the URDF variants have drifted)")
        short = s["name"].removeprefix(f"{side}_")
        if short == "wrist":
            continue  # owned by arm IK (openarm_{side}_joint7)
        mapping[short] = f"orcahand_{side}_{u['name']}"
    if len(mapping) != 16:
        raise RuntimeError(f"expected 16 finger joints, got {len(mapping)}")
    return mapping
