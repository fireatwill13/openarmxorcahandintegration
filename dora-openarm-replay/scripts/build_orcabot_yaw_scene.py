# Copyright 2026 AprilRoboticsAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Build the orcabot_yaw MuJoCo scene for wrist-trajectory replay.

Derives an enactic-pipeline-compatible MJCF from the canonical orcabot_yaw
URDF (orca_arm repo: OpenArm through joint6 + printed adapter + ORCA hand,
7 joints per arm):

- stages the URDF + meshes into models/orcabot_yaw/,
- renames each wrist-flex joint (orcahand carpals) to openarm_{side}_joint7
  so JointResolver sees a standard 7-joint arm,
- adds inert dummy finger_joint1/2 per side (resolver requires them; the ORCA
  fingers are not part of wrist replay and stay frozen by the IK node's
  DofFreezingTask),
- appends the {side}_joint*_ctrl / {side}_finger1_ctrl actuators the resolver
  resolves unconditionally (unused in the viewer's default qpos mode),
- writes scene.xml with floor/light and a bent-elbow "home" keyframe (the
  all-zero pose is fully extended AND parks joint4 on its lower limit — a
  singular start differential IK cannot escape),
- computes the replayer --rot-offset-euler for the right carpals frame by
  composing the V1-calibrated wrist offset with the model-derived
  link7 -> carpals rotation at the elbow-90 correspondence pose.

Run:  uv run python scripts/build_orcabot_yaw_scene.py
"""

import os
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "models", "orcabot_yaw")
ORCA_PKG = "/home/alex/based/orca/orca_arm/orca_arm"
SRC_URDF = os.path.join(ORCA_PKG, "orcabot_yaw_render.urdf")  # .dae-free variant
V1_SCENE = os.path.join(REPO, "..", "openarm_mujoco_v1", "v1", "scene.xml")

# V1-calibrated wrist offset (dataflow-mujoco-v1-replay.yaml), computed against
# openarm_right_link7's attitude at the elbow-90 correspondence pose:
V1_ROT_OFFSET_EULER = [-85.945, -1.109, 81.786]

WRIST_JOINTS = {
    "left": "orcahand_left_L-Carpals_719fff8c_to_TopTower-Model_4a80d30e",
    "right": "orcahand_right_R-Carpals_8d1f1041_to_TopTower-Model_4a80d30e",
}
CARPALS_BODY = {
    "left": "orcahand_left_L-Carpals_719fff8c",
    "right": "orcahand_right_R-Carpals_8d1f1041",
}
ELBOW90 = 1.571  # joint4, matches the V1 correspondence/home convention

os.makedirs(os.path.join(OUT, "assets"), exist_ok=True)

# ── 1. stage URDF + meshes, rename joints, add dummies ───────────────────────
tree = ET.parse(SRC_URDF)
root = tree.getroot()

n_meshes = 0
for mesh in root.iter("mesh"):
    rel = mesh.get("filename")  # "assets/xxx.stl"
    shutil.copy(os.path.join(ORCA_PKG, rel), os.path.join(OUT, rel))
    n_meshes += 1
print(f"[stage] {n_meshes} mesh refs copied into {OUT}/assets")

for side, jname in WRIST_JOINTS.items():
    j = next(j for j in root.findall("joint") if j.get("name") == jname)
    j.set("name", f"openarm_{side}_joint7")

for side in ("left", "right"):
    for i in (1, 2):
        link = ET.SubElement(root, "link")
        link.set("name", f"openarm_{side}_dummy_finger{i}")
        inr = ET.SubElement(link, "inertial")
        ET.SubElement(inr, "origin").set("xyz", "0 0 0")
        ET.SubElement(inr, "mass").set("value", "0.001")
        ii = ET.SubElement(inr, "inertia")
        for k in ("ixx", "iyy", "izz"):
            ii.set(k, "1e-8")
        for k in ("ixy", "ixz", "iyz"):
            ii.set(k, "0")
        j = ET.SubElement(root, "joint")
        j.set("name", f"openarm_{side}_finger_joint{i}")
        j.set("type", "prismatic")
        ET.SubElement(j, "parent").set("link", "world")
        ET.SubElement(j, "child").set("link", f"openarm_{side}_dummy_finger{i}")
        ET.SubElement(j, "origin").set("xyz", "0 0 0")
        ET.SubElement(j, "axis").set("xyz", "0 0 1")
        lim = ET.SubElement(j, "limit")
        lim.set("lower", "0")
        lim.set("upper", "0.001")
        lim.set("effort", "1")
        lim.set("velocity", "1")

# MuJoCo compiler directives (mirrors orca_arm's build_orcabot_mjcf.py)
mj = ET.SubElement(root, "mujoco")
comp = ET.SubElement(mj, "compiler")
comp.set("strippath", "false")
comp.set("discardvisual", "false")
comp.set("balanceinertia", "true")
comp.set("fusestatic", "false")

input_urdf = os.path.join(OUT, "_input.urdf")
tree.write(input_urdf, xml_declaration=True, encoding="utf-8")

# ── 2. compile URDF -> MJCF ──────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(input_urdf)
mjcf_path = os.path.join(OUT, "orcabot_yaw.xml")
mujoco.mj_saveLastXML(mjcf_path, model)
os.remove(input_urdf)

# normalize mesh paths (mj_saveLastXML may emit absolute paths)
mtree = ET.parse(mjcf_path)
mroot = mtree.getroot()
for compiler in mroot.iter("compiler"):
    compiler.set("meshdir", "assets")
for mesh in mroot.iter("mesh"):
    if mesh.get("file"):
        mesh.set("file", os.path.basename(mesh.get("file")))

# ── 3. actuators required by JointResolver (unused in qpos viewer mode) ──────
act = ET.SubElement(mroot, "actuator")
for side in ("left", "right"):
    for i in range(1, 8):
        m = ET.SubElement(act, "motor")
        m.set("name", f"{side}_joint{i}_ctrl")
        m.set("joint", f"openarm_{side}_joint{i}")
    m = ET.SubElement(act, "motor")
    m.set("name", f"{side}_finger1_ctrl")
    m.set("joint", f"openarm_{side}_finger_joint1")
ET.indent(mtree, space="  ")
mtree.write(mjcf_path, xml_declaration=True, encoding="utf-8")
print(f"[mjcf] wrote {mjcf_path}")

# ── 4. scene.xml with home keyframe ──────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(mjcf_path)
home = np.zeros(model.nq)
for side in ("left", "right"):
    jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint4"
    )
    home[model.jnt_qposadr[jid]] = ELBOW90
qpos_str = " ".join(f"{x:.6g}" for x in home)

scene = f"""<mujoco model="orcabot_yaw scene">
  <include file="orcabot_yaw.xml"/>

  <statistic center="0 0 0.43" extent=".85" meansize="0.05"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="160" elevation="-10"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>

  <keyframe>
    <key name="home" qpos="{qpos_str}"/>
  </keyframe>
</mujoco>
"""
scene_path = os.path.join(OUT, "scene.xml")
with open(scene_path, "w") as f:
    f.write(scene)
print(f"[scene] wrote {scene_path} (home: elbows at {ELBOW90} rad)")

# ── 5. validate against the pipeline's own contracts ─────────────────────────
from openarm_mujoco.v2 import JointResolver  # noqa: E402

smodel = mujoco.MjModel.from_xml_path(scene_path)
resolver = JointResolver(smodel)  # raises if any name is missing
for side in ("left", "right"):
    bid = mujoco.mj_name2id(smodel, mujoco.mjtObj.mjOBJ_BODY, CARPALS_BODY[side])
    assert bid >= 0, f"missing carpals body for {side}"
kid = mujoco.mj_name2id(smodel, mujoco.mjtObj.mjOBJ_KEY, "home")
assert kid >= 0
print(f"[check] JointResolver OK, carpals bodies OK, home keyframe OK "
      f"(nq={smodel.nq}, nu={smodel.nu})")

# ── 6. rot-offset for the carpals frame ──────────────────────────────────────
# R_off_new = R_off_v1 @ R_link7_v1(elbow90)^T @ R_carpals_yaw(elbow90):
# maps the recorded anatomical-wrist attitude to the carpals convention while
# preserving the physical correspondence calibrated on 2026-07-22.
v1 = mujoco.MjModel.from_xml_path(os.path.normpath(V1_SCENE))
d1 = mujoco.MjData(v1)
jid = mujoco.mj_name2id(v1, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_joint4")
d1.qpos[v1.jnt_qposadr[jid]] = ELBOW90
mujoco.mj_forward(v1, d1)
bid = mujoco.mj_name2id(v1, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_link7")
R_link7 = d1.xmat[bid].reshape(3, 3)

d2 = mujoco.MjData(smodel)
mujoco.mj_resetDataKeyframe(smodel, d2, kid)
mujoco.mj_forward(smodel, d2)
bid = mujoco.mj_name2id(smodel, mujoco.mjtObj.mjOBJ_BODY, CARPALS_BODY["right"])
R_carpals = d2.xmat[bid].reshape(3, 3)

R_off = (
    Rotation.from_euler("xyz", V1_ROT_OFFSET_EULER, degrees=True)
    * Rotation.from_matrix(R_link7).inv()
    * Rotation.from_matrix(R_carpals)
)
e = R_off.as_euler("xyz", degrees=True)
print(f"[offset] right-carpals --rot-offset-euler {e[0]:.3f} {e[1]:.3f} {e[2]:.3f}")
print("[offset] validate offline against a clip before trusting it "
      "(scripts/validate_wrist_offset.py)")
