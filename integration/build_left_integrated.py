#!/usr/bin/env python3
"""
build_left_integrated.py
------------------------
Take the OpenArm v1.0 bimanual URDF, remove the LEFT wrist (joint6/7) + the
stock gripper, and splice in the 3D-printed integrator bracket (LEFT_PIECE.stl)
mounted on openarm_left_link5.

This is step 1 (arm + adapter). The OrcaHand gets grafted onto the adapter in a
later step.

Run:   python3 build_left_integrated.py
Out:   openarm_left_integrated.urdf   (view with view_urdf.py)
"""
import copy
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
OPENARM = os.path.join(HERE, "openarm_description")
SRC = os.path.join(OPENARM, "assets/robot/openarm_v1.0/urdf/example/v1.urdf")
OUT = os.path.join(HERE, "openarm_left_integrated.urdf")

# ---------------------------------------------------------------------------
# Mount link and the parts to remove (LEFT side only; right arm untouched)
# ---------------------------------------------------------------------------
MOUNT_LINK = "openarm_left_link5"          # keep joint5 (forearm roll); mount here

DEL_LINKS = {
    "openarm_left_link6", "openarm_left_link7",
    "openarm_left_hand_tcp",
    "openarm_left_left_finger", "openarm_left_right_finger",
}
DEL_JOINTS = {
    "openarm_left_joint6", "openarm_left_joint7",
    "openarm_left_hand_tcp_joint",
    "openarm_left_finger_joint1", "openarm_left_finger_joint2",
}

# ---------------------------------------------------------------------------
# The adapter (from the Onshape export)
# ---------------------------------------------------------------------------
ADAPTER_LINK = "openarm_left_adapter"
ADAPTER_MESH = ("package://openarm_description/assets/robot/openarm_v1.0/"
                "left_final_assembly_orcaxopenarm/meshes/LEFT_PIECE.stl")
# Onshape's per-mesh offset (leave as exported; it positions the mesh in the
# adapter's own frame):
ADAPTER_VIS_ORIGIN_XYZ = "-0.188187 -0.0696202 -0.104346"
ADAPTER_VIS_ORIGIN_RPY = "0 0 0"

# >>> PLACEHOLDER mount transform (adapter frame relative to link5) <<<
# We don't yet know how the Onshape assembly frame relates to link5's frame,
# so this is a starting guess. Measure it from CAD (import link5.dae, mate the
# bracket, read the transform) or eyeball-correct in the viewer, then update.
JOINT_A_XYZ = "0 0 0"
JOINT_A_RPY = "0 0 0"

# ---------------------------------------------------------------------------
tree = ET.parse(SRC)
root = tree.getroot()

# 1) delete the LEFT wrist links/joints + gripper
removed_l = [l.get("name") for l in list(root.findall("link")) if l.get("name") in DEL_LINKS]
removed_j = [j.get("name") for j in list(root.findall("joint")) if j.get("name") in DEL_JOINTS]
for l in list(root.findall("link")):
    if l.get("name") in DEL_LINKS:
        root.remove(l)
for j in list(root.findall("joint")):
    if j.get("name") in DEL_JOINTS:
        root.remove(j)

# 2) add the adapter link (visual + cloned collision + placeholder inertial)
link = ET.SubElement(root, "link"); link.set("name", ADAPTER_LINK)

vis = ET.SubElement(link, "visual")
vo = ET.SubElement(vis, "origin"); vo.set("xyz", ADAPTER_VIS_ORIGIN_XYZ); vo.set("rpy", ADAPTER_VIS_ORIGIN_RPY)
vg = ET.SubElement(vis, "geometry"); vm = ET.SubElement(vg, "mesh")
vm.set("filename", ADAPTER_MESH); vm.set("scale", "1 1 1")
vmat = ET.SubElement(vis, "material"); vmat.set("name", "adapter_blue")
ET.SubElement(vmat, "color").set("rgba", "0.615686 0.811765 0.929412 1")

# collision = copy of the visual, minus the material (needed for MuJoCo/physics)
col = copy.deepcopy(vis); col.tag = "collision"
for m in col.findall("material"):
    col.remove(m)
link.append(col)

# placeholder inertial (Onshape didn't export one) -- replace with real values
inr = ET.SubElement(link, "inertial")
ET.SubElement(inr, "origin").set("xyz", "0 0 0")
ET.SubElement(inr, "mass").set("value", "0.1")
ii = ET.SubElement(inr, "inertia")
for k, v in dict(ixx="1e-4", ixy="0", ixz="0", iyy="1e-4", iyz="0", izz="1e-4").items():
    ii.set(k, v)

# 3) Joint A: link5 -> adapter (fixed, PLACEHOLDER transform)
jA = ET.SubElement(root, "joint")
jA.set("name", "openarm_left_link5_to_adapter_joint"); jA.set("type", "fixed")
ET.SubElement(jA, "parent").set("link", MOUNT_LINK)
ET.SubElement(jA, "child").set("link", ADAPTER_LINK)
ja_o = ET.SubElement(jA, "origin"); ja_o.set("xyz", JOINT_A_XYZ); ja_o.set("rpy", JOINT_A_RPY)

# 4) write
root.set("name", "openarm_left_integrated")
ET.indent(tree, space="  ")
tree.write(OUT, xml_declaration=True, encoding="utf-8")

# --- report ---
print("removed links :", removed_l)
print("removed joints:", removed_j)
print("added link    :", ADAPTER_LINK, "-> mounted on", MOUNT_LINK)
print("wrote         :", OUT)

# --- validate: single root, no orphan parents ---
links = {l.get("name") for l in root.findall("link")}
child = set(); parent = set()
for j in root.findall("joint"):
    p = j.find("parent"); c = j.find("child")
    if p is not None: parent.add(p.get("link"))
    if c is not None: child.add(c.get("link"))
roots = links - child
orphans = parent - links
print("root link(s)  :", roots, "(should be just {'world'})")
print("orphan parents:", orphans or "none")

# 5) portable copy: rewrite every package:// mesh path to an ABSOLUTE path so
#    any viewer (Foxglove, etc.) opens it without a package resolver.
PACKAGE_ROOTS = {"openarm_description": OPENARM}


def to_abs(fname):
    if fname.startswith("package://"):
        rest = fname[len("package://"):]
        pkg, _, tail = rest.partition("/")
        if pkg in PACKAGE_ROOTS:
            return os.path.join(PACKAGE_ROOTS[pkg], tail)
    return fname


missing = []
for mesh in root.iter("mesh"):
    ab = to_abs(mesh.get("filename", ""))
    mesh.set("filename", ab)
    if not os.path.isfile(ab):
        missing.append(ab)

OUT_PORTABLE = os.path.join(HERE, "openarm_left_integrated_portable.urdf")
tree.write(OUT_PORTABLE, xml_declaration=True, encoding="utf-8")
print("wrote portable:", OUT_PORTABLE)
print("missing meshes:", missing or "none (all resolve)")
