#!/usr/bin/env python3
"""
view_urdf.py -- visualize the links of a URDF.

Usage:
    python3 view_urdf.py                     # default: OpenArm v1.0
    python3 view_urdf.py path/to/robot.urdf  # any URDF (e.g. your combined one)

Full solid-mesh view needs:   pip install yourdfpy
If yourdfpy isn't installed, it falls back to a matplotlib "skeleton" plot of
every link frame (needs only numpy + matplotlib, which you already have).

Both modes print the full list of links.
"""
import os
import sys
import xml.etree.ElementTree as ET

# --- Where each ROS "package://" name lives on disk -------------------------
# Add entries here as you merge more descriptions (e.g. orcahand_description).
PACKAGE_ROOTS = {
    "openarm_description":
        "/Users/williaml.zhang/AprilRobotics/OpenArmxOrcaHand_Integration/openarm_description",
    # "orcahand_description": "/path/to/orcahand_description",
}

DEFAULT_URDF = os.path.join(
    PACKAGE_ROOTS["openarm_description"],
    "assets/robot/openarm_v1.0/urdf/example/v1.urdf",
)


def resolve(fname):
    """Map package://pkg/rest  ->  <root>/rest ; leave other paths untouched."""
    if fname.startswith("package://"):
        rest = fname[len("package://"):]
        pkg, _, tail = rest.partition("/")
        root = PACKAGE_ROOTS.get(pkg)
        if root:
            return os.path.join(root, tail)
        print(f"  [warn] no root configured for package '{pkg}' -- add it to PACKAGE_ROOTS")
    return fname


def list_links(root):
    links = [l.get("name") for l in root.findall("link")]
    print(f"\nLinks ({len(links)}):")
    for n in links:
        print("  ", n)
    return links


# --- Mode 1: full solid meshes via yourdfpy ---------------------------------
def view_meshes(path):
    import yourdfpy
    robot = yourdfpy.URDF.load(path, filename_handler=lambda fname: resolve(fname))
    print(f"\nLinks ({len(robot.link_map)}):")
    for n in robot.link_map:
        print("  ", n)
    print("\nOpening 3D window (close it to exit)...")
    robot.show()


# --- Mode 2: matplotlib skeleton (no extra deps) ----------------------------
def view_skeleton(path):
    import numpy as np
    import matplotlib.pyplot as plt

    root = ET.parse(path).getroot()
    list_links(root)

    def R(rpy):
        a, b, c = rpy
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def T(xyz, rpy):
        m = np.eye(4); m[:3, :3] = R(rpy); m[:3, 3] = xyz; return m

    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(x) for x in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
        rpy = [float(x) for x in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
        joints.append((j.find("parent").get("link"), j.find("child").get("link"), T(xyz, rpy)))

    links = [l.get("name") for l in root.findall("link")]
    kids = {c for _, c, _ in joints}
    base = next(l for l in links if l not in kids)
    world = {base: np.eye(4)}; edges = []
    changed = True
    while changed:
        changed = False
        for par, ch, t in joints:
            if par in world and ch not in world:
                world[ch] = world[par] @ t; edges.append((par, ch)); changed = True

    def col(n):
        return 'tab:blue' if 'left' in n else 'tab:red' if 'right' in n else 'black'

    def short(n):
        return n.replace('openarm_', '')

    fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection='3d')
    for par, ch in edges:
        a, b = world[par][:3, 3], world[ch][:3, 3]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=col(ch), lw=1.5, alpha=.6)
    for n, m in world.items():
        p = m[:3, 3]
        ax.scatter(*p, color=col(n), s=28)
        ax.text(p[0], p[1], p[2], '  ' + short(n), fontsize=7, color=col(n))

    pts = np.array([m[:3, 3] for m in world.values()])
    c = pts.mean(0); rng = (pts.max(0) - pts.min(0)).max() / 2 or 0.1
    ax.set_xlim(c[0] - rng, c[0] + rng)
    ax.set_ylim(c[1] - rng, c[1] + rng)
    ax.set_zlim(c[2] - rng, c[2] + rng)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('URDF links (zero pose)  |  blue=left  red=right  black=body')
    ax.view_init(elev=18, azim=-70)
    plt.tight_layout()
    plt.show()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URDF
    if not os.path.isfile(path):
        sys.exit(f"URDF not found: {path}")
    print("URDF:", path)
    try:
        view_meshes(path)
    except ImportError:
        print("\n[yourdfpy not installed -> showing skeleton instead]")
        print("For the full solid-mesh view:  pip install yourdfpy\n")
        view_skeleton(path)


if __name__ == "__main__":
    main()
