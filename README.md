# OpenArm × ORCA Hand integration

OpenArm V1 arms with ORCA Hands mounted through a custom 3D-printed adapter, plus
the inverse kinematics and Meta Quest hand-tracking teleop that drive the combined
robot (arm + wrist + 16 finger joints, single arm or bimanual).

## Repository layout

```
integration/            Custom URDF: OpenArm v1.0 + 3D-printed adapter (step 1, left side)
  build_left_integrated.py    removes left joint6/7 + stock gripper, mounts adapter on link5
  openarm_left_integrated.urdf  generated output
  view_urdf.py                visualize any URDF (yourdfpy, or matplotlib skeleton fallback)
  openarm_description/        just the OpenArm meshes/source URDF this needs + adapter mesh

dora-openarm-replay/    Combined model + IK teleop (dora-rs dataflows)
  models/orcabot_yaw/         MuJoCo model: OpenArm (J1-J6) + adapter + ORCA hand, both sides
  config/dataflow-hw-*.yaml   hardware teleop graphs (right, left, bimanual, arm-only, hand-only)
  config/dataflow-mujoco-*.yaml  simulation-only graphs
  scripts/quest_live_source.py  Quest WebXR hand tracking -> wrist IK targets + finger angles
  scripts/arm6_from_ik.py       7-DOF IK output -> 6 CAN arm motors (+ ORCA wrist)
  scripts/orca_follower.py      finger/wrist angles -> ORCA hand Feetech servos
  scripts/build_orcabot_yaw_scene.py  URDF -> MJCF conversion used to make models/orcabot_yaw

orca-retarget/          Human hand keypoints -> ORCA joint angles (semantic-name hand URDFs)
docs/PIPELINE.md        Full pipeline write-up: graph, IK, wrist handling, safety, gotchas
```

## How it works

```
Quest hand tracking -> quest_live_source.py
   |- wrist pose -> ik (solves on models/orcabot_yaw/scene.xml, target = ORCA carpals body)
   |                  |- J1-J6  -> arm6_from_ik.py -> dora-openarm -> CAN motors
   |                  '- joint7 (ORCA wrist flex) ----------------> orca_follower.py
   '- finger landmarks -> orca-retarget -> 16 finger angles ------> orca_follower.py -> Feetech servos
```

The IK solves on the full arm + adapter + hand chain, so it places the *palm* where
the operator's hand is. The ORCA wrist-flex joint is renamed `openarm_{side}_joint7`
so the solver sees a standard 7-joint arm; its output is split between the CAN arm
motors and the hand's wrist servo.

## Running (hardware)

```bash
cd dora-openarm-replay/config
dora build dataflow-hw-arm-hand-right.yaml     # installs the dora-openarm / dora-openarm-ik nodes
dora run   dataflow-hw-arm-hand-right.yaml     # or dataflow-hw-arm-hand-bimanual.yaml
```

Requires CAN interfaces (`can0` right, `can1` left), ORCA hands calibrated with
`orca_core`, and a Quest on the same network (WebXR on port 8765). See
`docs/PIPELINE.md` for start-pose, safety and calibration details.

## Notes

- `integration/` is the first-stage custom URDF (adapter on link5, hand not yet
  attached). The teleop runs on `models/orcabot_yaw`, where the same adapter is
  mounted on link6 with the ORCA hand attached. The source URDF for that model
  (`orcabot_yaw_render.urdf`) is not included.
- `view_urdf.py` has a hardcoded local path to `openarm_description`; edit it to
  point at `integration/openarm_description` on your machine.
- `orca-retarget/pyproject.toml` points `orca_teleop` at a local editable path;
  change it to your checkout of orca_teleop.
- The arm IK solver itself is the `dora-openarm-ik` node from enactic's
  `openarm_control`, installed at `dora build` time.

## Credits

- OpenArm description and meshes: enactic/openarm_description (Apache-2.0, see
  `integration/openarm_description/LICENSE.txt`)
- ORCA Hand models: orcahand/orcahand_description (MIT)
- Combined OpenArm + ORCA model derived from fracapuano/orca_arm
