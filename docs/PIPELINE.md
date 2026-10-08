# ORCABOT Bimanual VR Teleop — Pipeline Documentation

## 0. What it is
A single **Meta Quest 3** hand-tracks **both** of your hands; each hand drives one
**OpenArm V1** (6 CAN motors) plus one **ORCA hand** (17 Feetech servos). Your wrist
*pose* moves the arm (via inverse kinematics); your finger *poses* move the hand (via a
retargeter). It runs as a **dora-rs dataflow** — a graph of independent processes
("nodes") passing Apache-Arrow messages over shared memory — on the robot PC. No
controllers, one headset, both arms.

## 1. The dataflow graph (bimanual)
```
                       ┌───────────────── quest-live-source (--hand both) ─────────────────┐
   Quest WebXR ──WS──▶ │  WebXR server :8765; reads BOTH hands' wrist matrix + 21 landmarks │
   (adb reverse)       └── pose_left ─ pose_right ─ fingers_left ─ fingers_right ────────────┘
                              │           │              │                │
                     target_left │  target_right         │                │
                              ▼           ▼              │                │
                       ┌──────── ik (dora-openarm-ik, --mode bimanual) ────────┐
                       │  mink QP differential IK on the orcabot_yaw model      │
                       └── position_left (8) ──────── position_right (8) ───────┘
                              │   │                     │   │
                 ┌────────────┘   └──────┐   ┌──────────┘   └───────────┐
                 ▼ (wrist j7)            ▼   ▼ (wrist j7)               ▼
          arm6-left (--side left)   orca-left   arm6-right (--side right)   orca-right
            drops j7+gripper          │           drops j7+gripper           │
                 │ move_position(6)   │                │ move_position(6)    │
                 ▼                    ▼                ▼                     ▼
          follower-left          ORCA hand L      follower-right         ORCA hand R
          dora-openarm          orca_core         dora-openarm          orca_core
          MIT → can1            Feetech ttyACM0   MIT → can0            Feetech ttyACM1
          (6 DM motors)         (17 servos)       (6 DM motors)         (17 servos)
```
A **node = an OS process**. dora's daemon spawns them, wires each named `output → input`
edge as a shared-memory channel (zero-copy Arrow arrays), clocks the source off a
`dora/timer/millis/33` (~30 Hz) tick, and on Ctrl-C sends SIGINT so every node's
`finally:` runs (that's how followers torque off cleanly).

## 2. The nodes
| Node | File / package | Role | Origin |
|---|---|---|---|
| `quest-live-source` | `quest_live_source.py` | WebXR ingress; wrist matrix → EE pose; landmarks → finger angles | existing single-hand; **`--hand both` (`_run_both`) added** |
| `ik` | `dora-openarm-ik` (`openarm_control`) | mink QP differential IK, both arms one solve | upstream (enactic) |
| `arm6-adapter` | `arm6_from_ik.py` | Trims IK 8-vec → 6 CAN joints | **made; then `--side`-aware** |
| `follower` | `dora-openarm` (`openarm_driver`) | CAN motor driver (MIT position) | upstream; **motor-count + step_limit patched** |
| `orca-follower` | `orca_follower.py` | Drives ORCA hand (fingers + wrist) via `orca_core` | **made** |

## 3. Arm path → CAN
**Hardware:** OpenArm V1 is nominally 7-DoF; mounting the ORCA hand **removed joint-7's
motor** (and there's no gripper), so each arm = **6 DM CAN motors J1–J6** (J1‑2 DM8009,
J3‑4 DM4340, J5‑6 DM4310). Left = **can1**, right = **can0**, CAN-FD @ 1M/5M.

**Driver:** MIT mode `MITParam(kp, kd, position, vel=0, tau_ff=0)` — pure position, **no
gravity/velocity feedforward**. Gains `kps=[70,70,70,60,10,10]`, `kds=[2.75,2.5,2,2,0.7,0.6]`.

Fixes:
- **Motor-count hardcode** (assumes 7/8-motor V1 → IndexError on 6) → derive from
  `len(send_ids)`. (site-packages patch)
- **Align crawl** `_align`: on a target >`align_threshold` (0.1 rad) from last command it
  **creeps** at `step_limit` rad/step until within 0.1 rad, then tracks **1:1**. `step_limit`
  raised `0.001`→`0.01` (~1.7→17°/s). Limit is in **joint space (rad)** on purpose —
  directly bounds motor speed everywhere, no FK/Jacobian, safe near singularities.

**`arm6_from_ik.py`:** IK emits `[j1..j6, j7_wrist, gripper]` (8); physical arm is 6, so this
**drops j7_wrist + gripper**, forwards `[j1..j6]`. `--side` selects `position_left`/`right`.

**Start-pose safety:** `--start-on-startup` torques on and moves to the config `start` pose;
we **re-measure the live pose into it** each launch so torque-on = zero motion.

## 4. Hand path → Feetech
**Hardware:** each ORCA hand = **17 Feetech servos** = `wrist` (joint7) + 16 finger joints on
a USB serial bus. Two hands → two `/dev/ttyACM*` that can swap on replug → both configs use
stable **`/dev/serial/by-id/…`** paths.

**`orca_follower.py`** per message: `fingers_*` = `{URDF names, radians}` → translate names to
orca_core ids (inverse of `semantic_to_model_joints`) → **rad→deg** → add `wrist` (from IK j7)
→ `set_joint_positions(OrcaJointPositions.from_dict(targets))`.

`orca_core` quirks handled:
- `set_neutral_position()` restores the config's `current_based_position` mode (no goal-current
  → won't move) → we `set_control_mode("position")` after neutral so streaming drives.
- **Calibration is a separate per-hand `calibration.yaml`** (motor limits + ratios), not in
  `config.yaml`; `is_calibrated()` refuses to move without it. The right hand's lived on the Mac
  (tell: macOS `cu.usbmodem` port in its config) and was copied over.

## 5. One human hand → arm AND hand
Two independent streams from the same WebXR sample:
- **Wrist pose (4×4)** → arm. `WristCalibrator` captures a one-shot `R_fix` from the robot's
  **home EE pose** on frame 1 (hold neutral at startup = "home"), then
  `R_ee = R_wrist·R_fix`, `p_ee = p_home + scale·(p_wrist − p_wrist_cal)`, clamped within
  `max_reach` of home. → IK → arm joints.
- **21 finger landmarks** → retargeter → 16 ORCA finger angles → servos.

## 6. The wrist: driven by the arm's IK, actuated on the hand
Two "wrists": (a) the arm's own **J5/J6** wrist joints, part of the 6 CAN motors, orient the
EE through the chain; (b) the ORCA hand's **`wrist` joint (joint7)**, the wrist-flex servo on
Feetech. Historically J7 was an arm joint; mounting the hand **moved joint7's actuation to the
hand's Feetech bus** ("joint7 split to Feetech") but the **IK still models it as part of the
arm** (its 8-vec includes `j7_wrist`). So the wrist DoF is **computed by the arm's IK** and, in
our wiring, **actuated on the hand**: `orca_follower --wrist` takes `position_{side}[6]` (IK j7,
rad) → `wrist_scale·deg(j7)+offset` → the ORCA `wrist` joint (ROM-clamped `[-65,35]°`). The wrist
orientation you make is realized **mostly by the arm** (J5/J6 + chain), with a small wrist-flex
routed to the hand's joint7 — i.e. the wrist is "connected to the OpenArm," not a standalone 1:1
hand-wrist joint.

## 7. Retargeting (fingers)
`orca_retarget`'s **adaptive_analytical** `FingerRetargeter(side)`: 21 landmarks (MediaPipe
order) → auto-calibrated **`mano_scale`** (hand-size normalization) → per-finger angles
(degrees) for ORCA joints. In `quest_live_source`: WebXR landmarks → `retargeter_landmarks_from_webxr`
→ inverse-permute (DataInfra↔MediaPipe, identity round-trip) → `rt.retarget()` → `{semantic: deg}`
→ `np.deg2rad` → emit `fingers_*` as `{URDF names, rad}`.
- **Names:** URDF joint names ↔ orca_core ids bridged by inverting `semantic_to_model_joints`.
- **Units bug we hit:** retargeter native = deg; source converts to rad (for the MuJoCo viewer);
  but `orca_core.set_joint_positions` + ROMs are **degrees**. Feeding rad made fingers barely move
  → `math.degrees()` fix. Tell: `thumb_abd` pinned at `0.9599 rad = exactly 55.0°` (its ROM max).

## 8. Inverse kinematics — and why the wrist is imprecise
`dora-openarm-ik` = **mink differential IK** (QP/step) over the bimanual `orcabot_yaw` model;
`--mode bimanual` solves both arms. Output `[j1..j6, j7, gripper]` per arm. Weights
`--pos-cost 1.0`, **`--ori-cost 0.25`**, `--posture-cost 0.1`, damping regularizers,
`--max-iters 5`, `--dt 0.08`.

**Why wrist / orientation tracking is inaccurate:**
1. **Orientation weighted 4× less than position** (0.25 vs 1.0) → QP trades wrist orientation for
   hitting position + a comfortable posture. Position tracks better than pointing.
2. **WebXR-wrist → robot-EE frame map is a first guess** (code's own TODO: the rotation offset was
   for the old DataInfra tracker frame, not the WebXR wrist frame; unsolved). `R_fix` is one-shot
   from "home," so any imprecision biases all orientation.
3. **Differential IK, 5 iters @ dt 0.08** — small steps toward target each tick; fast wrist
   rotations lag and never fully converge.
4. **Wrist DoF split & under-tuned:** gross orientation from the arm chain (J5/J6); the hand's
   joint7 gets IK `j7` via `--wrist-scale/offset` whose **sign/scale isn't calibrated** (right OK
   at scale 1 in a small ±10° range; left unverified). Neither is a precise 1:1 wrist.
5. **Arm control lag compounds it:** MIT with no gravity FF → ~`load/kp` steady-state error
   (~5°/joint); align crawl adds lag on quick moves — both smear orientation more than position.
6. **Redundancy resolution:** a 6-DoF arm on a 6-DoF pose has little slack; low ori-cost lets the
   posture/nullspace term pull orientation off-target.

Net: **position tracks well; orientation (the wrist) is the loosest axis.** To tighten: raise
`ori-cost`, solve the WebXR→EE frame offset, calibrate `wrist-scale/offset`, add gravity FF.

## 9. Bimanual adjustments (one hand → two)
1. **One WebXR server owns 8765** → can't run two sources → `quest_live_source --hand both`
   (`_run_both`) builds per-hand machinery for both sides and emits `pose_left/right` +
   `fingers_left/right`. Single-hand `_run` left untouched.
2. **Side-aware nodes:** `arm6_from_ik.py` + `orca_follower.py` got `--side`; `orca_follower` also
   `--fingers-input` (its finger input id was hardcoded `fingers_right`). Defaults preserve
   single-hand behavior.
3. **IK `--mode bimanual`** (one QP → both `position_*`).
4. **Two of each downstream branch:** arm6-left→follower-left (can1), arm6-right→follower-right
   (can0), orca-left, orca-right — each reading its own by-id port + calibration + `position_{side}`.

## 10. Chronological gotchas (debugging trail)
1. Launch env: no-`build:` dataflows need `source ~/orca_teleop/.venv/bin/activate` + plain
   `dora run` (not `uv run … --uv`).
2. `ORCAHAND_DESCRIPTION_DIR` must be exported.
3. WebXR needs a secure context → `adb reverse tcp:8765 tcp:8765`.
4. `hands=[]` → enable Quest hand tracking, controllers down, headset worn.
5. Hand dead after neutral → force **POSITION** mode.
6. Fingers barely move → **rad→deg**.
7. Right hand "not calibrated" → its `calibration.yaml` was on the Mac; copied over; ports → by-id;
   baud set.
8. Arm slow → `step_limit` 0.001→0.01.

## 11. Safety model
E-stop per arm + power switch per hand; `is_calibrated()` gate; hold-current start poses (no
launch lurch); `joint_delta_position_limits` glitch backstop (>0.2 rad command jump → force-stop);
joint ROM clamps (arm + hand); align crawl on large arm moves; POSITION-mode fingers with optional
`--max-current`.

## 12. Status / caveats (as of 2026-07-31)
- Left arm+hand and right arm+hand each **verified working solo**.
- Bimanual graph **built + statically verified, not yet hardware-run**.
- Wrist orientation is the known-loose axis (see §8); left wrist sign unconfirmed.
- Arm tracking has ~5° per-joint steady-state lag (no gravity FF).
- Code base is **not under git** — see MANIFEST.md.
