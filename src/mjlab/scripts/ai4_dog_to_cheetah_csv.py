"""Retarget an AI4Animation dog position clip to the articulated Cheetah.

AI4Animation ``*_joint_pos.txt`` files contain flattened XYZ positions for the
24-joint DogSet skeleton at 60 Hz.  This script performs the same coordinate
normalization used by the public Kine2Go retargeter, then solves a small MuJoCo
IK problem for the four Cheetah feet.  The spine hinge is initialized from the
source fore/hind body angle and is intentionally kept as an explicit DOF.

The result is a generalized-coordinate CSV suitable for
``cheetah_csv_to_npz``.  The retargeter is deliberately offline and deterministic;
it does not run PPO or hide IK failures behind a learned policy.
"""

from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import mjlab
from mjlab.asset_zoo.robots.cheetah.cheetah_constants import get_spec
from mjlab.tasks.tracking.config.cheetah.env_cfgs import _CHEETAH_JOINTS

_PELVIS_ID = 0
_NECK_ID = 3
# AI4Animation ordering: left shoulder, left hip, right shoulder, right hip.
_HIP_IDS = (6, 16, 11, 20)
_TOE_IDS = (10, 19, 15, 23)
_SITE_NAMES = ("FL", "RL", "FR", "RR")
_XYZ_DIM = 3
_COORD_ROT = Rotation.from_euler("xyz", (0.5 * np.pi, 0.0, 0.0))
_ROOT_ROT = Rotation.from_euler("xyz", (0.0, 0.0, 0.47 * np.pi))


def _root_pose(points: np.ndarray) -> tuple[np.ndarray, Rotation]:
  pelvis = points[_PELVIS_ID]
  neck = points[_NECK_ID]
  forward = neck - pelvis + np.array((0.0, 0.0, 0.04))
  forward /= np.linalg.norm(forward)

  shoulder_delta = points[_HIP_IDS[0]] - points[_HIP_IDS[2]]
  hip_delta = points[_HIP_IDS[1]] - points[_HIP_IDS[3]]
  shoulder_delta /= np.linalg.norm(shoulder_delta)
  hip_delta /= np.linalg.norm(hip_delta)
  left = 0.5 * (shoulder_delta + hip_delta)
  left /= np.linalg.norm(left)
  up = np.cross(forward, left)
  up /= np.linalg.norm(up)
  left = np.cross(up, forward)
  left[2] = 0.0
  left /= np.linalg.norm(left)
  matrix = np.array(
    [
      [forward[0], left[0], up[0]],
      [forward[1], left[1], up[1]],
      [forward[2], left[2], up[2]],
    ]
  )
  return 0.5 * (pelvis + neck), Rotation.from_matrix(matrix) * _ROOT_ROT


def _spine_angle(points: np.ndarray, root_pos: np.ndarray, root_rot: Rotation) -> float:
  front = 0.5 * (points[_HIP_IDS[0]] + points[_HIP_IDS[2]])
  rear = 0.5 * (points[_HIP_IDS[1]] + points[_HIP_IDS[3]])
  rear_local = root_rot.inv().apply(rear - front)
  # body_pitch_joint rotates the rear segment around +Y.  The neutral segment
  # points toward -X, so positive q raises the rear body.
  del root_pos
  return float(np.arctan2(rear_local[2], -rear_local[0]))


def _load_positions(path: str, frame_start: int, frame_end: int | None) -> np.ndarray:
  values = np.loadtxt(path, delimiter=",")
  if values.ndim == 1:
    values = values[None, :]
  if values.shape[1] % _XYZ_DIM != 0 or values.shape[1] // _XYZ_DIM <= _TOE_IDS[-1]:
    raise ValueError(
      "AI4 input must contain flattened XYZ positions for at least 24 joints."
    )
  values = values[frame_start:frame_end]
  return values.reshape(values.shape[0], -1, _XYZ_DIM)


def _clip_joint(model: mujoco.MjModel, name: str, value: float) -> float:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  low, high = model.jnt_range[joint_id]
  return float(np.clip(value, low, high))


def _solve_feet(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos: np.ndarray,
  targets: np.ndarray,
  joint_names: tuple[str, ...],
  site_names: tuple[str, ...],
  iterations: int,
) -> float:
  joint_qpos = [
    int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
    for name in joint_names
  ]
  joint_dofs = [
    int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
    for name in joint_names
  ]
  site_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names
  ]
  data.qpos[:] = qpos
  for _ in range(iterations):
    mujoco.mj_forward(model, data)
    residual = np.concatenate(
      [targets[i] - data.site_xpos[site_id] for i, site_id in enumerate(site_ids)]
    )
    if np.linalg.norm(residual) < 1.0e-4:
      break
    site_jacobians = []
    for site_id in site_ids:
      jacobian_pos = np.zeros((3, model.nv))
      jacobian_rot = np.zeros((3, model.nv))
      mujoco.mj_jacSite(model, data, jacobian_pos, jacobian_rot, site_id)
      site_jacobians.append(jacobian_pos)
    stacked_jacobian = np.vstack(site_jacobians)
    jacobian_leg = stacked_jacobian[:, joint_dofs]
    damping = 2.0e-4
    update = jacobian_leg.T @ np.linalg.solve(
      jacobian_leg @ jacobian_leg.T + damping * np.eye(12), residual
    )
    for name, qpos_index, delta in zip(joint_names, joint_qpos, update, strict=True):
      qpos[qpos_index] = _clip_joint(model, name, qpos[qpos_index] + 0.6 * delta)
    data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  return float(
    np.mean(
      [
        np.linalg.norm(targets[i] - data.site_xpos[site_id])
        for i, site_id in enumerate(site_ids)
      ]
    )
  )


def main(
  input_file: str,
  output_file: str,
  scale: float = 0.40,
  base_height: float = 0.22,
  frame_start: int = 0,
  frame_end: int | None = None,
  ik_iterations: int = 20,
) -> None:
  """Retarget an AI4Animation dog clip to Cheetah generalized coordinates."""
  source = _load_positions(input_file, frame_start, frame_end)
  flattened = source.reshape(-1, _XYZ_DIM)
  processed = _ROOT_ROT.apply(_COORD_ROT.apply(flattened)).reshape(source.shape)
  source_root0, _ = _root_pose(processed[0])

  model = get_spec().compile()
  data = mujoco.MjData(model)
  qpos = np.zeros(model.nq)
  qpos[3] = 1.0
  joint_qpos = {
    name: int(
      model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    )
    for name in _CHEETAH_JOINTS
  }
  output = []
  foot_errors = []
  for points in tqdm(processed, desc="AI4 -> Cheetah", unit="frame"):
    source_root, root_rot = _root_pose(points)
    root_delta = (source_root - source_root0) * scale
    root_pos = np.array((root_delta[0], root_delta[1], base_height))
    qpos[:3] = root_pos
    qpos[3:7] = root_rot.as_quat(scalar_first=True)
    qpos[joint_qpos["body_pitch_joint"]] = _clip_joint(
      model,
      "body_pitch_joint",
      _spine_angle(points, source_root, root_rot),
    )

    source_local_feet = [
      root_rot.inv().apply(points[toe_id] - source_root) * scale for toe_id in _TOE_IDS
    ]
    targets = np.stack(
      [root_pos + root_rot.apply(local) for local in source_local_feet]
    )
    foot_errors.append(
      _solve_feet(
        model,
        data,
        qpos,
        targets,
        tuple(name for name in _CHEETAH_JOINTS if name != "body_pitch_joint"),
        _SITE_NAMES,
        ik_iterations,
      )
    )
    output.append(
      np.concatenate(
        [
          root_pos,
          root_rot.as_quat(),
          np.asarray([qpos[joint_qpos[name]] for name in _CHEETAH_JOINTS]),
        ]
      )
    )

  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(output_file, np.asarray(output), delimiter=",")
  print(
    f"[INFO] Saved {len(output)} retargeted frames to {output_file}; "
    f"foot IK mean={np.mean(foot_errors):.4f} m, "
    f"p95={np.percentile(foot_errors, 95):.4f} m"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
