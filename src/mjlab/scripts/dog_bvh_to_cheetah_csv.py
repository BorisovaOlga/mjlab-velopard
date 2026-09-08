"""Retarget the bundled dog BVH clips to the articulated Cheetah.

The clips in ``docs/reborn_spined_cheetah_train/dog_mocap_bvh`` are standard
BVH files, not AI4Animation flattened position files.  Their source axes are
X-forward, Y-up, Z-left and their distances are centimetres.  This converter
parses the hierarchy, evaluates world-space joint positions for every frame,
then performs MuJoCo IK on the four Cheetah feet while preserving a sagittal
spine target from the dog's hind-to-fore trunk vector.
"""

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import mjlab
from mjlab.asset_zoo.robots.cheetah.cheetah_constants import get_spec
from mjlab.scripts.ai4_dog_to_cheetah_csv import _clip_joint, _solve_feet
from mjlab.tasks.tracking.config.cheetah.env_cfgs import _CHEETAH_JOINTS

_CM_TO_M = 0.01
_COORD_ROT = Rotation.from_euler("xyz", (0.5 * np.pi, 0.0, 0.0))
_SITE_NAMES = ("FL", "RL", "FR", "RR")
_LEG_JOINTS = tuple(name for name in _CHEETAH_JOINTS if name != "body_pitch_joint")
_SOURCE_HIPS = ("LeftShoulder", "LeftUpLeg", "RightShoulder", "RightUpLeg")
_SOURCE_FEET = ("LeftHand:end", "LeftFoot:end", "RightHand:end", "RightFoot:end")
_ROBOT_HIPS = (
  "left_front_hip_roll_link",
  "left_hip_roll_link",
  "right_front_hip_roll_link",
  "right_hip_roll_link",
)


@dataclass
class _BvhJoint:
  name: str
  offset: np.ndarray
  channels: tuple[str, ...]
  children: list["_BvhJoint"] = field(default_factory=list)
  end_offset: np.ndarray | None = None


def _parse_joint(lines: list[str], index: int) -> tuple[_BvhJoint, int]:
  header = lines[index].split()
  if header[0] not in {"ROOT", "JOINT"}:
    raise ValueError(f"Expected ROOT or JOINT, got: {lines[index]}")
  name = header[1]
  index += 1
  if lines[index].strip() != "{":
    raise ValueError(f"Expected '{{' after joint {name}")
  index += 1

  offset = np.zeros(3)
  channels: tuple[str, ...] = ()
  children: list[_BvhJoint] = []
  end_offset: np.ndarray | None = None
  while index < len(lines):
    tokens = lines[index].split()
    if not tokens:
      index += 1
      continue
    if tokens[0] == "OFFSET":
      offset = np.asarray([float(value) for value in tokens[1:4]])
      index += 1
    elif tokens[0] == "CHANNELS":
      channels = tuple(tokens[2 : 2 + int(tokens[1])])
      index += 1
    elif tokens[0] == "JOINT":
      child, index = _parse_joint(lines, index)
      children.append(child)
    elif tokens[0] == "End":
      index += 1
      if lines[index].strip() != "{":
        raise ValueError(f"Expected '{{' for end site of {name}")
      index += 1
      end_tokens = lines[index].split()
      if end_tokens[0] != "OFFSET":
        raise ValueError(f"Expected end-site offset for {name}")
      end_offset = np.asarray([float(value) for value in end_tokens[1:4]])
      index += 2
    elif tokens[0] == "}":
      return _BvhJoint(name, offset, channels, children, end_offset), index + 1
    else:
      raise ValueError(f"Unexpected hierarchy line: {lines[index]}")
  raise ValueError(f"Unclosed joint {name}")


def _parse_bvh(path: str) -> tuple[_BvhJoint, np.ndarray, float]:
  lines = [line.strip() for line in Path(path).read_text().splitlines()]
  if not lines or lines[0] != "HIERARCHY":
    raise ValueError(f"{path} is not a BVH file")
  root, index = _parse_joint(lines, 1)
  while index < len(lines) and not lines[index]:
    index += 1
  if lines[index] != "MOTION":
    raise ValueError("BVH hierarchy is not followed by MOTION")
  frame_count = int(lines[index + 1].split(":", 1)[1])
  frame_time = float(lines[index + 2].split(":", 1)[1])
  values = np.asarray(
    [[float(value) for value in line.split()] for line in lines[index + 3 :] if line]
  )
  if values.shape[0] != frame_count:
    raise ValueError(f"Expected {frame_count} frames, found {values.shape[0]}")
  return root, values, frame_time


def _evaluate_frame(
  root: _BvhJoint, values: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Rotation]]:
  positions: dict[str, np.ndarray] = {}
  rotations: dict[str, Rotation] = {}
  cursor = 0

  def visit(
    joint: _BvhJoint,
    parent_pos: np.ndarray | None,
    parent_rot: Rotation,
  ) -> None:
    nonlocal cursor
    translation = np.zeros(3)
    angles: list[float] = []
    axes: list[str] = []
    for channel in joint.channels:
      value = values[cursor]
      cursor += 1
      if channel.endswith("position"):
        translation["XYZ".index(channel[0])] = value
      else:
        axes.append(channel[0])
        angles.append(value)
    local_rot = (
      Rotation.from_euler("".join(axes), angles, degrees=True)
      if axes
      else Rotation.identity()
    )
    world_pos = (
      translation + joint.offset
      if parent_pos is None
      else parent_pos + parent_rot.apply(joint.offset)
    )
    world_rot = parent_rot * local_rot
    positions[joint.name] = world_pos
    rotations[joint.name] = world_rot
    if joint.end_offset is not None:
      positions[f"{joint.name}:end"] = world_pos + world_rot.apply(joint.end_offset)
    for child in joint.children:
      visit(child, world_pos, world_rot)

  visit(root, None, Rotation.identity())
  if cursor != values.shape[0]:
    raise ValueError("BVH frame has unused channel values")
  return positions, rotations


def _root_pose(
  positions: dict[str, np.ndarray],
) -> tuple[np.ndarray, Rotation]:
  front = positions["Spine1"]
  neck = positions["Neck"]
  left_shoulder = positions["LeftShoulder"]
  right_shoulder = positions["RightShoulder"]
  forward = neck - front
  forward /= np.linalg.norm(forward)
  left = left_shoulder - right_shoulder
  left /= np.linalg.norm(left)
  up = np.cross(forward, left)
  up /= np.linalg.norm(up)
  left = np.cross(up, forward)
  left /= np.linalg.norm(left)
  matrix = np.column_stack((forward, left, up))
  return front, Rotation.from_matrix(matrix)


def _canonicalize_heading(
  evaluated: list[tuple[dict[str, np.ndarray], dict[str, Rotation]]],
) -> list[tuple[dict[str, np.ndarray], dict[str, Rotation]]]:
  """Rotate the clip so its initial horizontal heading is simulator +X."""
  first_root, first_rot = _root_pose(evaluated[0][0])
  del first_root
  initial_forward = first_rot.apply((1.0, 0.0, 0.0))
  yaw = float(np.arctan2(initial_forward[2], initial_forward[0]))
  alignment = Rotation.from_euler("y", yaw)
  return [
    (
      {name: alignment.apply(position) for name, position in positions.items()},
      rotations,
    )
    for positions, rotations in evaluated
  ]


def _spine_angle(
  positions: dict[str, np.ndarray], root_pos: np.ndarray, root_rot: Rotation
) -> float:
  rear_local = root_rot.inv().apply(positions["Hips"] - root_pos)
  return float(np.arctan2(rear_local[2], -rear_local[0]))


def main(
  input_file: str,
  output_file: str,
  scale: float = 0.005,
  base_height: float = 0.22,
  frame_start: int = 0,
  frame_end: int | None = None,
  ik_iterations: int = 30,
  project_lateral: bool = True,
) -> None:
  """Convert one BVH dog clip to Cheetah generalized-coordinate CSV.

  By default, lateral root displacement and yaw are projected out to match the
  current forward-only imitation task.  Set ``project_lateral=False`` when the
  clip itself is intended to contain turning or side-stepping.
  """
  root, frames, frame_time = _parse_bvh(input_file)
  frames = frames[frame_start:frame_end]
  if len(frames) < 2:
    raise ValueError("At least two BVH frames are required")

  evaluated = _canonicalize_heading([_evaluate_frame(root, frame) for frame in frames])
  first_root, _ = _root_pose(evaluated[0][0])
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
  output: list[np.ndarray] = []
  foot_errors: list[float] = []
  for positions, _ in tqdm(evaluated, desc="BVH -> Cheetah", unit="frame"):
    source_root, root_rot = _root_pose(positions)
    if project_lateral:
      forward = root_rot.apply((1.0, 0.0, 0.0))
      yaw = float(np.arctan2(forward[2], forward[0]))
      root_rot = Rotation.from_euler("y", yaw) * root_rot
    source_delta = source_root - first_root
    if project_lateral:
      source_delta[2] = 0.0
    root_delta = _COORD_ROT.apply(source_delta) * scale
    root_pos = np.array((root_delta[0], root_delta[1], base_height))
    qpos[:3] = root_pos
    qpos[3:7] = (_COORD_ROT * root_rot).as_quat(scalar_first=True)
    target_root_rot = _COORD_ROT * root_rot
    qpos[joint_qpos["body_pitch_joint"]] = _clip_joint(
      model,
      "body_pitch_joint",
      _spine_angle(positions, source_root, root_rot),
    )
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    target_feet = []
    for source_hip, source_foot, robot_hip in zip(
      _SOURCE_HIPS, _SOURCE_FEET, _ROBOT_HIPS, strict=True
    ):
      source_delta = root_rot.inv().apply(
        positions[source_foot] - positions[source_hip]
      )
      robot_hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_hip)
      target_feet.append(
        data.xpos[robot_hip_id] + target_root_rot.apply(source_delta * scale)
      )
    foot_errors.append(
      _solve_feet(
        model,
        data,
        qpos,
        np.asarray(target_feet),
        _LEG_JOINTS,
        _SITE_NAMES,
        ik_iterations,
      )
    )
    output.append(
      np.concatenate(
        [
          root_pos,
          target_root_rot.as_quat(),
          np.asarray([qpos[joint_qpos[name]] for name in _CHEETAH_JOINTS]),
        ]
      )
    )

  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(output_file, np.asarray(output), delimiter=",")
  print(
    f"[INFO] Saved {len(output)} frames ({1.0 / frame_time:.1f} Hz) to {output_file}; "
    f"foot IK mean={np.mean(foot_errors):.4f} m, "
    f"p95={np.percentile(foot_errors, 95):.4f} m"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
