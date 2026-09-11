"""Convert a Cheetah generalized-coordinate trajectory to tracking NPZ.

The input CSV has one row per frame and the following columns::

    base_x, base_y, base_z, quat_x, quat_y, quat_z, quat_w, <13 joint positions>

The joint order is the MuJoCo order used by the articulated Cheetah.  The
converter computes body poses and velocities with MuJoCo forward kinematics,
which is the format consumed by ``MotionCommand``.
"""

from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.cheetah.env_cfgs import (
  _CHEETAH_JOINTS,
  reborn_cheetah_flat_tracking_env_cfg,
)


def _save_motion(log: dict[str, Any], output_file: str, output_fps: float) -> None:
  arrays = {key: np.stack(value, axis=0) for key, value in log.items() if key != "fps"}
  arrays["fps"] = np.asarray([output_fps], dtype=np.float32)
  np.savez_compressed(output_file, **arrays)


def main(
  input_file: str,
  output_file: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cpu",
  line_range: tuple[int, int] | None = None,
  rigid_spine: bool = False,
) -> None:
  """Convert Cheetah CSV to the NPZ contract used by the imitation task."""
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING] CUDA is unavailable; falling back to CPU.")
    device = "cpu"

  motion = MotionLoader(
    motion_file=input_file,
    input_fps=int(input_fps),
    output_fps=int(output_fps),
    device=device,
    line_range=line_range,
  )
  if motion.motion_dof_poss.shape[1] != len(_CHEETAH_JOINTS):
    raise ValueError(
      f"Expected {len(_CHEETAH_JOINTS)} Cheetah joint columns, "
      f"got {motion.motion_dof_poss.shape[1]}."
    )

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(
    reborn_cheetah_flat_tracking_env_cfg(rigid_spine=rigid_spine).scene,
    device=device,
  )
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot: Entity = scene["robot"]
  joint_indexes = robot.find_joints(_CHEETAH_JOINTS, preserve_order=True)[0]

  log: dict[str, Any] = {
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  scene.reset()
  for _ in tqdm(range(motion.output_frames), desc="Cheetah FK", unit="frame"):
    (
      (
        root_pos,
        root_quat,
        root_lin_vel,
        root_ang_vel,
        joint_pos,
        joint_vel,
      ),
      _,
    ) = motion.get_next_state()

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = root_pos
    root_state[:, :3] += scene.env_origins[:, :3]
    root_state[:, 3:7] = root_quat
    root_state[:, 7:10] = root_lin_vel
    root_state[:, 10:13] = root_ang_vel
    robot.write_root_state_to_sim(root_state)

    qpos = robot.data.default_joint_pos.clone()
    qvel = robot.data.default_joint_vel.clone()
    qpos[:, joint_indexes] = joint_pos
    qvel[:, joint_indexes] = joint_vel
    robot.write_joint_state_to_sim(qpos, qvel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

  _save_motion(log, output_file, output_fps)
  print(f"[INFO] Saved {motion.output_frames} frames to {output_file}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
