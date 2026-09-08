from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # On the initial reset (before any env step) the robot is still at its spawn
  # pose rather than a walked-to position, so ``distance`` is meaningless and
  # would spuriously promote every env from level 0 to 1, ignoring
  # ``max_init_terrain_level``. Freeze levels on that first reset.
  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


class adaptive_command_velocity:
  """Expand forward commands after sustained success near the stage frontier."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.score_ema = 0.0
    self.last_advance_score = 0.0
    self.episodes_since_update = 0
    self.score_initialized = False
    self.stage_initialized = False
    self.stage_started_step = int(env.common_step_counter)
    self.current_max = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    command_name: str,
    reward_name: str,
    max_velocity: float,
    increment: float = 0.25,
    success_threshold: float = 0.8,
    min_episodes: int = 4096,
    min_steps_per_stage: int = 1200,
    frontier_fraction: float = 0.75,
    ema_alpha: float = 0.2,
  ) -> dict[str, torch.Tensor]:
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command '{command_name}' not found."
    command_cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
    velocity_command = cast(UniformVelocityCommand, command_term)
    if not self.stage_initialized:
      self.current_max = command_cfg.ranges.lin_vel_x[1]
      self.stage_initialized = True

    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    episode_steps = env.episode_length_buf[env_ids]
    minimum_command = command_cfg.ranges.lin_vel_x[0]
    frontier = minimum_command + frontier_fraction * (
      self.current_max - minimum_command
    )
    episode_targets = velocity_command.target_vel_command_b[env_ids, 0]
    eligible = (episode_steps > 0) & (episode_targets >= frontier)
    if eligible.any():
      reward_cfg = env.reward_manager.get_term_cfg(reward_name)
      weighted_sum = env.reward_manager._episode_sums[reward_name][env_ids][eligible]
      denominator = (
        episode_steps[eligible].float() * env.step_dt * abs(reward_cfg.weight)
      )
      batch_score = torch.mean(weighted_sum / torch.clamp(denominator, min=1.0e-6))
      score = float(batch_score.clamp(min=0.0, max=1.0).item())
      if not self.score_initialized:
        self.score_ema = score
        self.score_initialized = True
      else:
        self.score_ema = (1.0 - ema_alpha) * self.score_ema + ema_alpha * score
      self.episodes_since_update += int(eligible.sum().item())

      stage_age = int(env.common_step_counter) - self.stage_started_step
      if (
        self.episodes_since_update >= min_episodes
        and stage_age >= min_steps_per_stage
        and self.score_ema >= success_threshold
        and self.current_max < max_velocity
      ):
        self.last_advance_score = self.score_ema
        self.current_max = min(max_velocity, self.current_max + increment)
        command_cfg.ranges.lin_vel_x = (
          minimum_command,
          self.current_max,
        )
        self.episodes_since_update = 0
        self.score_ema = 0.0
        self.score_initialized = False
        self.stage_started_step = int(env.common_step_counter)

    frontier = minimum_command + frontier_fraction * (
      self.current_max - minimum_command
    )
    stage_age = int(env.common_step_counter) - self.stage_started_step
    return {
      "lin_vel_x_max": torch.tensor(self.current_max),
      "tracking_score_ema": torch.tensor(self.score_ema),
      "last_advance_score": torch.tensor(self.last_advance_score),
      "frontier_command": torch.tensor(frontier),
      "eligible_episodes": torch.tensor(self.episodes_since_update),
      "stage_age_steps": torch.tensor(stage_age),
    }
