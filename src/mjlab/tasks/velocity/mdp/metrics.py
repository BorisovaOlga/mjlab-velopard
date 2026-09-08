from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.metrics_manager import MetricsTermCfg


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def mechanical_cost_of_transport(
  env: ManagerBasedRlEnv,
  mass: float,
  gravity: float = 9.81,
  minimum_speed: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Mechanical cost of transport based on absolute actuator power."""
  asset: Entity = env.scene[asset_cfg.name]
  mechanical_power = torch.sum(
    torch.abs(asset.data.qfrc_actuator * asset.data.joint_vel), dim=1
  )
  forward_speed = torch.clamp(
    torch.abs(asset.data.root_link_lin_vel_b[:, 0]), min=minimum_speed
  )
  return mechanical_power / (mass * gravity * forward_speed)


def joint_abs_torque(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Absolute joint-space actuator torque for one selected joint."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(asset.data.qfrc_actuator[:, asset_cfg.joint_ids]).squeeze(1)


def joint_abs_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Absolute angular velocity for one selected joint."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]).squeeze(1)


def joint_abs_mechanical_power(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Sum absolute mechanical power over selected joints."""
  asset: Entity = env.scene[asset_cfg.name]
  torque = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
  velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(torque * velocity), dim=1)


def joint_positive_mechanical_power(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Sum positive actuator work rate over selected joints."""
  asset: Entity = env.scene[asset_cfg.name]
  power = (
    asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    * asset.data.joint_vel[:, asset_cfg.joint_ids]
  )
  return torch.clamp(power, min=0.0).sum(dim=1)


def joint_absorbed_mechanical_power(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Sum the magnitude of negative actuator work rate over selected joints."""
  asset: Entity = env.scene[asset_cfg.name]
  power = (
    asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    * asset.data.joint_vel[:, asset_cfg.joint_ids]
  )
  return torch.clamp(-power, min=0.0).sum(dim=1)


class joint_position_range:
  """Track the position range reached by selected joints within an episode."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    joint_count = len(asset_cfg.joint_ids)
    shape = (env.num_envs, joint_count)
    self.minimum = torch.full(shape, torch.inf, device=env.device)
    self.maximum = torch.full(shape, -torch.inf, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    position = asset.data.joint_pos[:, asset_cfg.joint_ids]
    torch.minimum(self.minimum, position, out=self.minimum)
    torch.maximum(self.maximum, position, out=self.maximum)
    return (self.maximum - self.minimum).mean(dim=1)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.minimum[env_ids] = torch.inf
    self.maximum[env_ids] = -torch.inf


def actuator_saturation_fraction(
  env: ManagerBasedRlEnv,
  effort_limit: float,
  asset_cfg: SceneEntityCfg,
  threshold: float = 0.95,
) -> torch.Tensor:
  """Fraction of selected joints operating near their effort limit."""
  asset: Entity = env.scene[asset_cfg.name]
  torque = torch.abs(asset.data.qfrc_actuator[:, asset_cfg.joint_ids])
  return (torque >= threshold * effort_limit).float().mean(dim=1)


def joint_speed_limit_fraction(
  env: ManagerBasedRlEnv,
  velocity_limit: float,
  asset_cfg: SceneEntityCfg,
  threshold: float = 0.95,
) -> torch.Tensor:
  """Fraction of selected joints running near the motor no-load speed."""
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
  return (speed >= threshold * velocity_limit).float().mean(dim=1)


def spine_power_fraction(
  env: ManagerBasedRlEnv,
  spine_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Fraction of absolute actuator power supplied by the spine joint."""
  asset: Entity = env.scene[asset_cfg.name]
  total_power = torch.sum(
    torch.abs(asset.data.qfrc_actuator * asset.data.joint_vel), dim=1
  )
  spine_power = torch.sum(
    torch.abs(
      asset.data.qfrc_actuator[:, spine_cfg.joint_ids]
      * asset.data.joint_vel[:, spine_cfg.joint_ids]
    ),
    dim=1,
  )
  return spine_power / torch.clamp(total_power, min=1.0e-6)
