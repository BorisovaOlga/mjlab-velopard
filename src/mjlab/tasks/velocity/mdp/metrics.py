from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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
