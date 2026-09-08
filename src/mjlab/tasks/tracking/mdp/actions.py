"""Actions specific to reference-motion tracking tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp.commands import MotionCommand


@dataclass(kw_only=True)
class ReferenceJointPositionActionCfg(BaseActionCfg):
  """Joint-position residual action around the current motion reference."""

  command_name: str = "motion"

  def __post_init__(self) -> None:
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> ReferenceJointPositionAction:
    return ReferenceJointPositionAction(self, env)


class ReferenceJointPositionAction(BaseAction):
  """Apply policy residuals on top of the current reference joint pose."""

  cfg: ReferenceJointPositionActionCfg

  def __init__(self, cfg: ReferenceJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._command = cast(
      "MotionCommand", env.command_manager.get_term(cfg.command_name)
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    """Scale the residual without applying the final-target clip yet."""
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale + self._offset

  def apply_actions(self) -> None:
    reference = self._command.joint_pos[:, self._target_ids]
    target = reference + self._processed_actions

    if self.cfg.clip is not None:
      target = torch.clamp(
        target,
        min=self._clip[:, :, 0],
        max=self._clip[:, :, 1],
      )
    else:
      limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
      target = torch.clamp(target, min=limits[..., 0], max=limits[..., 1])

    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      target - encoder_bias,
      joint_ids=self._target_ids,
    )
