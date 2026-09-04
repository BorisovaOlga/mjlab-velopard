from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def gait_phase(env: ManagerBasedRlEnv, period: float) -> torch.Tensor:
  """Encode the periodic gait clock as sine and cosine."""
  phase = (env.episode_length_buf * env.step_dt / period) % 1.0
  angle = 2.0 * math.pi * phase
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)


class contact_gait_phase:
  """Encode four rotary-gallop states from ordered foot touchdown events.

  States advance as ``FL -> RR -> FR -> RL`` and are encoded as sine/cosine,
  preserving the two-dimensional observation used by existing checkpoints.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self.state = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    sequence: tuple[int, int, int, int] = (0, 3, 1, 2),
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    first_contact = sensor.compute_first_contact(dt=env.step_dt)

    # FL is an absolute synchronization event. Other touchdowns advance only
    # when they match the next expected event, so contact chatter or an
    # out-of-order touchdown cannot arbitrarily jump between phases.
    previous_state = self.state.clone()
    fl_touchdown = first_contact[:, sequence[0]]
    next_value = torch.where(
      fl_touchdown, torch.zeros_like(previous_state), previous_state
    )
    for next_state in range(1, 4):
      expected = sequence[next_state]
      should_advance = (
        ~fl_touchdown
        & (previous_state == next_state - 1)
        & first_contact[:, expected]
      )
      next_value = torch.where(
        should_advance, torch.full_like(next_value, next_state), next_value
      )
    self.state = next_value

    labels = ("fl_to_rr", "collected", "fr_to_rl", "extended")
    log = env.extras.setdefault("log", {})
    for state_index, label in enumerate(labels):
      log[f"Metrics/contact_phase_{label}_fraction"] = (
        (self.state == state_index).float().mean()
      )
    angle = 0.5 * math.pi * self.state.float()
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.state[env_ids] = 0


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
