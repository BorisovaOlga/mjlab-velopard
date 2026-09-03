from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def forward_velocity_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward forward progress without saturating far below the target speed.

  Unlike the exponential tracking reward, this provides a useful learning signal
  even when a robot commanded to run fast initially moves very slowly.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  target_speed = torch.clamp(command[:, 0], min=1.0e-6)
  forward_speed = asset.data.root_link_lin_vel_b[:, 0]
  progress = torch.clamp(forward_speed / target_speed, min=0.0, max=1.0)
  env.extras["log"]["Metrics/forward_speed"] = forward_speed.mean()
  return progress


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


def planar_drift_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize lateral velocity and yaw rate for straight-line running."""
  asset: Entity = env.scene[asset_cfg.name]
  lateral_velocity = asset.data.root_link_lin_vel_b[:, 1]
  yaw_rate = asset.data.root_link_ang_vel_b[:, 2]
  env.extras["log"]["Metrics/lateral_speed_abs"] = lateral_velocity.abs().mean()
  env.extras["log"]["Metrics/yaw_rate_abs"] = yaw_rate.abs().mean()
  return torch.square(lateral_velocity) + torch.square(yaw_rate)


class world_straight_line_l2:
  """Penalize deviation from the world +X line through each reset position."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get(
      "asset_cfg", _DEFAULT_ASSET_CFG
    )
    asset: Entity = env.scene[self._asset_cfg.name]
    self.initial_y = asset.data.root_link_pos_w[:, 1].clone()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    lateral_position_scale: float = 2.0,
    heading_scale: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    """Measure lateral position relative to the current episode's start line."""
    asset: Entity = env.scene[asset_cfg.name]
    lateral_velocity_w = asset.data.root_link_lin_vel_w[:, 1]
    lateral_position = asset.data.root_link_pos_w[:, 1] - self.initial_y
    heading_error = torch.atan2(
      torch.sin(asset.data.heading_w), torch.cos(asset.data.heading_w)
    )
    env.extras["log"]["Metrics/world_lateral_speed_abs"] = (
      lateral_velocity_w.abs().mean()
    )
    env.extras["log"]["Metrics/world_lateral_position_abs"] = (
      lateral_position.abs().mean()
    )
    env.extras["log"]["Metrics/world_heading_error_abs"] = heading_error.abs().mean()
    return (
      torch.square(lateral_velocity_w)
      + lateral_position_scale * torch.square(lateral_position)
      + heading_scale * torch.square(heading_error)
    )

  def reset(self, env_ids: torch.Tensor) -> None:
    asset: Entity = self._env.scene[self._asset_cfg.name]
    self.initial_y[env_ids] = asset.data.root_link_pos_w[env_ids, 1]


def joint_deviation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize selected joint positions relative to their default pose."""
  asset: Entity = env.scene[asset_cfg.name]
  error = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.sum(torch.square(error), dim=1)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


class footfall_sequence:
  """Reward a cyclic order of foot contacts.

  The contact sensor slot order is supplied explicitly by ``sequence``.  A correct
  touchdown advances the expected foot, while an out-of-order touchdown is
  penalized without advancing the phase.  This makes the term invariant to gait
  frequency and avoids imposing an artificial clock on the learned gait.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.expected_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    sequence: tuple[int, ...],
    command_name: str,
    command_threshold: float = 0.3,
    wrong_contact_penalty: float = 1.0,
    actual_speed_threshold: float = 0.75,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    sequence_tensor = torch.tensor(sequence, device=env.device, dtype=torch.long)
    expected_foot = sequence_tensor[self.expected_phase]
    correct = first_contact.gather(1, expected_foot.unsqueeze(1)).squeeze(1)
    # A simultaneous landing must not count as a clean touchdown merely because
    # the expected foot happens to be among the landing feet.  Charge every
    # additional touchdown as an ordering error.
    expected_mask = torch.nn.functional.one_hot(
      expected_foot, num_classes=first_contact.shape[1]
    ).bool()
    wrong_count = (first_contact & ~expected_mask).float().sum(dim=1)
    forward_speed = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    active = (command[:, 0] > command_threshold) & (
      forward_speed > actual_speed_threshold
    )

    reward = correct.float() - wrong_contact_penalty * wrong_count
    reward *= active.float()
    self.expected_phase = torch.where(
      correct & active,
      (self.expected_phase + 1) % len(sequence),
      self.expected_phase,
    )
    env.extras["log"]["Metrics/gallop_correct_touchdown"] = correct.float().mean()
    return reward

  def reset(self, env_ids: torch.Tensor) -> None:
    self.expected_phase[env_ids] = 0


def clock_gait(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  period: float,
  offsets: tuple[float, ...],
  stance_ratio: float,
  command_threshold: float = 0.15,
) -> torch.Tensor:
  """Reward contacts matching a periodic per-foot stance schedule."""
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  assert sensor.data.current_contact_time is not None
  global_phase = (env.episode_length_buf * env.step_dt / period).unsqueeze(1)
  offset = torch.tensor(offsets, device=env.device, dtype=global_phase.dtype)
  desired_contact = ((global_phase + offset) % 1.0) < stance_ratio
  actual_contact = sensor.data.current_contact_time > 0.0
  # A correct stance is worth four times a correct swing sample.  Otherwise a
  # policy with all feet permanently airborne exploits the longer swing windows.
  correct_stance = desired_contact & actual_contact
  correct_swing = (~desired_contact) & (~actual_contact)
  agreement = (correct_stance.float() + 0.25 * correct_swing.float()).mean(dim=1)
  active = command[:, 0] > command_threshold
  reward = agreement * active.float()
  env.extras["log"]["Metrics/clock_gait_agreement"] = reward.mean()
  return reward


def feline_gallop_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  period: float,
  stance_intervals: tuple[tuple[float, float], ...],
  command_threshold: float = 1.0,
  correct_swing_reward: float = 0.25,
  false_contact_penalty: float = 2.0,
  missed_contact_penalty: float = 1.0,
  actual_speed_threshold: float = 0.75,
) -> torch.Tensor:
  """Score an exact rotary-gallop contact schedule.

  False contacts receive a larger negative score than a correct stance.  This
  prevents the policy from exploiting the schedule by keeping every foot on the
  ground and collecting credit for whichever foot is currently expected.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  assert sensor.data.current_contact_time is not None
  phase = (env.episode_length_buf * env.step_dt / period) % 1.0
  desired_contact = torch.stack(
    tuple((phase >= start) & (phase < end) for start, end in stance_intervals),
    dim=1,
  )
  actual_contact = sensor.data.current_contact_time > 0.0
  correct_stance = desired_contact & actual_contact
  correct_swing = ~desired_contact & ~actual_contact
  false_contact = ~desired_contact & actual_contact
  missed_contact = desired_contact & ~actual_contact
  score = (
    correct_stance.float()
    + correct_swing_reward * correct_swing.float()
    - false_contact_penalty * false_contact.float()
    - missed_contact_penalty * missed_contact.float()
  ).mean(dim=1)
  forward_speed = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
  active = (command[:, 0] > command_threshold) & (
    forward_speed > actual_speed_threshold
  )
  reward = score * active.float()
  agreement = (desired_contact == actual_contact).float().mean(dim=1)
  all_feet_contact = actual_contact.all(dim=1) & active
  env.extras["log"]["Metrics/feline_gallop_contact_agreement"] = (
    agreement * active.float()
  ).mean()
  env.extras["log"]["Metrics/all_feet_contact_fraction"] = (
    all_feet_contact.float().mean()
  )
  return reward


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize deviation from the default pose for near-zero commands."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  active = command[:, 0] <= command_threshold
  return torch.sum(torch.square(error), dim=1) * active.float()


class stride_length:
  """Reward forward foot excursion from lift-off to the next touchdown."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    num_feet = len(cfg.params["asset_cfg"].site_ids)
    self.takeoff_x = torch.zeros((env.num_envs, num_feet), device=env.device)
    self.has_takeoff = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.bool
    )
    del asset

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    base_stride: float = 0.10,
    speed_slope: float = 0.035,
    max_stride: float = 0.24,
    std: float = 0.04,
    command_threshold: float = 0.15,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    foot_delta_w = asset.data.site_pos_w[
      :, asset_cfg.site_ids
    ] - asset.data.root_link_pos_w.unsqueeze(1)
    root_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(
      -1, foot_delta_w.shape[1], -1
    )
    foot_delta_b = quat_apply_inverse(root_quat, foot_delta_w)
    foot_x = foot_delta_b[:, :, 0]

    first_air = sensor.compute_first_air(dt=env.step_dt)
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    self.takeoff_x = torch.where(first_air, foot_x, self.takeoff_x)
    self.has_takeoff |= first_air

    excursion = foot_x - self.takeoff_x
    target = torch.clamp(
      base_stride + speed_slope * command[:, 0], max=max_stride
    ).unsqueeze(1)
    landing = first_contact & self.has_takeoff
    reward = torch.exp(-torch.square(excursion - target) / std**2) * landing.float()
    active = command[:, 0] > command_threshold
    reward = reward.sum(dim=1) * active.float()
    landed = landing.float().sum()
    mean_excursion = (excursion * landing.float()).sum() / torch.clamp(landed, min=1)
    env.extras["log"]["Metrics/stride_length_at_landing"] = mean_excursion
    self.has_takeoff &= ~first_contact
    return reward

  def reset(self, env_ids: torch.Tensor) -> None:
    self.takeoff_x[env_ids] = 0.0
    self.has_takeoff[env_ids] = False


def flight_phase(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  speed_threshold: float = 2.0,
  min_air_time: float = 0.025,
  phase_period: float | None = None,
  phase_windows: tuple[tuple[float, float], ...] = (),
  actual_speed_threshold: float = 0.75,
) -> torch.Tensor:
  """Reward a true flight phase, when all four feet are clear of the ground."""
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None
  all_feet_airborne = torch.all(current_air_time > min_air_time, dim=1)
  forward_speed = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
  active = (command[:, 0] > speed_threshold) & (
    forward_speed > actual_speed_threshold
  )
  in_target_phase = torch.ones_like(all_feet_airborne)
  if phase_period is not None and phase_windows:
    phase = (env.episode_length_buf * env.step_dt / phase_period) % 1.0
    in_target_phase = torch.zeros_like(all_feet_airborne)
    for start, end in phase_windows:
      in_target_phase |= (phase >= start) & (phase < end)
  flight = all_feet_airborne & active & in_target_phase
  env.extras["log"]["Metrics/flight_phase_fraction"] = flight.float().mean()
  return flight.float()


def flight_contact_violation(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  period: float,
  phase_windows: tuple[tuple[float, float], ...],
  actual_speed_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize every grounded foot inside commanded flight windows."""
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  assert sensor.data.current_contact_time is not None
  phase = (env.episode_length_buf * env.step_dt / period) % 1.0
  in_flight_window = torch.zeros_like(phase, dtype=torch.bool)
  for start, end in phase_windows:
    in_flight_window |= (phase >= start) & (phase < end)
  actual_contact = sensor.data.current_contact_time > 0.0
  contact_fraction = actual_contact.float().mean(dim=1)
  forward_speed = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
  active = (command[:, 0] > 0.0) & (forward_speed > actual_speed_threshold)
  violation = contact_fraction * (in_flight_window & active).float()
  env.extras["log"]["Metrics/flight_contact_violation"] = violation.mean()
  return violation


def excess_foot_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  maximum_contacts: int = 1,
  actual_speed_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize support overlap beyond the allowed number of grounded feet."""
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  assert sensor.data.current_contact_time is not None
  num_contacts = (sensor.data.current_contact_time > 0.0).float().sum(dim=1)
  excess = torch.clamp(num_contacts - maximum_contacts, min=0.0)
  forward_speed = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
  active = (command[:, 0] > 0.0) & (forward_speed > actual_speed_threshold)
  normalized_excess = excess / max(1, 4 - maximum_contacts)
  penalty = normalized_excess * active.float()
  env.extras["log"]["Metrics/excess_foot_contacts"] = penalty.mean()
  return penalty


def sustained_flight(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  target_duration: float = 0.06,
  actual_speed_threshold: float = 1.0,
) -> torch.Tensor:
  """Reward the measured duration for which all feet remain airborne."""
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  assert sensor.data.current_air_time is not None
  full_flight_time = sensor.data.current_air_time.min(dim=1).values
  forward_speed = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
  active = (command[:, 0] > 0.0) & (forward_speed > actual_speed_threshold)
  reward = torch.clamp(full_flight_time / target_duration, min=0.0, max=1.0)
  reward *= active.float()
  env.extras["log"]["Metrics/full_flight_duration"] = full_flight_time.mean()
  return reward


class early_gait_cycle:
  """Penalize an FL touchdown that repeats before a minimum cycle duration."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.last_touchdown_time = torch.full(
      (env.num_envs,), -1.0, device=env.device, dtype=torch.float
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    minimum_period: float = 0.30,
    foot_index: int = 0,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    first_contact = sensor.compute_first_contact(dt=env.step_dt)[:, foot_index]
    current_time = env.episode_length_buf * env.step_dt
    has_previous = self.last_touchdown_time >= 0.0
    measured_period = current_time - self.last_touchdown_time
    valid_touchdown = first_contact & has_previous
    early_fraction = torch.clamp(
      (minimum_period - measured_period) / minimum_period, min=0.0, max=1.0
    )
    penalty = early_fraction * valid_touchdown.float()
    self.last_touchdown_time = torch.where(
      first_contact, current_time, self.last_touchdown_time
    )
    measured_sum = (measured_period * valid_touchdown.float()).sum()
    measured_count = valid_touchdown.float().sum()
    env.extras["log"]["Metrics/measured_gait_period"] = measured_sum / torch.clamp(
      measured_count, min=1.0
    )
    env.extras["log"]["Metrics/early_gait_cycle_fraction"] = penalty.mean()
    return penalty

  def reset(self, env_ids: torch.Tensor) -> None:
    self.last_touchdown_time[env_ids] = -1.0


def extended_flight_posture(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  speed_threshold: float = 2.0,
  min_air_time: float = 0.025,
  front_target_x: float = 0.27,
  hind_target_x: float = -0.24,
  position_std: float = 0.06,
  symmetry_std: float = 0.04,
  phase_period: float | None = None,
  phase_windows: tuple[tuple[float, float], ...] = (),
  metric_prefix: str = "extended_flight",
  actual_speed_threshold: float = 0.75,
) -> torch.Tensor:
  """Reward the stretched feline posture during the flight phase.

  Site order must be ``(FL, FR, RL, RR)``.  Positions are expressed in the
  root-body frame, making the reward invariant to world position and heading.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None

  foot_delta_w = asset.data.site_pos_w[
    :, asset_cfg.site_ids
  ] - asset.data.root_link_pos_w.unsqueeze(1)
  root_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(
    -1, foot_delta_w.shape[1], -1
  )
  foot_x = quat_apply_inverse(root_quat, foot_delta_w)[:, :, 0]
  front_x = foot_x[:, :2]
  hind_x = foot_x[:, 2:]
  front_mean = front_x.mean(dim=1)
  hind_mean = hind_x.mean(dim=1)

  position_error = torch.square(front_mean - front_target_x) + torch.square(
    hind_mean - hind_target_x
  )
  symmetry_error = torch.square(front_x[:, 0] - front_x[:, 1]) + torch.square(
    hind_x[:, 0] - hind_x[:, 1]
  )
  # Cauchy kernels retain a useful gradient when the initial posture is far
  # from the target; the previous product of narrow exponentials vanished.
  position_reward = 1.0 / (1.0 + position_error / position_std**2)
  symmetry_reward = 1.0 / (1.0 + symmetry_error / symmetry_std**2)
  in_flight = torch.all(current_air_time > min_air_time, dim=1)
  in_target_phase = torch.ones_like(in_flight)
  if phase_period is not None and phase_windows:
    phase = (env.episode_length_buf * env.step_dt / phase_period) % 1.0
    in_target_phase = torch.zeros_like(in_flight)
    for start, end in phase_windows:
      in_target_phase |= (phase >= start) & (phase < end)
  forward_speed = asset.data.root_link_lin_vel_b[:, 0]
  active = (command[:, 0] > speed_threshold) & (
    forward_speed > actual_speed_threshold
  )
  reward = (
    position_reward
    * symmetry_reward
    * in_flight.float()
    * in_target_phase.float()
    * active.float()
  )

  env.extras["log"][f"Metrics/{metric_prefix}_foot_span"] = (
    front_mean - hind_mean
  ).mean()
  env.extras["log"][f"Metrics/{metric_prefix}_front_foot_x"] = front_mean.mean()
  env.extras["log"][f"Metrics/{metric_prefix}_hind_foot_x"] = hind_mean.mean()
  return reward


def jumping_in_place(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  minimum_forward_speed: float = 0.75,
  min_air_time: float = 0.025,
) -> torch.Tensor:
  """Penalize flight without meaningful forward translation."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_air_time is not None
  all_feet_airborne = torch.all(sensor.data.current_air_time > min_air_time, dim=1)
  forward_speed = asset.data.root_link_lin_vel_b[:, 0]
  speed_deficit = torch.clamp(
    (minimum_forward_speed - forward_speed) / minimum_forward_speed,
    min=0.0,
    max=1.0,
  )
  penalty = all_feet_airborne.float() * speed_deficit
  env.extras["log"]["Metrics/jumping_in_place_fraction"] = (
    (all_feet_airborne & (forward_speed < minimum_forward_speed)).float().mean()
  )
  return penalty


class hind_propulsion:
  """Reward forward acceleration generated during the hind-leg stance."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.previous_speed = torch.zeros(env.num_envs, device=env.device)
    self.initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period: float,
    push_windows: tuple[tuple[float, float], ...],
    target_acceleration: float = 8.0,
    speed_threshold: float = 2.0,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    assert sensor.data.current_contact_time is not None
    speed = asset.data.root_link_lin_vel_b[:, 0]
    acceleration = torch.where(
      self.initialized,
      (speed - self.previous_speed) / env.step_dt,
      torch.zeros_like(speed),
    )
    self.previous_speed = speed.clone()
    self.initialized[:] = True
    phase = (env.episode_length_buf * env.step_dt / period) % 1.0
    in_push_phase = torch.zeros_like(phase, dtype=torch.bool)
    for start, end in push_windows:
      in_push_phase |= (phase >= start) & (phase < end)
    hind_contact = (sensor.data.current_contact_time[:, 2:] > 0.0).any(dim=1)
    active = command[:, 0] > speed_threshold
    reward = torch.clamp(acceleration / target_acceleration, min=0.0, max=1.0)
    reward *= (in_push_phase & hind_contact & active).float()
    env.extras["log"]["Metrics/hind_push_acceleration"] = torch.clamp(
      acceleration, min=0.0
    ).mean()
    return reward

  def reset(self, env_ids: torch.Tensor) -> None:
    self.previous_speed[env_ids] = 0.0
    self.initialized[env_ids] = False


def spine_phase_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  period: float,
  sensor_name: str,
  stance_intervals: tuple[tuple[float, float], ...],
  minimum_contact_agreement: float = 0.75,
  amplitude: float = 0.4,
  phase_offset: float = 0.0,
  std: float = 0.18,
  speed_threshold: float = 1.8,
  actual_speed_threshold: float = 0.5,
) -> torch.Tensor:
  """Synchronize the spine only while feet follow the rotary contact clock."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  phase = (env.episode_length_buf * env.step_dt / period + phase_offset) % 1.0
  target = amplitude * torch.cos(2.0 * torch.pi * phase)
  spine_pos = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(1)
  reward = torch.exp(-torch.square(spine_pos - target) / std**2)
  assert sensor.data.current_contact_time is not None
  actual_contact = sensor.data.current_contact_time > 0.0
  expected_contact = torch.zeros_like(actual_contact)
  for foot_index, (start, end) in enumerate(stance_intervals):
    expected_contact[:, foot_index] = (phase >= start) & (phase < end)
  contact_agreement = (actual_contact == expected_contact).float().mean(dim=1)
  active = (command[:, 0] > speed_threshold) & (
    asset.data.root_link_lin_vel_b[:, 0] > actual_speed_threshold
  ) & (contact_agreement >= minimum_contact_agreement)
  env.extras["log"]["Metrics/spine_phase_error"] = torch.abs(spine_pos - target).mean()
  env.extras["log"]["Metrics/spine_contact_gate_fraction"] = active.float().mean()
  return reward * active.float()


class spine_flexion:
  """Reward zero-centered spine oscillation over a finite time window."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    window_s = cfg.params["window_s"]
    self.window_steps = max(2, round(window_s / env.step_dt))
    self.history = torch.zeros(
      (env.num_envs, self.window_steps), device=env.device, dtype=torch.float32
    )
    self.index = 0
    self.samples = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    window_s: float,
    speed_threshold: float = 1.8,
    actual_speed_threshold: float = 0.5,
    target_range: float = 0.8,
    range_std: float = 0.2,
    center_std: float = 0.12,
  ) -> torch.Tensor:
    del window_s
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    spine_pos = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(1)
    self.history[:, self.index] = spine_pos
    self.index = (self.index + 1) % self.window_steps
    self.samples = min(self.samples + 1, self.window_steps)

    spine_range = self.history.max(dim=1).values - self.history.min(dim=1).values
    spine_center = self.history.mean(dim=1)
    range_reward = torch.exp(-torch.square(spine_range - target_range) / range_std**2)
    centered_reward = torch.exp(-torch.square(spine_center) / center_std**2)
    active = (
      (command[:, 0] > speed_threshold)
      & (asset.data.root_link_lin_vel_b[:, 0] > actual_speed_threshold)
      & (self.samples == self.window_steps)
    )
    reward = range_reward * centered_reward * active.float()
    env.extras["log"]["Metrics/spine_range"] = spine_range.mean()
    env.extras["log"]["Metrics/spine_center"] = spine_center.mean()
    return reward

  def reset(self, env_ids: torch.Tensor) -> None:
    self.history[env_ids] = 0.0


class bilateral_foot_amplitude:
  """Penalize unequal left/right foot amplitudes over one complete gait cycle."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    period = cfg.params["period"]
    self.window_steps = max(2, round(period / env.step_dt))
    self.history = torch.zeros(
      (env.num_envs, self.window_steps, 4, 2),
      device=env.device,
      dtype=torch.float32,
    )
    self.index = 0
    self.samples = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    period: float,
    amplitude_std: float = 0.02,
  ) -> torch.Tensor:
    del period
    asset: Entity = env.scene[asset_cfg.name]
    foot_delta_w = asset.data.site_pos_w[
      :, asset_cfg.site_ids
    ] - asset.data.root_link_pos_w.unsqueeze(1)
    root_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(
      -1, foot_delta_w.shape[1], -1
    )
    foot_pos_b = quat_apply_inverse(root_quat, foot_delta_w)
    self.history[:, self.index] = foot_pos_b[:, :, (0, 2)]
    self.index = (self.index + 1) % self.window_steps
    self.samples = min(self.samples + 1, self.window_steps)

    amplitude = self.history.max(dim=1).values - self.history.min(dim=1).values
    # Site order: FL, FR, RL, RR. Compare sagittal (x) and vertical (z)
    # amplitudes within the front and hind pairs.
    front_error = amplitude[:, 0] - amplitude[:, 1]
    hind_error = amplitude[:, 2] - amplitude[:, 3]
    cost = torch.square(front_error / amplitude_std).sum(dim=1) + torch.square(
      hind_error / amplitude_std
    ).sum(dim=1)
    cost *= float(self.samples == self.window_steps)

    env.extras["log"]["Metrics/front_left_vertical_amplitude"] = amplitude[
      :, 0, 1
    ].mean()
    env.extras["log"]["Metrics/front_right_vertical_amplitude"] = amplitude[
      :, 1, 1
    ].mean()
    env.extras["log"]["Metrics/front_vertical_amplitude_difference"] = (
      front_error[:, 1].abs().mean()
    )
    env.extras["log"]["Metrics/front_sagittal_amplitude_difference"] = (
      front_error[:, 0].abs().mean()
    )
    return cost

  def reset(self, env_ids: torch.Tensor) -> None:
    self.history[env_ids] = 0.0


class bilateral_actuator_power:
  """Penalize unequal mean mechanical power between left and right sides."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    period = cfg.params["period"]
    self.window_steps = max(2, round(period / env.step_dt))
    self.left_history = torch.zeros(
      (env.num_envs, self.window_steps), device=env.device
    )
    self.right_history = torch.zeros_like(self.left_history)
    self.index = 0
    self.samples = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    left_asset_cfg: SceneEntityCfg,
    right_asset_cfg: SceneEntityCfg,
    period: float,
    metric_name: str = "left_right_power_difference",
  ) -> torch.Tensor:
    del period
    asset: Entity = env.scene[left_asset_cfg.name]
    left_power = torch.abs(
      asset.data.qfrc_actuator[:, left_asset_cfg.joint_ids]
      * asset.data.joint_vel[:, left_asset_cfg.joint_ids]
    ).sum(dim=1)
    right_power = torch.abs(
      asset.data.qfrc_actuator[:, right_asset_cfg.joint_ids]
      * asset.data.joint_vel[:, right_asset_cfg.joint_ids]
    ).sum(dim=1)
    self.left_history[:, self.index] = left_power
    self.right_history[:, self.index] = right_power
    self.index = (self.index + 1) % self.window_steps
    self.samples = min(self.samples + 1, self.window_steps)

    left_mean = self.left_history.mean(dim=1)
    right_mean = self.right_history.mean(dim=1)
    normalized_difference = torch.abs(left_mean - right_mean) / torch.clamp(
      left_mean + right_mean, min=1.0e-6
    )
    normalized_difference *= float(self.samples == self.window_steps)
    env.extras["log"][f"Metrics/{metric_name}"] = normalized_difference.mean()
    return normalized_difference

  def reset(self, env_ids: torch.Tensor) -> None:
    self.left_history[env_ids] = 0.0
    self.right_history[env_ids] = 0.0


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
