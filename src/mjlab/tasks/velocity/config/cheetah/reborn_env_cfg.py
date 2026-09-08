"""Research configuration for learning useful Cheetah spinal undulation."""

from mjlab.asset_zoo.robots.cheetah import get_reborn_cheetah_robot_cfg
from mjlab.asset_zoo.robots.cheetah.cheetah_constants import (
  HIP_ACTUATOR,
  KNEE_ACTUATOR,
  SPINE_ACTUATOR,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationTermCfg, TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .env_cfgs import cheetah_flat_env_cfg

_SPINE_JOINT = ("body_pitch_joint",)
_FEET = ("FL", "FR", "RL", "RR")
_LEG_JOINTS = (
  "left_front_hip_roll_joint",
  "left_front_hip_pitch_joint",
  "left_front_knee_pitch_joint",
  "right_front_hip_roll_joint",
  "right_front_hip_pitch_joint",
  "right_front_knee_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_pitch_joint",
  "left_knee_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_pitch_joint",
  "right_knee_pitch_joint",
)
_HIP_PITCH_JOINTS = (
  "left_front_hip_pitch_joint",
  "right_front_hip_pitch_joint",
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
)
_HIP_JOINTS = tuple(name for name in _LEG_JOINTS if "hip_" in name)
_KNEE_JOINTS = tuple(name for name in _LEG_JOINTS if "knee_" in name)


def _joints(names: tuple[str, ...]) -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=names, preserve_order=True)


def _feet() -> SceneEntityCfg:
  return SceneEntityCfg("robot", site_names=_FEET, preserve_order=True)


def reborn_cheetah_flat_env_cfg(
  play: bool = False,
  *,
  rigid_spine: bool = False,
  spine_reward: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create the phase-assisted, leg-coupled Cheetah velocity task."""
  cfg = cheetah_flat_env_cfg(play=play)
  cfg.scene.entities["robot"] = get_reborn_cheetah_robot_cfg(rigid_spine=rigid_spine)

  # Expose phase only for the bootstrap schedule.  It is not a mocap target or
  # an imitation input; the policy still has to discover the dynamics.
  gait_period = 0.5
  for group in cfg.observations.values():
    group.terms["gait_phase"] = ObservationTermCfg(
      func=mdp.gait_phase,
      params={"period": gait_period},
    )

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = {
    r".*hip_roll_joint": 0.35,
    r".*hip_pitch_joint": 1.10,
    r".*knee_pitch_joint": 0.80,
    r"body_pitch_joint": 0.65,
  }
  action.clip = {
    r".*hip_roll_joint": (-0.50, 0.50),
    r".*hip_pitch_joint": (-1.30, 1.30),
    r".*knee_pitch_joint": (-1.00, 0.50),
    r"body_pitch_joint": (0.0, 0.0) if rigid_spine else (-0.60, 0.30),
  }

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.resampling_time_range = (20.0, 20.0)
  twist_cmd.acceleration_limit_range = (1.0, 2.0)
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_world_envs = 1.0
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.ranges.lin_vel_x = (5.0, 5.0) if play else (0.5, 1.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  # Disturbance robustness is a later-stage concern. Early pushes obscure
  # whether failures come from gait discovery or recovery behavior.
  cfg.events.pop("push_robot", None)
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

  old_rewards = cfg.rewards
  track_velocity = old_rewards["track_linear_velocity"]
  track_velocity.weight = 4.0
  track_velocity.params["std"] = 0.6
  track_yaw = old_rewards["track_angular_velocity"]
  track_yaw.weight = 0.5
  upright = old_rewards["upright"]
  upright.weight = 0.5
  pose = old_rewards["pose"]
  pose.weight = 0.1
  air_time = old_rewards["air_time"]
  air_time.weight = 0.15
  foot_clearance = old_rewards["foot_clearance"]
  foot_clearance.weight = -0.15
  foot_swing_height = old_rewards["foot_swing_height"]
  foot_swing_height.weight = -0.1
  foot_slip = old_rewards["foot_slip"]
  foot_slip.weight = -0.2
  soft_landing = old_rewards["soft_landing"]
  soft_landing.weight = -1.0e-4
  joint_limits = old_rewards["dof_pos_limits"]
  joint_limits.weight = -2.0
  termination = old_rewards["termination_penalty"]
  termination.weight = -25.0

  cfg.rewards = {
    "track_linear_velocity": track_velocity,
    "forward_velocity_progress": RewardTermCfg(
      func=mdp.forward_velocity_progress,
      weight=1.0,
      params={"command_name": "twist"},
    ),
    "track_angular_velocity": track_yaw,
    "planar_drift": RewardTermCfg(func=mdp.planar_drift_l2, weight=-0.2),
    "straight_line_deviation": RewardTermCfg(
      func=mdp.straight_line_deviation_l2,
      weight=-0.5,
      params={"lateral_tolerance": 0.25, "heading_tolerance": 0.26},
    ),
    "upright": upright,
    "pose": pose,
    "gallop_sequence": RewardTermCfg(
      func=mdp.gallop_pair_sequence,
      weight=0.75,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        # Bootstrap from the observed ~0.17 s within-pair delay; the phase
        # reward still drives this interval toward the 0.12-cycle target.
        "pair_window": 0.24,
        "min_period": 0.12,
        "command_threshold": 1.0,
        "wrong_pair_penalty": 0.25,
      },
    ),
    "gallop_contact_schedule": RewardTermCfg(
      func=mdp.feline_gallop_contacts,
      weight=1.5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "period": gait_period,
        # FL, FR, RL, RR: fore support, flight, hind support, flight.
        "stance_intervals": (
          (0.16, 0.34),
          (0.22, 0.40),
          (0.61, 0.80),
          (0.54, 0.73),
        ),
        "command_threshold": 1.5,
        "stance_weight": 3.0,
        "swing_weight": 0.25,
      },
    ),
    "gallop_pair_phase": RewardTermCfg(
      func=mdp.gallop_pair_phase,
      weight=0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        # Rotary sequence: RL -> RR -> FR -> FL. Only within-pair offsets
        # are constrained; the phase between the pairs remains free.
        "pairs": ((1, 0), (2, 3)),
        "target_offsets": (0.12, 0.12),
        "initial_period": 0.5,
        "offset_std": 0.15,
        "min_period": 0.15,
        "max_period": 1.0,
        "max_pair_age_cycles": 1.5,
        "command_threshold": 1.5,
      },
    ),
    "spine_leg_coordination": RewardTermCfg(
      func=mdp.spine_leg_coordination,
      weight=0.0 if rigid_spine or not spine_reward else 0.5,
      params={
        "command_name": "twist",
        "spine_cfg": _joints(_SPINE_JOINT),
        "leg_cfg": _joints(_HIP_PITCH_JOINTS),
        "coordination_sign": -1.0,
        "negative_limit": 0.60,
        "positive_limit": 0.30,
        "filter_time_constant": 0.25,
        "amplitude_boost": 0.5,
        "excess_penalty": 8.0,
        "speed_threshold": 1.5,
        "minimum_tracking_ratio": 0.5,
        "minimum_leg_rms": 0.25,
        "minimum_spine_rms": 0.10,
      },
    ),
    "spine_contact_phase": RewardTermCfg(
      func=mdp.spine_contact_phase,
      weight=0.0 if rigid_spine or not spine_reward else 0.8,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "spine_cfg": _joints(_SPINE_JOINT),
        "extension_target": 0.22,
        "compression_target": -0.32,
        "position_std": 0.16,
        "velocity_scale": 0.8,
        "pair_window": 0.24,
        "pair_hold": 0.16,
        "min_period": 0.12,
        "speed_threshold": 1.5,
      },
    ),
    "spine_clock_phase": RewardTermCfg(
      func=mdp.spine_phase_tracking,
      weight=0.0 if rigid_spine or not spine_reward else 0.4,
      params={
        "command_name": "twist",
        "asset_cfg": _joints(_SPINE_JOINT),
        "period": gait_period,
        "amplitude": 0.32,
        # Cosine phase +0.25 gives compression in fore stance and extension
        # in hind stance, matching the contact schedule above.
        "phase_offset": 0.25,
        "std": 0.18,
        "speed_threshold": 1.5,
        "actual_speed_threshold": 0.75,
      },
    ),
    "air_time": air_time,
    "stride_length": RewardTermCfg(
      func=mdp.stride_length,
      weight=0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "asset_cfg": _feet(),
        "base_stride": 0.08,
        "speed_slope": 0.02,
        "max_stride": 0.18,
        "std": 0.07,
      },
    ),
    "flight_phase": RewardTermCfg(
      func=mdp.flight_phase,
      weight=0.35,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "speed_threshold": 1.5,
        "min_air_time": 0.04,
      },
    ),
    "foot_clearance": foot_clearance,
    "foot_swing_height": foot_swing_height,
    "foot_slip": foot_slip,
    "soft_landing": soft_landing,
    "electrical_power": RewardTermCfg(
      func=mdp.electrical_power_cost,
      weight=-2.0e-3,
      params={"asset_cfg": _joints(_LEG_JOINTS + _SPINE_JOINT)},
    ),
    "joint_vel_l2": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-2.0e-5,
      params={"asset_cfg": _joints(_LEG_JOINTS)},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.02),
    "dof_pos_limits": joint_limits,
    "termination_penalty": termination,
  }

  cfg.terminations["straight_line_deviation"] = TerminationTermCfg(
    func=mdp.excessive_straight_line_deviation,
    params={
      "maximum_lateral_displacement": 0.75,
      "maximum_heading_error": 0.79,
    },
  )

  cfg.metrics.update(
    {
      "leg_mechanical_power": MetricsTermCfg(
        func=mdp.joint_abs_mechanical_power,
        params={"asset_cfg": _joints(_LEG_JOINTS)},
      ),
      "spine_mechanical_power": MetricsTermCfg(
        func=mdp.joint_abs_mechanical_power,
        params={"asset_cfg": _joints(_SPINE_JOINT)},
      ),
      "spine_position_range": MetricsTermCfg(
        func=mdp.joint_position_range,
        params={"asset_cfg": _joints(_SPINE_JOINT)},
        reduce="last",
      ),
      "leg_peak_positive_power": MetricsTermCfg(
        func=mdp.joint_positive_mechanical_power,
        params={"asset_cfg": _joints(_LEG_JOINTS)},
        reduce="max",
      ),
      "spine_positive_power": MetricsTermCfg(
        func=mdp.joint_positive_mechanical_power,
        params={"asset_cfg": _joints(_SPINE_JOINT)},
      ),
      "spine_absorbed_power": MetricsTermCfg(
        func=mdp.joint_absorbed_mechanical_power,
        params={"asset_cfg": _joints(_SPINE_JOINT)},
      ),
      "spine_power_fraction": MetricsTermCfg(
        func=mdp.spine_power_fraction,
        params={"spine_cfg": _joints(_SPINE_JOINT)},
      ),
      "leg_high_effort_fraction": MetricsTermCfg(
        func=mdp.actuator_saturation_fraction,
        params={
          "effort_limit": 7.0,
          "asset_cfg": _joints(_LEG_JOINTS),
        },
      ),
      "spine_high_effort_fraction": MetricsTermCfg(
        func=mdp.actuator_saturation_fraction,
        params={
          "effort_limit": 7.0,
          "asset_cfg": _joints(_SPINE_JOINT),
        },
      ),
      "hip_speed_limit_fraction": MetricsTermCfg(
        func=mdp.joint_speed_limit_fraction,
        params={
          "velocity_limit": HIP_ACTUATOR.velocity_limit,
          "asset_cfg": _joints(_HIP_JOINTS),
        },
      ),
      "knee_speed_limit_fraction": MetricsTermCfg(
        func=mdp.joint_speed_limit_fraction,
        params={
          "velocity_limit": KNEE_ACTUATOR.velocity_limit,
          "asset_cfg": _joints(_KNEE_JOINTS),
        },
      ),
      "spine_speed_limit_fraction": MetricsTermCfg(
        func=mdp.joint_speed_limit_fraction,
        params={
          "velocity_limit": SPINE_ACTUATOR.velocity_limit,
          "asset_cfg": _joints(_SPINE_JOINT),
        },
      ),
    }
  )

  if play:
    cfg.curriculum = {}
  else:
    cfg.curriculum = {
      "command_velocity": CurriculumTermCfg(
        func=mdp.adaptive_command_velocity,
        params={
          "command_name": "twist",
          "reward_name": "track_linear_velocity",
          "max_velocity": 5.0,
          "increment": 0.25,
          "success_threshold": 0.8,
          "min_episodes": 4096,
          "min_steps_per_stage": 1200,
          "frontier_fraction": 0.75,
          "ema_alpha": 0.2,
        },
      )
    }

  return cfg
