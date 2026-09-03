"""Reward configuration for the flexible-spine Cheetah gallop task."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

ROTARY_STANCE_INTERVALS = (
  (0.14, 0.28),  # FL
  (0.54, 0.68),  # FR
  (0.68, 0.82),  # RL
  (0.28, 0.42),  # RR
)
FLIGHT_WINDOWS = ((0.0, 0.14), (0.42, 0.54), (0.82, 1.0))


def _feet(site_names: tuple[str, ...]) -> SceneEntityCfg:
  return SceneEntityCfg("robot", site_names=site_names, preserve_order=True)


def configure_gallop_rewards(
  cfg: ManagerBasedRlEnvCfg,
  *,
  site_names: tuple[str, ...],
  gait_period: float,
  feet_sensor_name: str,
) -> None:
  """Configure locomotion, double-flight and flexible-spine rewards."""
  pose = cfg.rewards["pose"]
  pose.params["std_standing"] = {
    r".*_hip_roll_joint": 0.05,
    r".*_hip_pitch_joint": 0.05,
    r".*_knee_pitch_joint": 0.20,
    r"body_pitch_joint": 0.05,
  }
  pose.params["std_walking"] = {
    r".*_hip_roll_joint": 0.20,
    r".*_hip_pitch_joint": 0.45,
    r".*_knee_pitch_joint": 0.70,
    r"body_pitch_joint": 0.35,
  }
  pose.params["std_running"] = {
    r".*_hip_roll_joint": 0.30,
    r".*_hip_pitch_joint": 0.70,
    r".*_knee_pitch_joint": 1.00,
    r"body_pitch_joint": 1.00,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("body_front_link",)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (
    "body_front_link",
  )
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"] = _feet(site_names)

  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_linear_velocity"].params["std"] = 3.5
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["upright"].weight = 0.5
  pose.weight = 0.2
  cfg.rewards["action_rate_l2"].weight = -0.01
  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.25
  cfg.rewards["air_time"].params["threshold_max"] = 0.35

  cfg.rewards.update(
    {
      "forward_velocity_progress": RewardTermCfg(
        func=mdp.forward_velocity_progress,
        weight=8.0,
        params={"command_name": "twist"},
      ),
      "planar_drift_l2": RewardTermCfg(func=mdp.planar_drift_l2, weight=-1.0),
      "bilateral_foot_amplitude": RewardTermCfg(
        func=mdp.bilateral_foot_amplitude,
        weight=-2.0,
        params={
          "asset_cfg": _feet(site_names),
          "period": gait_period,
          "amplitude_std": 0.02,
        },
      ),
      "bilateral_actuator_power": RewardTermCfg(
        func=mdp.bilateral_actuator_power,
        weight=-1.0,
        params={
          "left_asset_cfg": SceneEntityCfg(
            "robot", joint_names=(r"^left_.*_joint$",)
          ),
          "right_asset_cfg": SceneEntityCfg(
            "robot", joint_names=(r"^right_.*_joint$",)
          ),
          "period": gait_period,
        },
      ),
      "gallop_sequence": RewardTermCfg(
        func=mdp.footfall_sequence,
        weight=3.0,
        params={
          "sensor_name": feet_sensor_name,
          "sequence": (0, 3, 1, 2),
          "command_name": "twist",
          "command_threshold": 0.3,
          "wrong_contact_penalty": 1.0,
          "actual_speed_threshold": 0.75,
        },
      ),
      "clock_gallop": RewardTermCfg(
        func=mdp.clock_gait,
        weight=0.0,
        params={
          "sensor_name": feet_sensor_name,
          "command_name": "twist",
          "period": gait_period,
          "offsets": (0.0, 0.5, 0.25, 0.75),
          "stance_ratio": 0.20,
          "command_threshold": 0.15,
        },
      ),
      "feline_gallop_contacts": RewardTermCfg(
        func=mdp.feline_gallop_contacts,
        weight=6.0,
        params={
          "sensor_name": feet_sensor_name,
          "command_name": "twist",
          "period": gait_period,
          "stance_intervals": ROTARY_STANCE_INTERVALS,
          "command_threshold": 1.0,
          "correct_swing_reward": 0.25,
          "false_contact_penalty": 2.0,
          "missed_contact_penalty": 1.0,
          "actual_speed_threshold": 0.75,
        },
      ),
      "stride_length": RewardTermCfg(
        func=mdp.stride_length,
        weight=3.0,
        params={
          "sensor_name": feet_sensor_name,
          "command_name": "twist",
          "asset_cfg": _feet(site_names),
          "base_stride": 0.10,
          "speed_slope": 0.035,
          "max_stride": 0.24,
          "std": 0.04,
        },
      ),
      "flight_phase": RewardTermCfg(
        func=mdp.flight_phase,
        weight=4.0,
        params={
          "sensor_name": feet_sensor_name,
          "command_name": "twist",
          "speed_threshold": 2.0,
          "min_air_time": 0.025,
          "phase_period": gait_period,
          "phase_windows": FLIGHT_WINDOWS,
          "actual_speed_threshold": 0.75,
        },
      ),
      "jumping_in_place": RewardTermCfg(
        func=mdp.jumping_in_place,
        weight=-4.0,
        params={
          "sensor_name": feet_sensor_name,
          "asset_cfg": SceneEntityCfg("robot"),
          "minimum_forward_speed": 0.75,
          "min_air_time": 0.025,
        },
      ),
      "extended_flight_posture": _flight_posture(
        site_names,
        feet_sensor_name,
        gait_period,
        weight=3.5,
        front_target_x=0.24,
        hind_target_x=-0.20,
        phase_windows=((0.0, 0.14), (0.82, 1.0)),
        metric_prefix="extended_flight",
      ),
      "collected_flight_posture": _flight_posture(
        site_names,
        feet_sensor_name,
        gait_period,
        weight=4.0,
        front_target_x=0.10,
        hind_target_x=-0.04,
        phase_windows=((0.42, 0.54),),
        metric_prefix="collected_flight",
      ),
      "hind_propulsion": RewardTermCfg(
        func=mdp.hind_propulsion,
        weight=2.0,
        params={
          "sensor_name": feet_sensor_name,
          "command_name": "twist",
          "asset_cfg": SceneEntityCfg("robot"),
          "period": gait_period,
          "push_windows": ((0.28, 0.42), (0.68, 0.82)),
          "target_acceleration": 8.0,
          "speed_threshold": 2.0,
        },
      ),
      "spine_flexion": RewardTermCfg(
        func=mdp.spine_flexion,
        weight=0.25,
        params={
          "command_name": "twist",
          "asset_cfg": SceneEntityCfg(
            "robot", joint_names=("body_pitch_joint",)
          ),
          "window_s": 0.5,
          "speed_threshold": 1.8,
          "actual_speed_threshold": 0.5,
          "target_range": 0.8,
          "range_std": 0.2,
          "center_std": 0.12,
        },
      ),
      "spine_phase_tracking": RewardTermCfg(
        func=mdp.spine_phase_tracking,
        weight=0.5,
        params={
          "command_name": "twist",
          "asset_cfg": SceneEntityCfg(
            "robot", joint_names=("body_pitch_joint",)
          ),
          "period": gait_period,
          "sensor_name": feet_sensor_name,
          "stance_intervals": ROTARY_STANCE_INTERVALS,
          "minimum_contact_agreement": 0.75,
          "amplitude": -0.4,
          "phase_offset": 0.0,
          "std": 0.18,
          "speed_threshold": 1.8,
          "actual_speed_threshold": 0.5,
        },
      ),
      "stand_still": RewardTermCfg(
        func=mdp.stand_still,
        weight=-0.5,
        params={
          "command_name": "twist",
          "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
          "command_threshold": 0.1,
        },
      ),
      "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    }
  )

  for name in ("foot_clearance", "foot_swing_height"):
    cfg.rewards[name].params["target_height"] = 0.092
    cfg.rewards[name].weight = -0.5


def _flight_posture(
  site_names: tuple[str, ...],
  sensor_name: str,
  period: float,
  *,
  weight: float,
  front_target_x: float,
  hind_target_x: float,
  phase_windows: tuple[tuple[float, float], ...],
  metric_prefix: str,
) -> RewardTermCfg:
  return RewardTermCfg(
    func=mdp.extended_flight_posture,
    weight=weight,
    params={
      "sensor_name": sensor_name,
      "command_name": "twist",
      "asset_cfg": _feet(site_names),
      "speed_threshold": 2.0,
      "min_air_time": 0.025,
      "front_target_x": front_target_x,
      "hind_target_x": hind_target_x,
      "position_std": 0.08,
      "symmetry_std": 0.05,
      "phase_period": period,
      "phase_windows": phase_windows,
      "metric_prefix": metric_prefix,
      "actual_speed_threshold": 0.75,
    },
  )


def configure_collision_rewards(
  cfg: ManagerBasedRlEnvCfg,
  *,
  self_collision_sensor: str,
  shank_sensor: str,
  trunk_sensor: str,
) -> None:
  """Add rough-terrain collision penalties."""
  for reward_name, sensor_name in (
    ("self_collisions", self_collision_sensor),
    ("shank_collision", shank_sensor),
    ("trunk_head_collision", trunk_sensor),
  ):
    cfg.rewards[reward_name] = RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": sensor_name},
    )
