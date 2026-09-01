"""Go2-style velocity task for the fixed-spine Cheetah robot."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .env_cfgs import cheetah_flat_env_cfg


def cheetah_classic_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a fixed-spine, diagonal-trot task modelled on Unitree Go2."""
  cfg = cheetah_flat_env_cfg(play=play)

  # Go2 uses a 0.6 s clock and diagonal pairs. Cheetah sensor order is
  # FL, FR, RL, RR, hence FL+RR and FR+RL share their respective phases.
  gait_period = 0.6
  for group in cfg.observations.values():
    group.terms["gait_phase"].params["period"] = gait_period

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # Use exactly the same command as the flexible-spine gallop experiment:
  # fixed world +X at 5 m/s, with no lateral or yaw command.
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.rel_world_envs = 1.0
  twist_cmd.heading_command = False
  twist_cmd.ranges.lin_vel_x = (5.0, 5.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None

  # Remove all feline/high-speed shaping. The remaining reward set follows the
  # generic Go2 velocity task and therefore represents conventional robot trot.
  classic_reward_names = {
    "track_linear_velocity",
    "forward_velocity_progress",
    "track_angular_velocity",
    "planar_drift_l2",
    "upright",
    "pose",
    "dof_pos_limits",
    "action_rate_l2",
    "foot_clearance",
    "foot_slip",
    "soft_landing",
    "stand_still",
    "joint_acc_l2",
    "termination_penalty",
  }
  cfg.rewards = {
    name: reward for name, reward in cfg.rewards.items() if name in classic_reward_names
  }
  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_linear_velocity"].params["std"] = 2.0
  cfg.rewards["forward_velocity_progress"].weight = 8.0
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["planar_drift_l2"].weight = -1.0
  cfg.rewards["upright"].weight = 1.0
  cfg.rewards["pose"].weight = 1.0
  cfg.rewards["dof_pos_limits"].weight = -10.0
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["foot_clearance"].params["command_threshold"] = 0.1
  cfg.rewards["foot_slip"].weight = -0.25
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.1
  cfg.rewards["soft_landing"].weight = -1.0e-3
  cfg.rewards["soft_landing"].params["command_threshold"] = 0.1
  cfg.rewards["stand_still"].weight = -1.0
  cfg.rewards["termination_penalty"].weight = -200.0
  cfg.rewards["foot_gait"] = RewardTermCfg(
    func=mdp.feet_gait,
    weight=0.5,
    params={
      "period": gait_period,
      "offset": (0.5, 0.0, 0.0, 0.5),
      "threshold": 0.56,
      "command_threshold": 0.1,
      "command_name": "twist",
      "sensor_name": "feet_ground_contact",
    },
  )

  # No velocity curriculum: every training and play environment receives the
  # same 5 m/s command, so CoT is evaluated under the same target condition.
  cfg.curriculum.pop("command_vel", None)

  # Keep the posture reward scoped explicitly to the twelve leg joints.
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(".*",)
  )
  return cfg
