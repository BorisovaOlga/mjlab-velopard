"""Tracking environment for an articulated, flexible-spine Cheetah."""

from pathlib import Path

from mjlab.asset_zoo.robots.cheetah import get_reborn_cheetah_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.actions import ReferenceJointPositionActionCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

_CHEETAH_JOINTS = (
  "body_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_pitch_joint",
  "left_knee_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_pitch_joint",
  "right_knee_pitch_joint",
  "left_front_hip_roll_joint",
  "left_front_hip_pitch_joint",
  "left_front_knee_pitch_joint",
  "right_front_hip_roll_joint",
  "right_front_hip_pitch_joint",
  "right_front_knee_pitch_joint",
)

# The order is deliberate: the first body anchors the global reference and the
# second body is the actuated spine segment.  The four knee links keep the
# reference grounded in the locomotion that matters for imitation.
_TRACKED_BODIES = (
  "body_front_link",
  "body_pitch_link",
  "left_front_knee_pitch_link",
  "right_front_knee_pitch_link",
  "left_knee_pitch_link",
  "right_knee_pitch_link",
)

_DEFAULT_MOTION_FILE = Path(
  "docs/reborn_spined_cheetah_train/dog_mocap_bvh/cheetah_motion.npz"
)

_SPINE_JOINTS = ("body_pitch_joint",)
_LEG_JOINTS = tuple(name for name in _CHEETAH_JOINTS if name not in _SPINE_JOINTS)


def reborn_cheetah_flat_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a flat-ground imitation environment for the flexible Cheetah."""
  cfg = make_tracking_env_cfg()
  cfg.sim.nconmax = None
  cfg.sim.njmax = 300
  cfg.scene.entities = {
    "robot": get_reborn_cheetah_robot_cfg(),
  }

  action = ReferenceJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    command_name="motion",
  )
  cfg.actions["joint_pos"] = action
  action.actuator_names = (".*",)
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
    # The BVH retarget contains extension up to roughly +0.7 rad.  Tracking
    # must not clip that target before the policy can learn it.
    r"body_pitch_joint": (-0.90, 0.90),
  }

  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.anchor_body_name = "body_front_link"
  motion.body_names = _TRACKED_BODIES
  motion.motion_file = (
    str(_DEFAULT_MOTION_FILE) if _DEFAULT_MOTION_FILE.exists() else ""
  )
  motion.sampling_mode = "uniform"
  motion.pose_range = {}
  motion.velocity_range = {}
  motion.joint_position_range = (-0.02, 0.02)
  motion.debug_vis = False

  # The reference trajectory is the source of forward motion.  Disturbance and
  # parameter randomization are enabled only after the first imitation policy
  # tracks the clip reliably.
  cfg.events.pop("push_robot", None)
  cfg.events.pop("base_com", None)
  cfg.events.pop("encoder_bias", None)
  cfg.events.pop("foot_friction", None)
  cfg.rewards.pop("self_collisions", None)
  cfg.rewards["motion_global_root_pos"].weight = 1.0
  cfg.rewards["motion_global_root_ori"].weight = 1.0
  cfg.rewards["motion_joint_pos"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp,
    weight=0.75,
    params={"command_name": "motion", "std": 0.35},
  )
  cfg.rewards["motion_joint_vel"] = RewardTermCfg(
    func=mdp.motion_joint_velocity_error_exp,
    weight=0.10,
    # Use a per-joint MSE and a wider scale so the high-speed reference does
    # not underflow to zero before the policy learns the reversal phase.
    params={"command_name": "motion", "std": 4.0},
  )
  cfg.rewards["motion_body_pos"].weight = 2.0
  cfg.rewards["motion_body_ori"].weight = 1.5
  cfg.rewards["motion_body_lin_vel"].weight = 0.5
  cfg.rewards["motion_body_ang_vel"].weight = 0.25
  cfg.rewards["action_rate_l2"].weight = -0.02
  cfg.rewards["joint_limit"].weight = -2.0
  all_joints = SceneEntityCfg("robot", joint_names=_CHEETAH_JOINTS)
  cfg.rewards["joint_torque_l2"] = RewardTermCfg(
    func=mdp.joint_torque_l2,
    weight=-5.0e-4,
    params={"asset_cfg": all_joints},
  )
  cfg.rewards["mechanical_power_abs"] = RewardTermCfg(
    func=mdp.joint_abs_mechanical_power,
    weight=-5.0e-4,
    params={"asset_cfg": all_joints},
  )

  leg_joints = SceneEntityCfg("robot", joint_names=_LEG_JOINTS)
  spine_joint = SceneEntityCfg("robot", joint_names=_SPINE_JOINTS)
  cfg.metrics.update(
    {
      "joint_abs_torque": MetricsTermCfg(
        func=mdp.joint_abs_torque, params={"asset_cfg": all_joints}
      ),
      "joint_abs_velocity": MetricsTermCfg(
        func=mdp.joint_abs_velocity, params={"asset_cfg": all_joints}
      ),
      "leg_mechanical_power": MetricsTermCfg(
        func=mdp.joint_abs_mechanical_power, params={"asset_cfg": leg_joints}
      ),
      "spine_mechanical_power": MetricsTermCfg(
        func=mdp.joint_abs_mechanical_power, params={"asset_cfg": spine_joint}
      ),
      "leg_positive_power": MetricsTermCfg(
        func=mdp.joint_positive_mechanical_power, params={"asset_cfg": leg_joints}
      ),
      "spine_positive_power": MetricsTermCfg(
        func=mdp.joint_positive_mechanical_power, params={"asset_cfg": spine_joint}
      ),
      "spine_absorbed_power": MetricsTermCfg(
        func=mdp.joint_absorbed_mechanical_power, params={"asset_cfg": spine_joint}
      ),
      "leg_mechanical_energy": MetricsTermCfg(
        func=mdp.joint_abs_mechanical_energy,
        params={"asset_cfg": leg_joints},
        reduce="sum",
      ),
      "spine_mechanical_energy": MetricsTermCfg(
        func=mdp.joint_abs_mechanical_energy,
        params={"asset_cfg": spine_joint},
        reduce="sum",
      ),
      "spine_positive_energy": MetricsTermCfg(
        func=mdp.joint_positive_mechanical_energy,
        params={"asset_cfg": spine_joint},
        reduce="sum",
      ),
      "spine_absorbed_energy": MetricsTermCfg(
        func=mdp.joint_absorbed_mechanical_energy,
        params={"asset_cfg": spine_joint},
        reduce="sum",
      ),
      "spine_power_fraction": MetricsTermCfg(
        func=mdp.spine_power_fraction,
        params={"spine_cfg": spine_joint, "asset_cfg": all_joints},
      ),
    }
  )
  cfg.terminations["anchor_pos"].params["threshold"] = 0.35
  cfg.terminations["anchor_ori"].params["threshold"] = 1.0
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_front_knee_pitch_link",
    "right_front_knee_pitch_link",
    "left_knee_pitch_link",
    "right_knee_pitch_link",
  )

  cfg.viewer.body_name = "body_front_link"
  cfg.viewer.distance = 1.4
  cfg.viewer.elevation = -10.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    motion.sampling_mode = "start"
    motion.joint_position_range = (0.0, 0.0)

  return cfg


__all__ = [
  "_CHEETAH_JOINTS",
  "_TRACKED_BODIES",
  "reborn_cheetah_flat_tracking_env_cfg",
]
