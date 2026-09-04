"""Go2-style straight locomotion baseline for the flexible Cheetah robot.

This first training stage deliberately has no gait clock, contact sequence,
flight or spine-motion rewards.  It learns robust forward locomotion before a
later checkpoint is fine-tuned for rotary gallop.
"""

import math

from mjlab.asset_zoo.robots import CHEETAH_ACTION_SCALE, get_cheetah_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def cheetah_go2_baseline_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create flat, straight-ahead locomotion training without prescribed gait."""
  cfg = make_velocity_env_cfg()
  # The Cheetah MJCF has more initial collision candidates than the generic
  # default. Match the proven flat-quadruped simulation limits.
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None
  cfg.scene.entities = {"robot": get_cheetah_robot_cfg()}

  site_names = ("FR", "FL", "RR", "RL")
  foot_geom_names = (
    "right_front_knee_pitch_link_collision_2",
    "left_front_knee_pitch_link_collision_2",
    "right_knee_pitch_link_collision_2",
    "left_knee_pitch_link_collision_2",
  )

  # Use the physical toe sites and only the four terminal foot collision geoms.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in site_names
      )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="geom",
        pattern=foot_geom_names,
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    ),
  )

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = CHEETAH_ACTION_SCALE

  # Keep the desired heading at world zero. The heading controller supplies a
  # corrective yaw command if the robot begins to turn left or right.
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.resampling_time_range = (1.0e9, 1.0e9)
  command.rel_standing_envs = 0.0
  command.rel_heading_envs = 1.0
  command.rel_forward_envs = 0.0
  command.heading_command = True
  command.heading_control_stiffness = 1.0
  command.ranges.lin_vel_x = (0.5, 0.5)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (-1.0, 1.0)
  command.ranges.heading = (0.0, 0.0)

  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)

  cfg.viewer.body_name = "body_front_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Go2-like neutral-pose regularization.  The spine remains controllable, but
  # receives a narrow tolerance so it does not learn useless pumping yet.
  cfg.rewards["pose"].params["std_standing"] = {
    r".*hip_roll_joint": 0.05,
    r".*hip_pitch_joint": 0.10,
    r".*knee_pitch_joint": 0.15,
    "body_pitch_joint": 0.03,
  }
  moving_pose_std = {
    r".*hip_roll_joint": 0.15,
    r".*hip_pitch_joint": 0.35,
    r".*knee_pitch_joint": 0.50,
    "body_pitch_joint": 0.05,
  }
  cfg.rewards["pose"].params["std_walking"] = moving_pose_std
  cfg.rewards["pose"].params["std_running"] = moving_pose_std
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("body_front_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("body_front_link",)
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = site_names

  # Flat ground, as in the classical quadruped baseline experiment.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )
  cfg.curriculum.pop("terrain_levels", None)

  # Curriculum steps are simulator steps: one PPO iteration is 24 steps.
  # Target speed: 0.5 -> 1.5 -> 3.0 -> 5.0 m/s.
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {
          "step": 0,
          "lin_vel_x": (0.5, 0.5),
          "lin_vel_y": (0.0, 0.0),
          "ang_vel_z": None,
        },
        {
          "step": 1_000 * 24,
          "lin_vel_x": (1.5, 1.5),
          "lin_vel_y": None,
          "ang_vel_z": None,
        },
        {
          "step": 2_500 * 24,
          "lin_vel_x": (3.0, 3.0),
          "lin_vel_y": None,
          "ang_vel_z": None,
        },
        {
          "step": 4_500 * 24,
          "lin_vel_x": (5.0, 5.0),
          "lin_vel_y": None,
          "ang_vel_z": None,
        },
      ],
    },
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    for sensor in cfg.scene.sensors or ():
      if sensor.name == "foot_height_scan":
        assert isinstance(sensor, TerrainHeightSensorCfg)
        sensor.debug_vis = False
    # Evaluate the current stage-1 checkpoint at the speed on which it was
    # trained. Increase this together with the curriculum stage later.
    command.ranges.lin_vel_x = (0.5, 0.5)

  return cfg


def _cheetah_go2_baseline_continuation_env_cfg(
  speed: float,
  play: bool,
) -> ManagerBasedRlEnvCfg:
  """Create a fixed-speed continuation without resetting to stage-one speed."""
  cfg = cheetah_go2_baseline_env_cfg(play=play)
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.ranges.lin_vel_x = (speed, speed)

  if not play:
    # Resume creates a new environment and resets its simulator step counter.
    # Use a small speed increase so the velocity reward remains informative.
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (speed, speed),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": None,
      }
    ]

  return cfg


def cheetah_go2_baseline_stage2_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Continue the learned baseline at a fixed forward speed of 0.75 m/s."""
  return _cheetah_go2_baseline_continuation_env_cfg(speed=0.75, play=play)


def cheetah_go2_baseline_stage3_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Continue the learned baseline at a fixed forward speed of 1.0 m/s."""
  return _cheetah_go2_baseline_continuation_env_cfg(speed=1.0, play=play)


def cheetah_go2_baseline_stage4_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Continue the learned baseline at a fixed forward speed of 1.25 m/s."""
  return _cheetah_go2_baseline_continuation_env_cfg(speed=1.25, play=play)


def cheetah_go2_baseline_stage5_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Continue the learned baseline at a fixed forward speed of 1.375 m/s."""
  return _cheetah_go2_baseline_continuation_env_cfg(speed=1.375, play=play)
