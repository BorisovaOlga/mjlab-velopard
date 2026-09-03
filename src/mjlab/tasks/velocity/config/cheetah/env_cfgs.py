"""Robot, sensor, command and environment configuration for Cheetah velocity."""

import math
from typing import Literal

from mjlab.asset_zoo.robots.cheetah import (
  CHEETAH_ACTION_SCALE,
  get_cheetah_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from .gallop_rewards_cfg import configure_collision_rewards, configure_gallop_rewards

TerrainType = Literal["rough", "obstacles"]
DEFAULT_GAIT_PERIOD = 0.6
# Keep the phase observation identical to the checkpoint used to start fine-tuning.
# Changing it abruptly changes the meaning of two policy inputs (sin/cos phase).
FLIGHT_FINETUNE_PERIOD = DEFAULT_GAIT_PERIOD
FLIGHT_FINETUNE_STANCES = (
  (0.12, 0.25),  # FL
  (0.58, 0.71),  # FR
  (0.71, 0.84),  # RL
  (0.25, 0.40),  # RR
)
FLIGHT_FINETUNE_WINDOWS = ((0.0, 0.12), (0.40, 0.58), (0.84, 1.0))

CHEETAH_JOINT_NAMES = (
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
  "body_pitch_joint",
)

"""Create Cheetah rough terrain velocity configuration."""
def cheetah_rough_env_cfg(
  play: bool = False,           # Function parameter to determine if the environment is in play mode or not. play=False means the environment is for learning
  gait_period: float = DEFAULT_GAIT_PERIOD,
) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg() # Create a base configuration for the velocity environment using the make_velocity_env_cfg function.

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_cheetah_robot_cfg()}       # Add the cheetah robot configuration to the scene. 

  cfg.metrics["mechanical_cost_of_transport"] = MetricsTermCfg(
    func=mdp.mechanical_cost_of_transport,
    params={"mass": 5.424414725317205},
  )
  for joint_name in CHEETAH_JOINT_NAMES:
    joint_cfg = SceneEntityCfg("robot", joint_names=(joint_name,))
    cfg.metrics[f"actuator_torque_abs/{joint_name}"] = MetricsTermCfg(
      func=mdp.joint_abs_torque,
      params={"asset_cfg": joint_cfg},
    )
    cfg.metrics[f"actuator_velocity_abs/{joint_name}"] = MetricsTermCfg(
      func=mdp.joint_abs_velocity,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(joint_name,))},
    )

  """A shared clock lets the policy coordinate all feet and the spine instead of
  inferring gait phase from contacts after they have already happened."""
  for group in cfg.observations.values():
    group.terms["gait_phase"] = ObservationTermCfg(
      func=mdp.gait_phase, params={"period": gait_period}
    )

  # Set raycast sensor frame to the cheetah root body (body_front_link).
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "body_front_link"

  # The Cheetah MJCF now provides per-foot sites (FR, FL, RR, RL). Use those
  # site names for foot sensors and rewards so configs mirror Go1 style.
  site_names = ("FL", "FR", "RL", "RR")
  # One support sphere per foot, in the same FL, FR, RL, RR order as sites.
  # Broad leg regexes also matched hip/shank geoms and made air-time unreliable.
  geom_names = (
    "left_front_knee_pitch_link_collision_2",
    "right_front_knee_pitch_link_collision_2",
    "left_knee_pitch_link_collision_2",
    "right_knee_pitch_link_collision_2",
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom", pattern=geom_names, entity="robot", preserve_order=True
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="body_front_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="body_front_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # Thigh/link collisions: anchor front vs hind to avoid overlap.
  thigh_geom_names = (
    r"^left_front_.*hip.*_collision\d*",
    r"^right_front_.*hip.*_collision\d*",
    r"^left_(?!front).*hip.*_collision\d*",
    r"^right_(?!front).*hip.*_collision\d*",
  )
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=thigh_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # Calf/shank collisions: anchor front vs hind to avoid overlap.
  calf_geom_names = (
    r"^left_front_.*knee.*_collision\d*",
    r"^right_front_.*knee.*_collision\d*",
    r"^left_(?!front).*knee.*_collision\d*",
    r"^right_(?!front).*knee.*_collision\d*",
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=calf_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  trunk_head_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=("spine_collision", "body_front_link_collision_1"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    thigh_ground_cfg,
    shank_ground_cfg,
    trunk_head_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = CHEETAH_ACTION_SCALE

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # Start every environment with the first fixed curriculum stage.  The target
  # reaches 5 m/s later; play mode below always evaluates at exactly 5 m/s.
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  # Keep the non-random command fixed along world +X.  At reset the robot faces
  # +X, so this is straight ahead; after yaw drift it creates a restoring
  # lateral body-frame command instead of accepting the new heading as forward.
  twist_cmd.rel_world_envs = 1.0
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.ranges.lin_vel_x = (1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

  # Previous randomized command configuration.  Uncomment this block and the
  # command curriculum below to restore sampling of different target speeds.
  # twist_cmd.rel_standing_envs = 0.1
  # twist_cmd.rel_forward_envs = 1.0
  # twist_cmd.ranges.lin_vel_x = (0.0, 4.0)
  # twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  # twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  cfg.viewer.body_name = "body_front_link"
  cfg.viewer.distance = 1.2
  cfg.viewer.elevation = -10.0

  # Replace the base foot_friction with per-axis friction events for condim 6.
  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "axes": [0],
      # Lighter robot - slightly lower friction range by default.
      "ranges": (0.2, 1.2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      # Spin friction (lateral) typically very small for light paws.
      "ranges": (1e-5, 2e-3),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-6, 5e-4),
      "shared_random": True,
    },
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("body_front_link",)

  configure_gallop_rewards(
    cfg,
    site_names=site_names,
    gait_period=gait_period,
    feet_sensor_name=feet_ground_cfg.name,
  )

  # A policy initialized from scratch rarely discovers useful forward motion
  # when every environment immediately requests 5 m/s.  Keep each stage fixed
  # and straight, but raise the shared target as locomotion becomes established.
  # The runner collects 24 environment steps per learning iteration.
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (1.0, 1.0)},
        {"step": 500 * 24, "lin_vel_x": (2.0, 2.0)},
        {"step": 1500 * 24, "lin_vel_x": (3.5, 3.5)},
        {"step": 3000 * 24, "lin_vel_x": (5.0, 5.0)},
      ],
    },
  )
  configure_collision_rewards(
    cfg,
    self_collision_sensor=self_collision_cfg.name,
    shank_sensor=shank_ground_cfg.name,
    trunk_sensor=trunk_head_ground_cfg.name,
  )

  # On rough terrain the quadruped tilts significantly; don't terminate on
  # orientation alone. Let out_of_terrain_bounds handle resets.
  cfg.terminations.pop("fell_over", None)

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def cheetah_flat_env_cfg(
  play: bool = False,
  gait_period: float = DEFAULT_GAIT_PERIOD,
) -> ManagerBasedRlEnvCfg:
  """Create Cheetah flat terrain velocity configuration."""
  cfg = cheetah_rough_env_cfg(play=play, gait_period=gait_period)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensors and collision sensors not needed on flat.
  remove_sensors = {
    "terrain_scan",
    "self_collision",
    "thigh_ground_touch",
    "shank_ground_touch",
    "trunk_ground_touch",
  }
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  # Remove granular collision rewards (not useful on flat ground).
  for key in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  # On flat terrain fell_over is sufficient; thigh contact implies fallen.
  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )
  cfg.terminations["body_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": 0.11},
  )
  cfg.rewards["termination_penalty"] = RewardTermCfg(
    func=envs_mdp.is_terminated,
    weight=-50.0,
  )

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (5.0, 5.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  return cfg


def cheetah_flat_flight_finetune_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create staged straight-running and flight training; play evaluates at 5 m/s."""
  cfg = cheetah_flat_env_cfg(play=play, gait_period=FLIGHT_FINETUNE_PERIOD)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  if play:
    twist_cmd.ranges.lin_vel_x = (5.0, 5.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  # Training keeps the base 1→2→3.5→5 m/s command curriculum. Starting a
  # random policy directly at 5 m/s previously encouraged falling and sliding.

  if not play:
    # Let the policy adapt to the corrected direction before reintroducing
    # external disturbances in a later robustness-training stage.
    cfg.events.pop("push_robot", None)
    cfg.rewards["termination_penalty"].weight = -200.0
    # Stage 1 (0-1499 iterations): learn stable straight locomotion while the
    # command curriculum raises speed from 1 to 2 m/s.
    cfg.rewards["planar_drift_l2"].weight = -4.0
    cfg.rewards["world_straight_line_l2"] = RewardTermCfg(
      func=mdp.world_straight_line_l2,
      weight=-0.5,
      params={"lateral_position_scale": 2.0, "heading_scale": 3.0},
    )
    cfg.rewards["hip_roll_deviation_l2"] = RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=0.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r".*_hip_roll_joint",)
        )
      },
    )
    cfg.rewards["hip_roll_velocity_l2"] = RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-0.002,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r".*_hip_roll_joint",)
        )
      },
    )
    cfg.rewards["bilateral_foot_amplitude"].weight = -1.0
    cfg.rewards["bilateral_actuator_power"].weight = -0.5
    cfg.rewards["front_pair_power_balance"] = RewardTermCfg(
      func=mdp.bilateral_actuator_power,
      weight=0.0,
      params={
        "left_asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r"^left_front_.*_joint$",)
        ),
        "right_asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r"^right_front_.*_joint$",)
        ),
        "period": FLIGHT_FINETUNE_PERIOD,
        "metric_name": "front_pair_power_difference",
      },
    )
    cfg.rewards["hind_pair_power_balance"] = RewardTermCfg(
      func=mdp.bilateral_actuator_power,
      weight=0.0,
      params={
        "left_asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r"^left_(?!front_).*_joint$",)
        ),
        "right_asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r"^right_(?!front_).*_joint$",)
        ),
        "period": FLIGHT_FINETUNE_PERIOD,
        "metric_name": "hind_pair_power_difference",
      },
    )

    # Flight-related terms start disabled. Curriculum enables them only after
    # straight, symmetric running and then reduced contact overlap are learned.
    cfg.rewards["flight_phase"].weight = 0.0
    cfg.rewards["flight_phase"].params.update(
      {
        "min_air_time": 0.04,
        "phase_windows": FLIGHT_FINETUNE_WINDOWS,
      }
    )
    cfg.rewards["feline_gallop_contacts"].params["stance_intervals"] = (
      FLIGHT_FINETUNE_STANCES
    )
    cfg.rewards["extended_flight_posture"].params.update(
      {
        "min_air_time": 0.04,
        "phase_windows": ((0.0, 0.12), (0.84, 1.0)),
      }
    )
    cfg.rewards["collected_flight_posture"].params.update(
      {"min_air_time": 0.04, "phase_windows": ((0.40, 0.58),)}
    )
    cfg.rewards["hind_propulsion"].params["push_windows"] = (
      (0.25, 0.40),
      (0.71, 0.84),
    )
    cfg.rewards["spine_phase_tracking"].params["stance_intervals"] = (
      FLIGHT_FINETUNE_STANCES
    )
    cfg.rewards["action_rate_l2"].weight = -0.015
    cfg.rewards["flight_contact_violation"] = RewardTermCfg(
      func=mdp.flight_contact_violation,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "period": FLIGHT_FINETUNE_PERIOD,
        "phase_windows": FLIGHT_FINETUNE_WINDOWS,
        "actual_speed_threshold": 1.0,
      },
    )
    cfg.rewards["excess_foot_contacts"] = RewardTermCfg(
      func=mdp.excess_foot_contacts,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "maximum_contacts": 1,
        "actual_speed_threshold": 1.0,
      },
    )
    cfg.rewards["sustained_flight"] = RewardTermCfg(
      func=mdp.sustained_flight,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "target_duration": 0.06,
        "actual_speed_threshold": 1.0,
      },
    )
    cfg.rewards["early_gait_cycle"] = RewardTermCfg(
      func=mdp.early_gait_cycle,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "minimum_period": 0.30,
        "foot_index": 0,
      },
    )

    steps_per_iteration = 24
    stage_2 = 1500 * steps_per_iteration
    stage_3 = 3000 * steps_per_iteration
    stage_4 = 4000 * steps_per_iteration

    def add_reward_curriculum(
      reward_name: str, stages: list[dict[str, float | int]]
    ) -> None:
      cfg.curriculum[f"reward_{reward_name}"] = CurriculumTermCfg(
        func=envs_mdp.reward_curriculum,
        params={"reward_name": reward_name, "stages": stages},
      )

    # Stage 2 (1500+): discourage multi-foot support and implausibly short cycles.
    add_reward_curriculum(
      "excess_foot_contacts",
      [
        {"step": 0, "weight": 0.0},
        {"step": stage_2, "weight": -1.0},
        {"step": stage_3, "weight": -2.0},
        {"step": stage_4, "weight": -3.0},
      ],
    )
    add_reward_curriculum(
      "early_gait_cycle",
      [
        {"step": 0, "weight": 0.0},
        {"step": stage_2, "weight": -1.5},
        {"step": stage_3, "weight": -3.0},
        {"step": stage_4, "weight": -4.0},
      ],
    )
    # Stage 3 (3000+): first ask for short genuine all-feet-off-ground intervals.
    add_reward_curriculum(
      "flight_phase",
      [
        {"step": 0, "weight": 0.0},
        {"step": stage_3, "weight": 2.0},
        {"step": stage_4, "weight": 4.0},
      ],
    )
    add_reward_curriculum(
      "flight_contact_violation",
      [
        {"step": 0, "weight": 0.0},
        {"step": stage_3, "weight": -2.0},
        {"step": stage_4, "weight": -5.0},
      ],
    )
    # Stage 4 (4000+): only after flight appears, increase its duration objective.
    add_reward_curriculum(
      "sustained_flight",
      [
        {"step": 0, "weight": 0.0},
        {"step": stage_3, "weight": 1.0},
        {"step": stage_4, "weight": 3.0},
      ],
    )
  return cfg
