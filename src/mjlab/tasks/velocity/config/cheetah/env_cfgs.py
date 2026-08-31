"""Cheetah velocity environment configurations."""

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

TerrainType = Literal["rough", "obstacles"]

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


def cheetah_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Cheetah rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_cheetah_robot_cfg()}

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

  # A shared clock lets the policy coordinate all feet and the spine instead of
  # inferring gait phase from contacts after they have already happened.
  gait_period = 0.4
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
  # Match collision geoms while anchoring front vs hind to avoid overlap.
  geom_names = (
    r"^left_front_.*_collision",
    r"^right_front_.*_collision",
    r"^left_(?!front).*_collision",
    r"^right_(?!front).*_collision",
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
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
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
  # Train every environment with one fixed body-frame command: 5 m/s forward.
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  # Keep the non-random command fixed along world +X.  At reset the robot faces
  # +X, so this is straight ahead; after yaw drift it creates a restoring
  # lateral body-frame command instead of accepting the new heading as forward.
  twist_cmd.rel_world_envs = 1.0
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.ranges.lin_vel_x = (5.0, 5.0)
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

  # Reward regexes: match cheetah joint naming like 'left_front_hip_pitch_joint',
  # 'left_knee_pitch_joint', etc.
  # Per-joint stds for posture reward tuned for the Cheetah articulation.
  cfg.rewards["pose"].params["std_standing"] = {
    r".*_hip_roll_joint": 0.05,
    r".*_hip_pitch_joint": 0.05,
    r".*_knee_pitch_joint": 0.20,
    r"body_pitch_joint": 0.05,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_hip_roll_joint": 0.20,
    r".*_hip_pitch_joint": 0.45,
    r".*_knee_pitch_joint": 0.70,
    r"body_pitch_joint": 0.35,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_hip_roll_joint": 0.30,
    r".*_hip_pitch_joint": 0.70,
    r".*_knee_pitch_joint": 1.00,
    r"body_pitch_joint": 1.00,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("body_front_link",)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("body_front_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    # Reward modules expect either site_names or body_names; provide site_names
    # corresponding to per-foot sites added to the MJCF.
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names
    cfg.rewards[reward_name].params["asset_cfg"].preserve_order = True

  # With the hardware actuator limits, velocity tracking must dominate the
  # easy local optimum of standing still.  Relax smoothness and posture costs
  # while retaining joint-limit and collision protection.
  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_linear_velocity"].params["std"] = 2.0
  cfg.rewards["forward_velocity_progress"] = RewardTermCfg(
    func=mdp.forward_velocity_progress,
    weight=8.0,
    params={"command_name": "twist"},
  )
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["planar_drift_l2"] = RewardTermCfg(
    func=mdp.planar_drift_l2,
    weight=-1.0,
  )
  cfg.rewards["bilateral_foot_amplitude"] = RewardTermCfg(
    func=mdp.bilateral_foot_amplitude,
    weight=-0.5,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names, preserve_order=True),
      "period": gait_period,
      "amplitude_std": 0.02,
    },
  )
  cfg.rewards["upright"].weight = 0.5
  cfg.rewards["pose"].weight = 0.2
  cfg.rewards["action_rate_l2"].weight = -0.01
  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.25
  cfg.rewards["air_time"].params["threshold_max"] = 0.35

  # Contact sensor order: FL=0, FR=1, RL=2, RR=3.  Keep the requested order
  # explicit; the paper's rotary gallop order would be (0, 1, 3, 2).
  cfg.rewards["gallop_sequence"] = RewardTermCfg(
    func=mdp.footfall_sequence,
    weight=0.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "sequence": (0, 3, 1, 2),
      "command_name": "twist",
      "command_threshold": 0.3,
      "wrong_contact_penalty": 0.5,
    },
  )
  # Previous evenly spaced clock gait.  Kept disabled for comparison; the
  # feline-specific schedule below includes two distinct flight phases.
  cfg.rewards["clock_gallop"] = RewardTermCfg(
    func=mdp.clock_gait,
    weight=0.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "period": gait_period,
      "offsets": (0.0, 0.5, 0.25, 0.75),
      "stance_ratio": 0.20,
      "command_threshold": 0.15,
    },
  )
  # Sensor order: FL, FR, RL, RR.  Contacts progress through the fore pair,
  # collected flight, hind pair, hind push, and extended flight.
  cfg.rewards["feline_gallop_contacts"] = RewardTermCfg(
    func=mdp.feline_gallop_contacts,
    weight=2.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "period": gait_period,
      "stance_intervals": (
        (0.16, 0.34),  # FL
        (0.22, 0.40),  # FR
        (0.61, 0.80),  # RL
        (0.54, 0.73),  # RR
      ),
      "command_threshold": 1.0,
    },
  )
  cfg.rewards["stride_length"] = RewardTermCfg(
    func=mdp.stride_length,
    weight=3.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names, preserve_order=True),
      "base_stride": 0.10,
      "speed_slope": 0.035,
      "max_stride": 0.24,
      "std": 0.04,
    },
  )
  cfg.rewards["flight_phase"] = RewardTermCfg(
    func=mdp.flight_phase,
    weight=0.5,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "speed_threshold": 2.0,
      "min_air_time": 0.025,
      "phase_period": gait_period,
      "phase_windows": ((0.0, 0.16), (0.40, 0.54), (0.80, 1.0)),
    },
  )
  cfg.rewards["extended_flight_posture"] = RewardTermCfg(
    func=mdp.extended_flight_posture,
    weight=3.5,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names, preserve_order=True),
      "speed_threshold": 2.0,
      "min_air_time": 0.025,
      "front_target_x": 0.24,
      "hind_target_x": -0.20,
      "position_std": 0.08,
      "symmetry_std": 0.05,
      "phase_period": gait_period,
      "phase_windows": ((0.0, 0.16), (0.80, 1.0)),
      "metric_prefix": "extended_flight",
    },
  )
  cfg.rewards["collected_flight_posture"] = RewardTermCfg(
    func=mdp.extended_flight_posture,
    weight=4.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names, preserve_order=True),
      "speed_threshold": 2.0,
      "min_air_time": 0.025,
      "front_target_x": 0.10,
      "hind_target_x": -0.04,
      "position_std": 0.08,
      "symmetry_std": 0.05,
      "phase_period": gait_period,
      "phase_windows": ((0.40, 0.54),),
      "metric_prefix": "collected_flight",
    },
  )
  cfg.rewards["hind_propulsion"] = RewardTermCfg(
    func=mdp.hind_propulsion,
    weight=2.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot"),
      "period": gait_period,
      "push_window": (0.54, 0.80),
      "target_acceleration": 8.0,
      "speed_threshold": 2.0,
    },
  )
  cfg.rewards["spine_flexion"] = RewardTermCfg(
    func=mdp.spine_flexion,
    weight=2.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", joint_names=("body_pitch_joint",)),
      "window_s": 0.5,
      "speed_threshold": 1.8,
      "actual_speed_threshold": 0.5,
      "target_range": 0.8,
      "range_std": 0.2,
      "center_std": 0.12,
    },
  )
  cfg.rewards["spine_phase_tracking"] = RewardTermCfg(
    func=mdp.spine_phase_tracking,
    weight=1.5,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", joint_names=("body_pitch_joint",)),
      "period": gait_period,
      # Negative at phase 0/1 (extended flight), positive near phase 0.5
      # (collected flight).
      "amplitude": -0.4,
      "phase_offset": 0.0,
      "std": 0.18,
      "speed_threshold": 1.8,
      "actual_speed_threshold": 0.5,
    },
  )
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=mdp.stand_still,
    weight=-0.5,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "command_threshold": 0.1,
    },
  )
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-2.5e-7,
  )
  # Set foot clearance/swing targets to match the initial MJCF knee offset.
  # The knee/foot site local z-offset in the MJCF is approximately -0.092 m,
  # so the nominal body-to-foot distance at spawn is ~0.092 m.
  cheetah_nominal_foot_height = 0.092
  if "foot_clearance" in cfg.rewards:
    cfg.rewards["foot_clearance"].params["target_height"] = cheetah_nominal_foot_height
    cfg.rewards["foot_clearance"].weight = -0.5
  if "foot_swing_height" in cfg.rewards:
    cfg.rewards["foot_swing_height"].params["target_height"] = (
      cheetah_nominal_foot_height
    )
    cfg.rewards["foot_swing_height"].weight = -0.5

  # Disable command randomization/curriculum so it cannot overwrite the fixed
  # (5, 0, 0) m/s command during training.
  cfg.curriculum.pop("command_vel", None)

  # Previous randomized speed curriculum.  To restore it, uncomment this block
  # together with the randomized command configuration above.
  # command_curriculum = cfg.curriculum["command_vel"]
  # command_curriculum.params["velocity_stages"] = [
  #   {
  #     "step": 0,
  #     "lin_vel_x": (0.0, 1.0),
  #     "lin_vel_y": (0.0, 0.0),
  #     "ang_vel_z": (0.0, 0.0),
  #   },
  #   {"step": 1000 * 24, "lin_vel_x": (0.0, 1.8)},
  #   {"step": 2000 * 24, "lin_vel_x": (0.0, 3.0)},
  #   {"step": 3000 * 24, "lin_vel_x": (0.0, 4.0)},
  # ]

  # Per-body-group collision penalties.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": shank_ground_cfg.name},
  )
  cfg.rewards["trunk_head_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": trunk_head_ground_cfg.name},
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


def cheetah_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Cheetah flat terrain velocity configuration."""
  cfg = cheetah_rough_env_cfg(play=play)

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
