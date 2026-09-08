"""Tests for the clock-free Cheetah research environment."""

from unittest.mock import MagicMock

import mujoco
import pytest
import torch

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp.curriculums import adaptive_command_velocity
from mjlab.tasks.velocity.mdp.rewards import (
  footfall_sequence,
  gallop_pair_phase,
  gallop_pair_sequence,
  spine_contact_phase,
  spine_leg_coordination,
  straight_line_deviation_l2,
)
from mjlab.tasks.velocity.mdp.terminations import excessive_straight_line_deviation
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

ACTIVE_TASK = "Mjlab-Velocity-Flat-Reborn-Cheetah"
RIGID_TASK = "Mjlab-Velocity-Flat-Reborn-Cheetah-Rigid"
UNSHAPED_TASK = "Mjlab-Velocity-Flat-Reborn-Cheetah-Unshaped"


def test_reborn_config_removes_clock_and_bounds_actions() -> None:
  cfg = load_env_cfg(ACTIVE_TASK)
  rl_cfg = load_rl_cfg(ACTIVE_TASK)

  assert "gait_phase" in cfg.observations["actor"].terms
  assert "feline_gallop_contacts" not in cfg.rewards
  assert "spine_phase_tracking" not in cfg.rewards
  assert "gallop_pair_phase" in cfg.rewards
  assert cfg.rewards["gallop_sequence"].weight == pytest.approx(0.75)
  assert cfg.rewards["flight_phase"].weight == pytest.approx(0.35)
  assert cfg.rewards["spine_contact_phase"].weight == pytest.approx(0.8)
  assert "spine_leg_coordination" in cfg.rewards
  assert cfg.rewards["gallop_pair_phase"].weight == pytest.approx(0.25)
  assert cfg.rewards["spine_leg_coordination"].weight == pytest.approx(0.5)
  assert "planar_drift" in cfg.rewards
  assert "straight_line_deviation" in cfg.rewards
  assert "straight_line_deviation" in cfg.terminations
  assert rl_cfg.clip_actions == 1.0
  assert rl_cfg.experiment_name == "reborn_cheetah_velocity_v4"

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  assert action.clip is not None
  assert action.clip[r"body_pitch_joint"] == (-0.60, 0.30)
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  actuator_cfgs = articulation.actuators
  assert all(isinstance(actuator, DcMotorActuatorCfg) for actuator in actuator_cfgs)
  twist_cfg = cfg.commands["twist"]
  assert isinstance(twist_cfg, UniformVelocityCommandCfg)
  assert twist_cfg.rel_world_envs == 1.0
  assert twist_cfg.ranges.lin_vel_y == (0.0, 0.0)
  assert twist_cfg.ranges.ang_vel_z == (0.0, 0.0)


def test_reborn_rigid_ablation_compiles_with_locked_spine() -> None:
  cfg = load_env_cfg(RIGID_TASK)
  rl_cfg = load_rl_cfg(RIGID_TASK)
  model = Entity(cfg.scene.entities["robot"]).compile()

  assert isinstance(model, mujoco.MjModel)
  assert tuple(model.joint("body_pitch_joint").range) == pytest.approx(
    (-1.0e-4, 1.0e-4)
  )
  assert cfg.rewards["spine_leg_coordination"].weight == 0.0
  assert cfg.rewards["spine_contact_phase"].weight == 0.0
  assert rl_cfg.experiment_name == "reborn_cheetah_velocity_v4_rigid"


def test_reborn_unshaped_ablation_keeps_active_spine_without_spine_reward() -> None:
  cfg = load_env_cfg(UNSHAPED_TASK)
  rl_cfg = load_rl_cfg(UNSHAPED_TASK)
  model = Entity(cfg.scene.entities["robot"]).compile()

  assert tuple(model.joint("body_pitch_joint").range) == pytest.approx((-1.0, 1.0))
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  assert action.clip is not None
  assert action.clip[r"body_pitch_joint"] == (-0.60, 0.30)
  assert cfg.rewards["spine_leg_coordination"].weight == 0.0
  assert rl_cfg.experiment_name == "reborn_cheetah_velocity_v4_unshaped"


def test_straight_line_corridor_penalizes_and_terminates_without_locking_base() -> None:
  asset = MagicMock()
  asset.data.root_link_pos_w = torch.tensor([[0.0, 0.40, 0.3], [0.0, 0.80, 0.3]])
  asset.data.heading_w = torch.tensor([0.0, 0.0])
  env = MagicMock()
  env.scene.__getitem__.return_value = asset
  env.scene.env_origins = torch.zeros((2, 3))
  env.episode_length_buf = torch.tensor([10, 10])
  env.extras = {"log": {}}

  penalty = straight_line_deviation_l2(
    env, lateral_tolerance=0.25, heading_tolerance=0.26
  )
  terminated = excessive_straight_line_deviation(
    env,
    maximum_lateral_displacement=0.75,
    maximum_heading_error=0.79,
  )

  assert penalty[0].item() > 0.0
  assert terminated.tolist() == [False, True]


def test_spine_contact_phase_matches_hind_extension_and_front_compression() -> None:
  asset = MagicMock()
  asset.data.joint_pos = torch.tensor([[0.22]])
  asset.data.joint_vel = torch.tensor([[0.8]])
  sensor = MagicMock()
  sensor.compute_first_contact.side_effect = [
    torch.tensor([[False, False, True, False]]),
    torch.tensor([[False, False, False, True]]),
    torch.tensor([[True, False, False, False]]),
    torch.tensor([[False, True, False, False]]),
  ]
  env = MagicMock()
  env.num_envs = 1
  env.device = "cpu"
  env.step_dt = 0.02
  env.episode_length_buf = torch.tensor([50])
  env.scene.__getitem__.side_effect = lambda name: (
    sensor if name == "feet_ground_contact" else asset
  )
  env.command_manager.get_command.return_value = torch.tensor([[2.0, 0.0, 0.0]])
  env.extras = {"log": {}}
  spine_cfg = SceneEntityCfg("robot", joint_ids=[0])
  term = spine_contact_phase(
    RewardTermCfg(func=spine_contact_phase, weight=1.0, params={}), env
  )

  first_touchdown = term(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    spine_cfg=spine_cfg,
  )
  assert first_touchdown.item() == 0.0

  env.episode_length_buf = torch.tensor([55])
  rear_extension = term(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    spine_cfg=spine_cfg,
  ).item()

  term.reset(torch.tensor([0]))
  asset.data.joint_pos[:, 0] = -0.32
  asset.data.joint_vel[:, 0] = -0.8
  env.episode_length_buf = torch.tensor([100])
  front_touchdown = term(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    spine_cfg=spine_cfg,
  )
  assert front_touchdown.item() == 0.0

  env.episode_length_buf = torch.tensor([105])
  front_compression = term(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    spine_cfg=spine_cfg,
  ).item()

  assert rear_extension > 0.7
  assert front_compression > 0.7


def test_gallop_pair_sequence_orders_hind_then_front_pair() -> None:
  sensor = MagicMock()
  sensor.compute_first_contact.side_effect = [
    torch.tensor([[False, False, True, False]]),
    torch.tensor([[False, False, False, True]]),
    torch.tensor([[True, False, False, False]]),
    torch.tensor([[False, True, False, False]]),
    torch.tensor([[False, False, True, False]]),
    torch.tensor([[False, False, False, True]]),
    torch.tensor([[True, False, False, False]]),
    torch.tensor([[False, True, False, False]]),
  ]
  env = MagicMock()
  env.num_envs = 1
  env.device = "cpu"
  env.step_dt = 0.02
  env.episode_length_buf = torch.tensor([50])
  env.scene.__getitem__.return_value = sensor
  env.command_manager.get_command.return_value = torch.tensor([[2.0, 0.0, 0.0]])
  env.extras = {"log": {}}
  term = gallop_pair_sequence(
    RewardTermCfg(func=gallop_pair_sequence, weight=1.0, params={}), env
  )

  assert term(env, "feet_ground_contact", "twist").item() == 0.0
  env.episode_length_buf = torch.tensor([55])
  assert term(env, "feet_ground_contact", "twist").item() == pytest.approx(1.0)
  env.episode_length_buf = torch.tensor([61])
  assert term(env, "feet_ground_contact", "twist").item() == 0.0
  env.episode_length_buf = torch.tensor([67])
  assert term(env, "feet_ground_contact", "twist").item() == pytest.approx(1.0)
  env.episode_length_buf = torch.tensor([80])
  assert term(env, "feet_ground_contact", "twist").item() == 0.0
  env.episode_length_buf = torch.tensor([85])
  assert term(env, "feet_ground_contact", "twist").item() == pytest.approx(1.0)
  env.episode_length_buf = torch.tensor([90])
  assert term(env, "feet_ground_contact", "twist").item() == 0.0
  env.episode_length_buf = torch.tensor([95])
  assert term(env, "feet_ground_contact", "twist").item() == pytest.approx(1.0)


def test_footfall_sequence_advances_only_on_debounced_contacts() -> None:
  sensor = MagicMock()
  env = MagicMock()
  env.num_envs = 1
  env.device = "cpu"
  env.step_dt = 0.02
  env.episode_length_buf = torch.tensor([50])
  env.scene.__getitem__.return_value = sensor
  env.command_manager.get_command.return_value = torch.tensor([[2.0, 0.0, 0.0]])
  env.extras = {"log": {}}
  params = {
    "sensor_name": "feet_ground_contact",
    "sequence": (2, 3, 1, 0),
    "command_name": "twist",
    "command_threshold": 1.0,
    "wrong_contact_penalty": 0.25,
    "min_period": 0.12,
  }
  reward = footfall_sequence(
    RewardTermCfg(func=footfall_sequence, weight=1.0, params=params), env
  )
  sensor.compute_first_contact.return_value = torch.tensor(
    [[False, False, True, False]]
  )
  assert reward(
    env,
    sensor_name="feet_ground_contact",
    sequence=(2, 3, 1, 0),
    command_name="twist",
    command_threshold=1.0,
    wrong_contact_penalty=0.25,
    min_period=0.12,
  ).item() == pytest.approx(1.0)
  env.episode_length_buf = torch.tensor([51])
  assert reward(
    env,
    sensor_name="feet_ground_contact",
    sequence=(2, 3, 1, 0),
    command_name="twist",
    command_threshold=1.0,
    wrong_contact_penalty=0.25,
    min_period=0.12,
  ).item() == pytest.approx(0.0)
  assert reward.expected_phase.item() == 1


def test_spine_leg_coordination_prefers_matching_direction() -> None:
  asset = MagicMock()
  asset.data.joint_pos = torch.tensor([[-0.4, 0.0, 0.0, 0.0, 0.0]])
  asset.data.joint_vel = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0]])
  asset.data.root_link_lin_vel_b = torch.tensor([[2.0, 0.0, 0.0]])
  env = MagicMock()
  env.num_envs = 1
  env.device = "cpu"
  env.step_dt = 0.02
  env.scene.__getitem__.return_value = asset
  env.command_manager.get_command.return_value = torch.tensor([[2.0, 0.0, 0.0]])
  env.extras = {"log": {}}
  spine_cfg = SceneEntityCfg("robot", joint_ids=[0])
  leg_cfg = SceneEntityCfg("robot", joint_ids=[1, 2, 3, 4])

  params = {
    "command_name": "twist",
    "spine_cfg": spine_cfg,
    "leg_cfg": leg_cfg,
    "filter_time_constant": 0.05,
  }
  matching_term = spine_leg_coordination(
    RewardTermCfg(func=spine_leg_coordination, weight=1.0, params=params), env
  )
  matching = torch.zeros(1)
  for step in range(20):
    direction = 1.0 if step % 2 == 0 else -1.0
    asset.data.joint_vel[:, 1:3] = -direction
    asset.data.joint_vel[:, 3:5] = direction
    asset.data.joint_vel[:, 0] = -direction
    asset.data.joint_pos[:, 0] = -0.4 if direction > 0.0 else 0.2
    matching = matching_term(
      env,
      command_name="twist",
      spine_cfg=spine_cfg,
      leg_cfg=leg_cfg,
      filter_time_constant=0.05,
    )

  opposing_term = spine_leg_coordination(
    RewardTermCfg(func=spine_leg_coordination, weight=1.0, params=params), env
  )
  opposing = torch.zeros(1)
  for step in range(20):
    direction = 1.0 if step % 2 == 0 else -1.0
    asset.data.joint_vel[:, 1:3] = -direction
    asset.data.joint_vel[:, 3:5] = direction
    asset.data.joint_vel[:, 0] = direction
    asset.data.joint_pos[:, 0] = -0.4 if direction > 0.0 else 0.2
    opposing = opposing_term(
      env,
      command_name="twist",
      spine_cfg=spine_cfg,
      leg_cfg=leg_cfg,
      filter_time_constant=0.05,
    )

  assert matching.item() > 0.0
  assert opposing.item() < 0.0


def test_gallop_pair_phase_rewards_target_touchdown_offset() -> None:
  sensor = MagicMock()
  command = torch.tensor([[2.0, 0.0, 0.0]])
  env = MagicMock()
  env.num_envs = 1
  env.device = "cpu"
  env.step_dt = 0.02
  env.episode_length_buf = torch.tensor([50])
  env.scene.__getitem__.return_value = sensor
  env.command_manager.get_command.return_value = command
  env.extras = {"log": {}}
  params = {
    "sensor_name": "feet_ground_contact",
    "command_name": "twist",
    "pairs": ((1, 0), (2, 3)),
    "target_offsets": (0.12, 0.12),
    "initial_period": 0.5,
    "offset_std": 0.06,
  }
  reward = gallop_pair_phase(
    RewardTermCfg(func=gallop_pair_phase, weight=1.0, params=params), env
  )

  sensor.compute_first_contact.return_value = torch.tensor([[False, True, True, False]])
  reward(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    pairs=((1, 0), (2, 3)),
    target_offsets=(0.12, 0.12),
    initial_period=0.5,
    offset_std=0.06,
  )
  env.episode_length_buf = torch.tensor([53])
  sensor.compute_first_contact.return_value = torch.tensor([[True, False, False, True]])

  result = reward(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    pairs=((1, 0), (2, 3)),
    target_offsets=(0.12, 0.12),
    initial_period=0.5,
    offset_std=0.06,
  )

  assert result.item() == pytest.approx(1.0)


def test_gallop_pair_phase_rejects_touchdown_chatter() -> None:
  sensor = MagicMock()
  env = MagicMock()
  env.num_envs = 1
  env.device = "cpu"
  env.step_dt = 0.02
  env.episode_length_buf = torch.tensor([50])
  env.scene.__getitem__.return_value = sensor
  env.command_manager.get_command.return_value = torch.tensor([[2.0, 0.0, 0.0]])
  env.extras = {"log": {}}
  params = {
    "sensor_name": "feet_ground_contact",
    "command_name": "twist",
    "pairs": ((1, 0), (2, 3)),
    "target_offsets": (0.12, 0.12),
    "initial_period": 0.5,
    "offset_std": 0.06,
  }
  reward = gallop_pair_phase(
    RewardTermCfg(func=gallop_pair_phase, weight=1.0, params=params), env
  )

  sensor.compute_first_contact.return_value = torch.tensor([[False, True, True, False]])
  reward(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    pairs=((1, 0), (2, 3)),
    target_offsets=(0.12, 0.12),
    initial_period=0.5,
    offset_std=0.06,
  )
  env.episode_length_buf = torch.tensor([53])
  sensor.compute_first_contact.return_value = torch.tensor([[True, False, False, True]])
  first = reward(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    pairs=((1, 0), (2, 3)),
    target_offsets=(0.12, 0.12),
    initial_period=0.5,
    offset_std=0.06,
  )
  accepted_time = reward.last_touchdown.clone()

  env.episode_length_buf = torch.tensor([54])
  repeated = reward(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    pairs=((1, 0), (2, 3)),
    target_offsets=(0.12, 0.12),
    initial_period=0.5,
    offset_std=0.06,
  )

  assert first.item() == pytest.approx(1.0)
  assert repeated.item() == pytest.approx(first.item())
  assert reward.last_touchdown == pytest.approx(accepted_time)
  assert env.extras["log"]["Metrics/gallop_rejected_touchdown_fraction"] == 1.0


def test_adaptive_curriculum_expands_range_after_success() -> None:
  command_cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(20.0, 20.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.5, 1.0),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(0.0, 0.0),
    ),
  )
  env = MagicMock()
  env.num_envs = 2
  env.device = "cpu"
  env.step_dt = 0.02
  env.common_step_counter = 1200
  env.episode_length_buf = torch.tensor([100, 100])
  command_term = MagicMock()
  command_term.cfg = command_cfg
  command_term.target_vel_command_b = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
  env.command_manager.get_term.return_value = command_term
  env.reward_manager.get_term_cfg.return_value.weight = 4.0
  env.reward_manager._episode_sums = {
    "track_linear_velocity": torch.full((2,), 100 * 0.02 * 4.0 * 0.9)
  }
  params = {
    "command_name": "twist",
    "reward_name": "track_linear_velocity",
    "max_velocity": 5.0,
    "increment": 0.25,
    "success_threshold": 0.8,
    "min_episodes": 2,
    "min_steps_per_stage": 1200,
    "frontier_fraction": 0.75,
    "ema_alpha": 0.2,
  }
  env.common_step_counter = 0
  curriculum = adaptive_command_velocity(
    CurriculumTermCfg(func=adaptive_command_velocity, params=params), env
  )
  env.common_step_counter = 1200

  state = curriculum(
    env,
    torch.tensor([0, 1]),
    command_name="twist",
    reward_name="track_linear_velocity",
    max_velocity=5.0,
    increment=0.25,
    success_threshold=0.8,
    min_episodes=2,
    min_steps_per_stage=1200,
    frontier_fraction=0.75,
    ema_alpha=0.2,
  )

  assert command_cfg.ranges.lin_vel_x == (0.5, 1.25)
  assert state["lin_vel_x_max"].item() == pytest.approx(1.25)
  assert state["last_advance_score"].item() == pytest.approx(0.9)


def test_adaptive_curriculum_requires_frontier_episodes_and_stage_dwell() -> None:
  command_cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(20.0, 20.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.5, 1.0),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(0.0, 0.0),
    ),
  )
  env = MagicMock()
  env.num_envs = 2
  env.device = "cpu"
  env.step_dt = 0.02
  env.common_step_counter = 0
  env.episode_length_buf = torch.tensor([100, 100])
  command_term = MagicMock()
  command_term.cfg = command_cfg
  command_term.target_vel_command_b = torch.tensor([[0.6, 0.0, 0.0], [1.0, 0.0, 0.0]])
  env.command_manager.get_term.return_value = command_term
  env.reward_manager.get_term_cfg.return_value.weight = 4.0
  env.reward_manager._episode_sums = {
    "track_linear_velocity": torch.full((2,), 100 * 0.02 * 4.0 * 0.9)
  }
  params = {
    "command_name": "twist",
    "reward_name": "track_linear_velocity",
    "max_velocity": 5.0,
    "increment": 0.25,
    "success_threshold": 0.8,
    "min_episodes": 1,
    "min_steps_per_stage": 1200,
    "frontier_fraction": 0.75,
    "ema_alpha": 0.2,
  }
  curriculum = adaptive_command_velocity(
    CurriculumTermCfg(func=adaptive_command_velocity, params=params), env
  )

  env.common_step_counter = 1199
  state = curriculum(
    env,
    torch.tensor([0, 1]),
    command_name="twist",
    reward_name="track_linear_velocity",
    max_velocity=5.0,
    increment=0.25,
    success_threshold=0.8,
    min_episodes=1,
    min_steps_per_stage=1200,
    frontier_fraction=0.75,
    ema_alpha=0.2,
  )
  assert command_cfg.ranges.lin_vel_x == (0.5, 1.0)
  assert state["eligible_episodes"].item() == 1

  env.common_step_counter = 1200
  state = curriculum(
    env,
    torch.tensor([0, 1]),
    command_name="twist",
    reward_name="track_linear_velocity",
    max_velocity=5.0,
    increment=0.25,
    success_threshold=0.8,
    min_episodes=1,
    min_steps_per_stage=1200,
    frontier_fraction=0.75,
    ema_alpha=0.2,
  )
  assert command_cfg.ranges.lin_vel_x == (0.5, 1.25)
  assert state["eligible_episodes"].item() == 0
