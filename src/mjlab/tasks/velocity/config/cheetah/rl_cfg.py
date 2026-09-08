"""RL configuration for Cheetah velocity task."""

from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg


def cheetah_ppo_runner_cfg():
  """Currently reuse the Go1 RL runner config.

  Replace or adjust experiment names and hyperparameters as needed for the
  dedicated Cheetah model.
  """
  cfg = unitree_go1_ppo_runner_cfg()
  try:
    cfg.experiment_name = "cheetah_velocity"
  except Exception:
    pass
  return cfg


def reborn_cheetah_ppo_runner_cfg(
  *, rigid_spine: bool = False, spine_reward: bool = True
):
  """Create the longer, bounded-action run used by the Reborn task."""
  cfg = unitree_go1_ppo_runner_cfg()
  if rigid_spine:
    cfg.experiment_name = "reborn_cheetah_velocity_v4_rigid"
  elif not spine_reward:
    cfg.experiment_name = "reborn_cheetah_velocity_v4_unshaped"
  else:
    cfg.experiment_name = "reborn_cheetah_velocity_v4"
  cfg.max_iterations = 10_000
  cfg.clip_actions = 1.0
  return cfg
