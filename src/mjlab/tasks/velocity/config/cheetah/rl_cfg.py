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


def cheetah_classic_ppo_runner_cfg():
  """Go2-style PPO configuration with separate experiment logs."""
  cfg = unitree_go1_ppo_runner_cfg()
  cfg.experiment_name = "cheetah_classic_velocity"
  cfg.save_interval = 100
  cfg.max_iterations = 10_001
  return cfg
