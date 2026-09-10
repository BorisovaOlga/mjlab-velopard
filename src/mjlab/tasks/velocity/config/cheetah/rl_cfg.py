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
    # Fine-tuning from the stable 4 m/s baseline must be conservative.
    cfg.algorithm.learning_rate = 1.0e-4
  except Exception:
    pass
  return cfg
