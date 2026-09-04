"""PPO configuration for the Go2-style Cheetah locomotion baseline."""

from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg


def cheetah_baseline_ppo_runner_cfg():
  """Reuse the proven Unitree quadruped PPO hyperparameters."""
  cfg = unitree_go1_ppo_runner_cfg()
  cfg.experiment_name = "cheetah_go2_baseline"
  return cfg
