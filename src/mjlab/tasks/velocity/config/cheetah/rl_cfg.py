"""PPO configuration for the Go2-style Cheetah locomotion baseline."""

from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


def cheetah_baseline_ppo_runner_cfg():
  """Reuse the proven Unitree quadruped PPO hyperparameters."""
  cfg = unitree_go1_ppo_runner_cfg()
  cfg.experiment_name = "cheetah_go2_baseline"
  return cfg


def cheetah_high_speed_ppo_runner_cfg():
  """Use gentler PPO updates and preserve exploration above 3.5 m/s."""
  cfg = cheetah_baseline_ppo_runner_cfg()
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.algorithm.entropy_coef = 0.015
  cfg.save_interval = 25
  return cfg


def cheetah_high_speed_dense_checkpoints_ppo_runner_cfg():
  """High-speed fine-tuning with checkpoints dense enough to catch collapse."""
  cfg = cheetah_high_speed_ppo_runner_cfg()
  cfg.save_interval = 10
  return cfg


class CheetahHighSpeedFinetuneRunner(VelocityOnPolicyRunner):
  """Resume network weights but create a fresh, lower-rate optimizer."""

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    if load_cfg is None:
      load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": True,
        "rnd": False,
      }
    return super().load(
      path,
      load_cfg=load_cfg,
      strict=strict,
      map_location=map_location,
    )
