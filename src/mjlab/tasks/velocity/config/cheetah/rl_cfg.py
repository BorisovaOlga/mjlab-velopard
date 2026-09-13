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


def cheetah_gallop_finetune_cfg() -> RslRlOnPolicyRunnerCfg:
  """RL runner config for fine-tuning a gallop (rotational) policy from Go2.

  This config initializes from a stable Go2 checkpoint and lowers the
  learning rate for conservative fine-tuning aimed at recovering the
  rotational gallop and correct flight postures.
  """
  cfg = unitree_go1_ppo_runner_cfg()
  try:
    cfg.experiment_name = "cheetah_gallop_finetune"
    cfg.algorithm.learning_rate = 1.0e-4
    cfg.save_interval = 25
    # Resume from the Go2 baseline checkpoint. Update path/checkpoint as needed.
    cfg.resume = True
    cfg.load_run = "cheetah_go2_baseline/2026-09-10_14-15-33_straight_stable_from_10994"
    cfg.load_checkpoint = "model_11293.pt"
  except Exception:
    pass
  return cfg
