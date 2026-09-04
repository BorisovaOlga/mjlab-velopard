"""Task registration for the Go2-style Cheetah baseline."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import cheetah_go2_baseline_env_cfg
from .rl_cfg import cheetah_baseline_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline",
  env_cfg=cheetah_go2_baseline_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
