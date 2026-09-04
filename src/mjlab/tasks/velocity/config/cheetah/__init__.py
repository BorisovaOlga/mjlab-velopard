from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  cheetah_flat_contact_phase_finetune_env_cfg,
  cheetah_flat_duration_finetune_env_cfg,
  cheetah_flat_env_cfg,
  cheetah_flat_flight_finetune_env_cfg,
  cheetah_rough_env_cfg,
)
from .rl_cfg import cheetah_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Cheetah",
  env_cfg=cheetah_rough_env_cfg(),
  play_env_cfg=cheetah_rough_env_cfg(play=True),
  rl_cfg=cheetah_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Contact-Phase-Finetune",
  env_cfg=cheetah_flat_contact_phase_finetune_env_cfg(),
  play_env_cfg=cheetah_flat_contact_phase_finetune_env_cfg(play=True),
  rl_cfg=cheetah_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Duration-Finetune",
  env_cfg=cheetah_flat_duration_finetune_env_cfg(),
  play_env_cfg=cheetah_flat_duration_finetune_env_cfg(play=True),
  rl_cfg=cheetah_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Flight-Finetune",
  env_cfg=cheetah_flat_flight_finetune_env_cfg(),
  play_env_cfg=cheetah_flat_flight_finetune_env_cfg(play=True),
  rl_cfg=cheetah_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah",
  env_cfg=cheetah_flat_env_cfg(),
  play_env_cfg=cheetah_flat_env_cfg(play=True),
  rl_cfg=cheetah_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
