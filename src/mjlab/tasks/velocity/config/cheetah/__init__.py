from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  cheetah_flat_env_cfg,
  cheetah_rough_env_cfg,
)
from .reborn_env_cfg import reborn_cheetah_flat_env_cfg
from .rl_cfg import cheetah_ppo_runner_cfg, reborn_cheetah_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Cheetah",
  env_cfg=cheetah_rough_env_cfg(),
  play_env_cfg=cheetah_rough_env_cfg(play=True),
  rl_cfg=cheetah_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Reborn-Cheetah",
  env_cfg=reborn_cheetah_flat_env_cfg(),
  play_env_cfg=reborn_cheetah_flat_env_cfg(play=True),
  rl_cfg=reborn_cheetah_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Reborn-Cheetah-Rigid",
  env_cfg=reborn_cheetah_flat_env_cfg(rigid_spine=True),
  play_env_cfg=reborn_cheetah_flat_env_cfg(play=True, rigid_spine=True),
  rl_cfg=reborn_cheetah_ppo_runner_cfg(rigid_spine=True),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Reborn-Cheetah-Unshaped",
  env_cfg=reborn_cheetah_flat_env_cfg(spine_reward=False),
  play_env_cfg=reborn_cheetah_flat_env_cfg(play=True, spine_reward=False),
  rl_cfg=reborn_cheetah_ppo_runner_cfg(spine_reward=False),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah",
  env_cfg=cheetah_flat_env_cfg(),
  play_env_cfg=cheetah_flat_env_cfg(play=True),
  rl_cfg=cheetah_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
