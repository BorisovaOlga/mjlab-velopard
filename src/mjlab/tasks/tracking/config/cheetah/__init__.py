"""Motion imitation configuration for the articulated Cheetah."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import reborn_cheetah_flat_tracking_env_cfg
from .rl_cfg import reborn_cheetah_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Reborn-Cheetah",
  env_cfg=reborn_cheetah_flat_tracking_env_cfg(),
  play_env_cfg=reborn_cheetah_flat_tracking_env_cfg(play=True),
  rl_cfg=reborn_cheetah_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Reborn-Cheetah-Rigid",
  env_cfg=reborn_cheetah_flat_tracking_env_cfg(rigid_spine=True),
  play_env_cfg=reborn_cheetah_flat_tracking_env_cfg(play=True, rigid_spine=True),
  rl_cfg=reborn_cheetah_tracking_ppo_runner_cfg(
    experiment_name="reborn_cheetah_imitation_rigid_power"
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)
