"""Task registration for the Go2-style Cheetah baseline."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  cheetah_go2_baseline_env_cfg,
  cheetah_go2_baseline_stage2_env_cfg,
  cheetah_go2_baseline_stage3_env_cfg,
  cheetah_go2_baseline_stage4_env_cfg,
  cheetah_go2_baseline_stage5_env_cfg,
  cheetah_go2_baseline_stage6_env_cfg,
  cheetah_go2_baseline_stage7_env_cfg,
  cheetah_go2_baseline_stage8_env_cfg,
  cheetah_go2_baseline_stage9_env_cfg,
  cheetah_go2_baseline_stage10_env_cfg,
  cheetah_go2_baseline_stage11_env_cfg,
  cheetah_go2_baseline_stage12_env_cfg,
  cheetah_go2_baseline_stage13_env_cfg,
  cheetah_go2_baseline_stage14_env_cfg,
  cheetah_go2_baseline_stage15_env_cfg,
  cheetah_go2_baseline_stage16_env_cfg,
)
from .rl_cfg import (
  CheetahHighSpeedFinetuneRunner,
  cheetah_baseline_ppo_runner_cfg,
  cheetah_high_speed_dense_checkpoints_ppo_runner_cfg,
  cheetah_high_speed_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline",
  env_cfg=cheetah_go2_baseline_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage2",
  env_cfg=cheetah_go2_baseline_stage2_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage2_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage3",
  env_cfg=cheetah_go2_baseline_stage3_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage3_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage4",
  env_cfg=cheetah_go2_baseline_stage4_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage4_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage5",
  env_cfg=cheetah_go2_baseline_stage5_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage5_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage6",
  env_cfg=cheetah_go2_baseline_stage6_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage6_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage7",
  env_cfg=cheetah_go2_baseline_stage7_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage7_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage8",
  env_cfg=cheetah_go2_baseline_stage8_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage8_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage9",
  env_cfg=cheetah_go2_baseline_stage9_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage9_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage10",
  env_cfg=cheetah_go2_baseline_stage10_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage10_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage11",
  env_cfg=cheetah_go2_baseline_stage11_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage11_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage12",
  env_cfg=cheetah_go2_baseline_stage12_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage12_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage13",
  env_cfg=cheetah_go2_baseline_stage13_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage13_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage14",
  env_cfg=cheetah_go2_baseline_stage14_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage14_env_cfg(play=True),
  rl_cfg=cheetah_baseline_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage15",
  env_cfg=cheetah_go2_baseline_stage15_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage15_env_cfg(play=True),
  rl_cfg=cheetah_high_speed_ppo_runner_cfg(),
  runner_cls=CheetahHighSpeedFinetuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cheetah-Go2-Baseline-Stage16",
  env_cfg=cheetah_go2_baseline_stage16_env_cfg(),
  play_env_cfg=cheetah_go2_baseline_stage16_env_cfg(play=True),
  rl_cfg=cheetah_high_speed_dense_checkpoints_ppo_runner_cfg(),
  runner_cls=CheetahHighSpeedFinetuneRunner,
)
