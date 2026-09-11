"""Compare actuator energy metrics for one or more tracking checkpoints.

Example::

  uv run python -m mjlab.tasks.tracking.scripts.compare_energy \
    --pair 'flex::logs/.../model_2250.pt::docs/.../flex.npz;rigid::logs/.../model_2250.pt::docs/.../rigid.npz' \
    --output-dir logs/energy_compare_2250

The pair format is ``label::checkpoint[::motion_file[::task_id]]``.  The
motion file can be omitted when the task configuration already provides one.
When the task id is omitted, labels or paths containing ``rigid`` select the
rigid Cheetah task; all other pairs use the flexible task.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
import warp

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg

_FLEX_TASK = "Mjlab-Tracking-Flat-Reborn-Cheetah"
_RIGID_TASK = "Mjlab-Tracking-Flat-Reborn-Cheetah-Rigid"
_DEFAULT_GRAVITY = 9.81


@dataclass(frozen=True)
class Pair:
  label: str
  checkpoint: Path
  motion_file: Path | None
  task_id: str


@dataclass(frozen=True)
class Config:
  pair: str
  """Semicolon-separated label::checkpoint[::motion_file[::task_id]] specs."""

  output_dir: Path = Path("logs/energy_compare")
  num_envs: int = 32
  steps: int = 300
  device: str = "cpu"
  seed: int | None = None
  gravity: float = _DEFAULT_GRAVITY


def _infer_task(label: str, checkpoint: Path) -> str:
  value = f"{label} {checkpoint}".lower()
  return _RIGID_TASK if "rigid" in value else _FLEX_TASK


def _parse_pair(spec: str) -> Pair:
  parts = spec.split("::")
  if len(parts) not in (2, 3, 4):
    raise ValueError(
      f"Pair must be label::checkpoint[::motion_file[::task_id]], got {spec!r}"
    )

  label = parts[0].strip()
  checkpoint = Path(parts[1]).expanduser()
  motion_file = None
  if len(parts) >= 3 and parts[2].strip() not in ("", "-"):
    motion_file = Path(parts[2]).expanduser()
  task_id = parts[3] if len(parts) == 4 and parts[3] else _infer_task(label, checkpoint)
  if not label:
    raise ValueError(f"Pair label is empty: {spec!r}")
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
  if motion_file is not None and not motion_file.exists():
    raise FileNotFoundError(f"Motion file does not exist: {motion_file}")
  return Pair(label, checkpoint, motion_file, task_id)


def _mean_active(values: list[torch.Tensor]) -> float:
  if not values:
    return float("nan")
  return torch.cat(values).mean().item()


def _plot_summary(results: list[dict[str, Any]], output: Path) -> None:
  labels = [str(result["label"]) for result in results]
  x = np.arange(len(labels))
  width = 0.25
  fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

  axes[0, 0].bar(
    x - width, [r["leg_abs_power_w"] for r in results], width, label="legs"
  )
  axes[0, 0].bar(x, [r["spine_abs_power_w"] for r in results], width, label="spine")
  axes[0, 0].bar(
    x + width, [r["total_abs_power_w"] for r in results], width, label="total"
  )
  axes[0, 0].set_ylabel("Power (W)")
  axes[0, 0].set_title("Absolute actuator power")
  axes[0, 0].legend()

  axes[0, 1].bar(
    x - width / 2,
    [r["spine_positive_power_w"] for r in results],
    width,
    label="positive",
  )
  axes[0, 1].bar(
    x + width / 2,
    [r["spine_absorbed_power_w"] for r in results],
    width,
    label="absorbed",
  )
  axes[0, 1].set_ylabel("Power (W)")
  axes[0, 1].set_title("Spine power direction")
  axes[0, 1].legend()

  axes[1, 0].bar(
    x - width / 2, [r["energy_per_meter_j_m"] for r in results], width, label="J/m"
  )
  axes[1, 0].bar(
    x + width / 2, [r["cost_of_transport"] for r in results], width, label="CoT"
  )
  axes[1, 0].set_ylabel("Value")
  axes[1, 0].set_title("Locomotion efficiency")
  axes[1, 0].legend()

  axes[1, 1].bar(x, [r["forward_speed_mps"] for r in results], width)
  axes[1, 1].set_ylabel("Speed (m/s)")
  axes[1, 1].set_title("Forward speed")

  for axis in axes.flat:
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.25)
  fig.savefig(output, dpi=160)
  plt.close(fig)


def _plot_timeseries(series: list[dict[str, Any]], output: Path) -> None:
  fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
  for item in series:
    time = np.asarray(item["time_s"])
    label = str(item["label"])
    axes[0].plot(time, item["total_power_w"], label=f"{label} total")
    axes[0].plot(time, item["leg_power_w"], linestyle="--", label=f"{label} legs")
    axes[0].plot(time, item["spine_power_w"], linestyle=":", label=f"{label} spine")
    axes[1].plot(time, item["spine_positive_power_w"], label=f"{label} positive")
    axes[1].plot(
      time, item["spine_absorbed_power_w"], linestyle="--", label=f"{label} absorbed"
    )
    axes[2].plot(time, item["speed_mps"], label=label)
  axes[0].set_ylabel("Power (W)")
  axes[0].set_title("Power over playback")
  axes[1].set_ylabel("Spine power (W)")
  axes[1].set_title("Spine positive and absorbed power")
  axes[2].set_ylabel("Speed (m/s)")
  axes[2].set_xlabel("Time (s)")
  for axis in axes:
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
  fig.savefig(output, dpi=160)
  plt.close(fig)


def _evaluate(pair: Pair, cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
  env_cfg = load_env_cfg(pair.task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  motion = env_cfg.commands.get("motion")
  if not isinstance(motion, MotionCommandCfg):
    raise ValueError(f"Task {pair.task_id} does not define a motion command")
  if pair.motion_file is not None:
    motion.motion_file = str(pair.motion_file)
  motion.sampling_mode = "start"
  motion.joint_position_range = (0.0, 0.0)
  env_cfg.observations["actor"].enable_corruption = False
  if cfg.seed is not None:
    env_cfg.seed = cfg.seed

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  agent_cfg = load_rl_cfg(pair.task_id)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(pair.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), device=cfg.device)
  runner.load(
    str(pair.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=cfg.device,
  )
  policy = runner.get_inference_policy(device=cfg.device)

  robot = env.scene["robot"]
  spine_ids = robot.find_joints(("body_pitch_joint",), preserve_order=True)[0]
  spine_ids = torch.as_tensor(spine_ids, dtype=torch.long, device=env.device)
  spine_id_set = set(spine_ids.tolist())
  leg_ids = torch.tensor(
    [i for i in range(robot.num_joints) if i not in spine_id_set],
    dtype=torch.long,
    device=env.device,
  )
  dt = float(env.step_dt)
  mass = float(env.sim.mj_model.body_mass.sum())
  gravity = float(
    torch.linalg.vector_norm(torch.as_tensor(env.sim.mj_model.opt.gravity))
  )
  gravity = cfg.gravity if cfg.gravity > 0 else gravity

  observations = vec_env.get_observations()
  total_power: list[torch.Tensor] = []
  leg_power: list[torch.Tensor] = []
  spine_power: list[torch.Tensor] = []
  spine_positive: list[torch.Tensor] = []
  spine_absorbed: list[torch.Tensor] = []
  speeds: list[torch.Tensor] = []
  spine_q: list[torch.Tensor] = []
  done_count = 0
  with torch.inference_mode():
    for _ in range(cfg.steps):
      actions = policy(observations)
      observations, _, dones, _ = vec_env.step(actions)
      active = ~dones.bool()
      done_count += int(dones.sum().item())
      if not active.any():
        continue
      torque = robot.data.qfrc_actuator
      velocity = robot.data.joint_vel
      signed_power = torque * velocity
      absolute_power = signed_power.abs()
      total_power.append(absolute_power[active].sum(dim=-1))
      leg_power.append(absolute_power[active][:, leg_ids].sum(dim=-1))
      spine_power.append(absolute_power[active][:, spine_ids].sum(dim=-1))
      spine_positive.append(signed_power[active][:, spine_ids].clamp_min(0).sum(dim=-1))
      spine_absorbed.append(
        (-signed_power[active][:, spine_ids]).clamp_min(0).sum(dim=-1)
      )
      speeds.append(robot.data.root_link_lin_vel_w[active, 0])
      spine_q.append(robot.data.joint_pos[active][:, spine_ids].abs().sum(dim=-1))

  total = _mean_active(total_power)
  speed = _mean_active(speeds)
  energy_per_meter = total / speed if speed > 1.0e-6 else float("nan")
  result: dict[str, Any] = {
    "label": pair.label,
    "task_id": pair.task_id,
    "checkpoint": str(pair.checkpoint),
    "motion_file": str(pair.motion_file) if pair.motion_file else motion.motion_file,
    "mass_kg": mass,
    "gravity_mps2": gravity,
    "num_envs": cfg.num_envs,
    "steps": cfg.steps,
    "dt_s": dt,
    "duration_s": cfg.steps * dt,
    "total_abs_power_w": total,
    "leg_abs_power_w": _mean_active(leg_power),
    "spine_abs_power_w": _mean_active(spine_power),
    "spine_positive_power_w": _mean_active(spine_positive),
    "spine_absorbed_power_w": _mean_active(spine_absorbed),
    "spine_power_fraction": _mean_active(spine_power) / total,
    "energy_playback_j": total * cfg.steps * dt,
    "energy_6s_j": total * 6.0,
    "energy_10s_equivalent_j": total * 10.0,
    "forward_speed_mps": speed,
    "spine_abs_q_mean_rad": _mean_active(spine_q),
    "energy_per_meter_j_m": energy_per_meter,
    "cost_of_transport": energy_per_meter / (mass * gravity),
    "done_count": done_count,
  }
  series = {
    "label": pair.label,
    "time_s": (np.arange(len(total_power), dtype=np.float64) * dt).tolist(),
    "total_power_w": [x.mean().item() for x in total_power],
    "leg_power_w": [x.mean().item() for x in leg_power],
    "spine_power_w": [x.mean().item() for x in spine_power],
    "spine_positive_power_w": [x.mean().item() for x in spine_positive],
    "spine_absorbed_power_w": [x.mean().item() for x in spine_absorbed],
    "speed_mps": [x.mean().item() for x in speeds],
  }
  vec_env.close()
  return result, series


def _run(cfg: Config) -> None:
  if warp.config.kernel_cache_dir is None:
    warp.config.kernel_cache_dir = "/tmp/mjlab-warp"
  if not cfg.pair:
    raise ValueError("Provide at least one --pair")
  if cfg.num_envs < 1 or cfg.steps < 1:
    raise ValueError("num_envs and steps must be positive")
  pair_specs = [spec.strip() for spec in cfg.pair.split(";") if spec.strip()]
  pairs = [_parse_pair(spec) for spec in pair_specs]
  cfg.output_dir.mkdir(parents=True, exist_ok=True)

  results = []
  series = []
  for pair in pairs:
    result, item_series = _evaluate(pair, cfg)
    results.append(result)
    series.append(item_series)
    print(json.dumps(result, indent=2))

  (cfg.output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
  with (cfg.output_dir / "metrics.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
  (cfg.output_dir / "timeseries.json").write_text(json.dumps(series, indent=2))
  _plot_summary(results, cfg.output_dir / "summary.png")
  _plot_timeseries(series, cfg.output_dir / "timeseries.png")
  print(f"Saved metrics and plots to {cfg.output_dir}")


def main(
  pair: str,
  output_dir: Path = Path("logs/energy_compare"),
  num_envs: int = 32,
  steps: int = 300,
  device: str = "cpu",
  seed: int | None = None,
  gravity: float = _DEFAULT_GRAVITY,
) -> None:
  _run(
    Config(
      pair=pair,
      output_dir=output_dir,
      num_envs=num_envs,
      steps=steps,
      device=device,
      seed=seed,
      gravity=gravity,
    )
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
