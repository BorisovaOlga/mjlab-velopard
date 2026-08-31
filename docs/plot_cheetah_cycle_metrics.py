"""Collect and plot Cheetah actuator metrics over one averaged gait cycle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import mjlab.tasks  # noqa: F401  # Populate the task registry.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Velocity-Flat-Cheetah"
GAIT_PERIOD = 0.4
MASS = 5.424414725317205
GRAVITY = 9.81
PHASE_ORIGIN = 0.16
STAGES = (
  (0.00, 0.24, "Контакт передних лап", "#b8d8f0"),
  (0.24, 0.38, "Собранный полёт", "#f7d794"),
  (0.38, 0.64, "Контакт и толчок задних", "#b8e0b8"),
  (0.64, 1.00, "Растянутый полёт", "#d5c4e8"),
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path, help="Path to model_*.pt")
  parser.add_argument("--cycles", type=int, default=20)
  parser.add_argument("--warmup-cycles", type=int, default=5)
  parser.add_argument("--bins", type=int, default=20)
  parser.add_argument("--device", default=None)
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("docs/assets/cycle_metrics"),
  )
  return parser.parse_args()


def add_stage_background(ax: plt.Axes) -> None:
  for start, end, label, color in STAGES:
    ax.axvspan(start * 100, end * 100, color=color, alpha=0.38, label=label)
  for start, _, _, _ in STAGES[1:]:
    ax.axvline(start * 100, color="0.45", linewidth=0.8, linestyle="--")
  ax.set_xlim(0, 100)
  ax.grid(alpha=0.25)


def phase_average(
  phases: np.ndarray, values: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
  edges = np.linspace(0.0, 1.0, bins + 1)
  indices = np.clip(np.digitize(phases, edges) - 1, 0, bins - 1)
  averaged = np.full((bins, *values.shape[1:]), np.nan, dtype=np.float64)
  for index in range(bins):
    selected = values[indices == index]
    if len(selected):
      averaged[index] = selected.mean(axis=0)
  centers = 0.5 * (edges[:-1] + edges[1:]) * 100
  return centers, averaged


def short_name(name: str) -> str:
  replacements = {
    "left_front_": "FL_",
    "right_front_": "FR_",
    "left_": "RL_",
    "right_": "RR_",
    "body_pitch_joint": "spine",
    "_joint": "",
  }
  for old, new in replacements.items():
    name = name.replace(old, new)
  return name


def collect_rollout(
  args: argparse.Namespace,
) -> dict[str, np.ndarray | tuple[str, ...]]:
  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = args.checkpoint.resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.terminations = {}
  agent_cfg = load_rl_cfg(TASK_ID)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)
  robot = env.unwrapped.scene["robot"]
  joint_names = robot.joint_names

  steps_per_cycle = round(GAIT_PERIOD / env.unwrapped.step_dt)
  warmup_steps = args.warmup_cycles * steps_per_cycle
  collection_steps = args.cycles * steps_per_cycle
  phases: list[float] = []
  torques: list[np.ndarray] = []
  velocities: list[np.ndarray] = []
  powers: list[np.ndarray] = []
  cot: list[float] = []

  obs = env.get_observations()
  try:
    for step in range(warmup_steps + collection_steps):
      with torch.no_grad():
        actions = policy(obs)
      obs, _, _, _ = env.step(actions)
      if step < warmup_steps:
        continue

      torque = robot.data.qfrc_actuator[0]
      velocity = robot.data.joint_vel[0]
      joint_power = torch.abs(torque * velocity)
      speed = torch.clamp(torch.abs(robot.data.root_link_lin_vel_b[0, 0]), min=0.25)
      original_phase = (
        env.episode_length_buf[0] * env.unwrapped.step_dt / GAIT_PERIOD
      ) % 1.0
      shifted_phase = (original_phase - PHASE_ORIGIN) % 1.0

      phases.append(float(shifted_phase.cpu()))
      torques.append(torque.detach().cpu().numpy().copy())
      velocities.append(velocity.detach().cpu().numpy().copy())
      powers.append(joint_power.detach().cpu().numpy().copy())
      cot.append(float((joint_power.sum() / (MASS * GRAVITY * speed)).cpu()))
  finally:
    env.close()

  return {
    "phase": np.asarray(phases),
    "torque": np.asarray(torques),
    "velocity": np.asarray(velocities),
    "power": np.asarray(powers),
    "cot": np.asarray(cot),
    "joint_names": joint_names,
  }


def plot_cost_and_power(data: dict, bins: int, output_dir: Path) -> None:
  phase = data["phase"]
  x, cot = phase_average(phase, data["cot"][:, None], bins)
  _, power = phase_average(phase, data["power"], bins)
  fig, (cot_ax, power_ax) = plt.subplots(
    2, 1, figsize=(13, 8), sharex=True, constrained_layout=True
  )
  add_stage_background(cot_ax)
  add_stage_background(power_ax)
  cot_ax.plot(x, cot[:, 0], color="#1f77b4", linewidth=2.2)
  power_ax.plot(x, power.sum(axis=1), color="#c44e52", linewidth=2.2)
  cot_ax.set(title="Mechanical Cost of Transport внутри цикла", ylabel="CoT")
  power_ax.set(
    title="Суммарная абсолютная механическая мощность",
    xlabel="Фаза цикла, %",
    ylabel="Мощность, Вт",
  )
  cot_ax.legend(ncol=4, fontsize=8, loc="upper center")
  fig.savefig(output_dir / "cycle_cost_power.png", dpi=180)
  plt.close(fig)


def joint_groups(names: tuple[str, ...]) -> tuple[tuple[str, list[int]], ...]:
  return (
    ("Передние лапы", [i for i, name in enumerate(names) if "front" in name]),
    (
      "Задние лапы",
      [
        i
        for i, name in enumerate(names)
        if "front" not in name and name != "body_pitch_joint"
      ],
    ),
    ("Позвоночник", [i for i, name in enumerate(names) if name == "body_pitch_joint"]),
  )


def plot_joint_quantity(
  data: dict,
  key: str,
  ylabel: str,
  title: str,
  filename: str,
  bins: int,
  output_dir: Path,
) -> None:
  names = data["joint_names"]
  x, averaged = phase_average(data["phase"], data[key], bins)
  fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
  for ax, (group_title, indices) in zip(axes, joint_groups(names), strict=True):
    add_stage_background(ax)
    if indices:
      for index in indices:
        ax.plot(x, averaged[:, index], label=short_name(names[index]), linewidth=1.7)
    else:
      ax.axhline(0.0, color="black", linewidth=1.7, label="spine (fixed)")
    ax.set_title(group_title)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=3, fontsize=8, loc="upper center")
  axes[-1].set_xlabel("Фаза цикла, %")
  fig.suptitle(title, fontsize=15)
  fig.savefig(output_dir / filename, dpi=180)
  plt.close(fig)


def plot_left_right_power(data: dict, bins: int, output_dir: Path) -> None:
  names = data["joint_names"]
  x, power = phase_average(data["phase"], data["power"], bins)
  groups = {
    "Передняя левая": [i for i, n in enumerate(names) if n.startswith("left_front")],
    "Передняя правая": [i for i, n in enumerate(names) if n.startswith("right_front")],
    "Задняя левая": [
      i for i, n in enumerate(names) if n.startswith("left_") and "front" not in n
    ],
    "Задняя правая": [
      i for i, n in enumerate(names) if n.startswith("right_") and "front" not in n
    ],
  }
  fig, (front_ax, hind_ax) = plt.subplots(
    2, 1, figsize=(13, 8), sharex=True, constrained_layout=True
  )
  for ax in (front_ax, hind_ax):
    add_stage_background(ax)
    ax.set_ylabel("Мощность, Вт")
  for label in ("Передняя левая", "Передняя правая"):
    front_ax.plot(x, power[:, groups[label]].sum(axis=1), label=label, linewidth=2)
  for label in ("Задняя левая", "Задняя правая"):
    hind_ax.plot(x, power[:, groups[label]].sum(axis=1), label=label, linewidth=2)
  front_ax.set_title("Сравнение мощности передних лап")
  hind_ax.set_title("Сравнение мощности задних лап")
  hind_ax.set_xlabel("Фаза цикла, %")
  front_ax.legend()
  hind_ax.legend()
  fig.savefig(output_dir / "cycle_left_right_power.png", dpi=180)
  plt.close(fig)


def main() -> None:
  args = parse_args()
  if args.cycles < 1 or args.warmup_cycles < 0 or args.bins < 4:
    raise ValueError("cycles >= 1, warmup-cycles >= 0 and bins >= 4 are required.")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  data = collect_rollout(args)
  plot_cost_and_power(data, args.bins, args.output_dir)
  plot_joint_quantity(
    data,
    "torque",
    "Момент, Н·м",
    "Моменты приводов внутри цикла",
    "cycle_actuator_torques.png",
    args.bins,
    args.output_dir,
  )
  plot_joint_quantity(
    data,
    "velocity",
    "Скорость, рад/с",
    "Угловые скорости приводов внутри цикла",
    "cycle_actuator_velocities.png",
    args.bins,
    args.output_dir,
  )
  plot_left_right_power(data, args.bins, args.output_dir)
  for path in sorted(args.output_dir.glob("cycle_*.png")):
    print(path)


if __name__ == "__main__":
  main()
