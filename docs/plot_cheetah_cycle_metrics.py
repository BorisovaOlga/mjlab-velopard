"""Collect and plot Cheetah actuator metrics over one averaged gait cycle."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import torch

import mjlab.tasks  # noqa: F401  # Populate the task registry.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Velocity-Flat-Cheetah"
DEFAULT_GAIT_PERIOD = 0.6
MASS = 5.424414725317205
GRAVITY = 9.81
PHASE_ORIGIN = 0.14
STAGES = (
  (0.00, 0.14, "Контакт FL", "#b8d8f0"),
  (0.14, 0.28, "Контакт RR", "#b8e0b8"),
  (0.28, 0.40, "Собранный полёт", "#f7d794"),
  (0.40, 0.54, "Контакт FR", "#a9c9e8"),
  (0.54, 0.68, "Контакт RL", "#9dce9d"),
  (0.68, 1.00, "Растянутый полёт", "#d5c4e8"),
)
FOOT_LABELS = ("FL", "FR", "RL", "RR")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path, help="Path to model_*.pt")
  parser.add_argument(
    "--task-id",
    default=TASK_ID,
    help="Environment task used by the checkpoint",
  )
  parser.add_argument("--cycles", type=int, default=20)
  parser.add_argument(
    "--duration", type=float, default=10.0, help="Measurement duration in seconds"
  )
  parser.add_argument("--warmup-cycles", type=int, default=5)
  parser.add_argument("--bins", type=int, default=20)
  parser.add_argument(
    "--phase-source",
    choices=("contact", "clock"),
    default="contact",
    help="Align cycles to FL touchdowns (contact) or to the configured clock",
  )
  parser.add_argument(
    "--gait-period",
    type=float,
    default=DEFAULT_GAIT_PERIOD,
    help="Gait-cycle period in seconds (must match the training config)",
  )
  parser.add_argument("--device", default=None)
  parser.add_argument("--video-width", type=int, default=1920)
  parser.add_argument("--video-height", type=int, default=1080)
  parser.add_argument("--camera-azimuth", type=float, default=None)
  parser.add_argument("--camera-elevation", type=float, default=None)
  parser.add_argument(
    "--video",
    action="store_true",
    help="Save rollout.mp4 with one frame per measurement sample (clock only)",
  )
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


def align_to_fl_touchdowns(
  data: dict[str, np.ndarray | tuple[str, ...]],
  *,
  step_dt: float,
  nominal_period: float,
  cycles: int,
  bins: int,
) -> dict[str, np.ndarray | tuple[str, ...]]:
  """Align every FL-to-FL cycle without selecting only successful sequences."""
  contacts = np.asarray(data["contact"])
  rising = np.zeros_like(contacts, dtype=bool)
  rising[1:] = contacts[1:].astype(bool) & ~contacts[:-1].astype(bool)

  # Suppress only contact chatter, independently for every foot. A 40 ms
  # refractory interval is short enough to retain the observed 0.14 s cadence.
  refractory_steps = max(1, round(0.04 / step_dt))
  accepted_events: list[tuple[int, int]] = []
  last_event = np.full(len(FOOT_LABELS), -refractory_steps, dtype=int)
  for step, events in enumerate(rising):
    for foot in np.flatnonzero(events):
      if step - last_event[foot] >= refractory_steps:
        accepted_events.append((step, int(foot)))
        last_event[foot] = step

  fl_steps = [step for step, foot in accepted_events if foot == 0]
  cycle_bounds = [
    (start, end)
    for start, end in zip(fl_steps, fl_steps[1:], strict=False)
    if 0.04 <= (end - start) * step_dt <= 2.0 * nominal_period
  ]
  if not cycle_bounds:
    raise RuntimeError(
      "Could not detect FL-touchdown cycles. Try more --cycles or use "
      "--phase-source clock."
    )
  cycle_bounds = cycle_bounds[-cycles:]

  desired_order = (0, 3, 1, 2)  # FL, RR, FR, RL.
  correct_cycles: list[bool] = []
  for start, end in cycle_bounds:
    first_occurrence: list[int] = [0]
    for step, foot in accepted_events:
      if not start < step < end or foot in first_occurrence:
        continue
      first_occurrence.append(foot)
    correct_cycles.append(tuple(first_occurrence) == desired_order)

  transition_matrix = np.zeros((4, 4), dtype=int)
  for (_, previous), (_, current) in zip(
    accepted_events, accepted_events[1:], strict=False
  ):
    if previous != current:
      transition_matrix[previous, current] += 1

  flight_durations: list[float] = []
  airborne = contacts.sum(axis=1) == 0
  run_start: int | None = None
  for step, is_airborne in enumerate(np.r_[airborne, False]):
    if is_airborne and run_start is None:
      run_start = step
    elif not is_airborne and run_start is not None:
      flight_durations.append((step - run_start) * step_dt)
      run_start = None

  aligned = dict(data)
  target_phase = (np.arange(bins) + 0.5) / bins
  aligned["phase"] = np.tile(target_phase, len(cycle_bounds))
  for key in (
    "torque",
    "velocity",
    "power",
    "cot",
    "contact",
    "forward_speed",
    "lateral_speed",
  ):
    source = np.asarray(data[key])
    resampled_cycles: list[np.ndarray] = []
    for start, end in cycle_bounds:
      values = source[start:end]
      source_phase = (np.arange(len(values)) + 0.5) / len(values)
      flat_values = values.reshape(len(values), -1)
      if key == "contact":
        nearest = np.clip(
          np.searchsorted(source_phase, target_phase), 0, len(values) - 1
        )
        resampled = flat_values[nearest]
      else:
        resampled = np.stack(
          [np.interp(target_phase, source_phase, column) for column in flat_values.T],
          axis=1,
        )
      resampled_cycles.append(resampled.reshape(bins, *values.shape[1:]))
    aligned[key] = np.concatenate(resampled_cycles)
  durations = [(end - start) * step_dt for start, end in cycle_bounds]
  aligned["cycle_durations"] = np.asarray(durations)
  aligned["flight_durations"] = np.asarray(flight_durations)
  aligned["transition_matrix"] = transition_matrix
  aligned["correct_cycles"] = np.asarray(correct_cycles)
  print(
    f"Detected unbiased FL cycles: {len(cycle_bounds)}, mean period: "
    f"{np.mean(durations):.3f} s, range: {min(durations):.3f}-{max(durations):.3f} s"
  )
  print(
    "Correct FL→RR→FR→RL cycles: "
    f"{sum(correct_cycles)}/{len(correct_cycles)} ({np.mean(correct_cycles):.1%})"
  )
  if flight_durations:
    print(
      f"Full-flight intervals: {len(flight_durations)}, mean "
      f"{np.mean(flight_durations):.3f} s, max {max(flight_durations):.3f} s"
    )
  else:
    print("Full-flight intervals: 0")
  return aligned


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

  env_cfg = load_env_cfg(args.task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.terminations = {}
  if not any(s.name == "feet_ground_contact" for s in env_cfg.scene.sensors):
    # Tracking tasks do not need this sensor, but the report needs one channel
    # per foot in FL, FR, RL, RR order, matching the velocity task.
    env_cfg.scene.sensors = (
      *env_cfg.scene.sensors,
      ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
          mode="geom",
          pattern=(
            "left_front_knee_pitch_link_collision_2",
            "right_front_knee_pitch_link_collision_2",
            "left_knee_pitch_link_collision_2",
            "right_knee_pitch_link_collision_2",
          ),
          entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
      ),
    )
  agent_cfg = load_rl_cfg(args.task_id)
  if args.video:
    env_cfg.viewer.width = args.video_width
    env_cfg.viewer.height = args.video_height
    if args.camera_azimuth is not None:
      env_cfg.viewer.azimuth = args.camera_azimuth
    if args.camera_elevation is not None:
      env_cfg.viewer.elevation = args.camera_elevation
  base_env = ManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode="rgb_array" if args.video else None
  )
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)
  robot = env.unwrapped.scene["robot"]
  feet_contact_sensor = env.unwrapped.scene["feet_ground_contact"]
  print("Contact sensor channels:")
  for index, (label, primary_name) in enumerate(
    zip(FOOT_LABELS, feet_contact_sensor.primary_names, strict=True)
  ):
    print(f"  {index}: {label} = {primary_name}")
  joint_names = robot.joint_names

  steps_per_cycle = round(args.gait_period / env.unwrapped.step_dt)
  warmup_steps = args.warmup_cycles * steps_per_cycle
  # Extra nominal cycles give touchdown alignment enough complete edge-to-edge
  # intervals even when the learned cadence differs from the training clock.
  collection_steps = round(args.duration / env.unwrapped.step_dt)
  phases: list[float] = []
  times: list[float] = []
  torques: list[np.ndarray] = []
  velocities: list[np.ndarray] = []
  powers: list[np.ndarray] = []
  positions: list[np.ndarray] = []
  com_positions: list[np.ndarray] = []
  com_velocities: list[np.ndarray] = []
  mechanical_powers: list[np.ndarray] = []
  energy_trace: list[float] = []
  cot: list[float] = []
  contacts: list[np.ndarray] = []
  forward_speeds: list[float] = []
  lateral_speeds: list[float] = []
  total_energy = 0.0
  total_distance = 0.0
  video_stack = ExitStack()

  obs = env.get_observations()
  try:
    writer = (
      video_stack.enter_context(
        media.VideoWriter(
          args.output_dir / "rollout.mp4",
          shape=(args.video_height, args.video_width),
          fps=1.0 / base_env.step_dt,
          crf=16,
          ffmpeg_args=("-preset", "slow", "-movflags", "+faststart"),
        )
      )
      if args.video
      else None
    )
    for step in range(warmup_steps + collection_steps):
      with torch.no_grad():
        actions = policy(obs)
      obs, _, _, _ = env.step(actions)
      if step < warmup_steps:
        continue
      if args.video:
        frame = base_env.render()
        assert frame is not None
        assert writer is not None
        writer.add_image(frame)

      torque = robot.data.qfrc_actuator[0]
      velocity = robot.data.joint_vel[0]
      joint_power = torch.abs(torque * velocity)
      speed = torch.clamp(torch.abs(robot.data.root_link_lin_vel_b[0, 0]), min=0.25)
      dt = env.unwrapped.step_dt
      total_energy += float(joint_power.sum().cpu()) * dt
      total_distance += abs(float(robot.data.root_link_lin_vel_b[0, 0].cpu())) * dt
      original_phase = (
        env.episode_length_buf[0] * env.unwrapped.step_dt / args.gait_period
      ) % 1.0
      shifted_phase = (original_phase - PHASE_ORIGIN) % 1.0

      phases.append(float(shifted_phase.cpu()))
      times.append((step - warmup_steps) * env.unwrapped.step_dt)
      torques.append(torque.detach().cpu().numpy().copy())
      velocities.append(velocity.detach().cpu().numpy().copy())
      powers.append(joint_power.detach().cpu().numpy().copy())
      positions.append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
      mechanical_powers.append((torque * velocity).detach().cpu().numpy().copy())
      com_positions.append(robot.data.root_link_pos_w[0].detach().cpu().numpy().copy())
      com_velocities.append(
        robot.data.root_link_lin_vel_w[0].detach().cpu().numpy().copy()
      )
      energy_trace.append(total_energy)
      cot.append(float((joint_power.sum() / (MASS * GRAVITY * speed)).cpu()))
      assert feet_contact_sensor.data.found is not None
      contacts.append(
        (feet_contact_sensor.data.found[0].reshape(-1) > 0).float().cpu().numpy().copy()
      )
      forward_speeds.append(float(robot.data.root_link_lin_vel_b[0, 0].cpu()))
      lateral_speeds.append(float(robot.data.root_link_lin_vel_b[0, 1].cpu()))
  finally:
    video_stack.close()
    env.close()

  data: dict[str, np.ndarray | tuple[str, ...]] = {
    "phase": np.asarray(phases),
    "time": np.asarray(times),
    "torque": np.asarray(torques),
    "velocity": np.asarray(velocities),
    "power": np.asarray(powers),
    "position": np.asarray(positions),
    "mechanical_power": np.asarray(mechanical_powers),
    "com_position": np.asarray(com_positions),
    "com_velocity": np.asarray(com_velocities),
    "energy_trace": np.asarray(energy_trace),
    "cot": np.asarray(cot),
    "contact": np.asarray(contacts),
    "forward_speed": np.asarray(forward_speeds),
    "lateral_speed": np.asarray(lateral_speeds),
    "joint_names": joint_names,
    "total_energy": total_energy,
    "total_distance": total_distance,
    "measurement_duration": args.duration,
  }
  if args.phase_source == "contact":
    return align_to_fl_touchdowns(
      data,
      step_dt=env.unwrapped.step_dt,
      nominal_period=args.gait_period,
      cycles=args.cycles,
      bins=args.bins,
    )
  return data


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


def print_cot_summary(data: dict) -> None:
  """Print the rollout mean CoT and the mean for each gait stage."""
  phase = data["phase"]
  cot = data["cot"]
  print(f"Mean mechanical Cost of Transport: {np.mean(cot):.4f}")
  for start, end, label, _ in STAGES:
    mask = (phase >= start) & (phase < end)
    stage_mean = np.mean(cot[mask]) if np.any(mask) else np.nan
    print(f"  {label} ({start:.0%}-{end:.0%}): {stage_mean:.4f}")


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
    for index in indices:
      ax.plot(x, averaged[:, index], label=short_name(names[index]), linewidth=1.7)
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


def plot_foot_contacts(data: dict, bins: int, output_dir: Path) -> None:
  """Plot contact probability of every foot over the averaged gait cycle."""
  x, contacts = phase_average(data["phase"], data["contact"], bins)
  fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
  colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
  for foot_index, (ax, label, color) in enumerate(
    zip(axes, FOOT_LABELS, colors, strict=True)
  ):
    add_stage_background(ax)
    ax.step(x, contacts[:, foot_index], where="mid", color=color, linewidth=2.2)
    ax.fill_between(
      x, 0.0, contacts[:, foot_index], step="mid", color=color, alpha=0.25
    )
    ax.set_ylabel(label)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks((0.0, 1.0), ("воздух", "контакт"))
  axes[0].set_title("Контакты лап внутри усреднённого цикла")
  axes[-1].set_xlabel("Фаза цикла, %")
  fig.savefig(output_dir / "cycle_foot_contacts.png", dpi=180)
  plt.close(fig)


def plot_cycle_diagnostics(data: dict, output_dir: Path) -> None:
  """Plot unbiased cadence, flight duration and touchdown-transition diagnostics."""
  if "cycle_durations" not in data:
    return
  durations = data["cycle_durations"]
  flights = data["flight_durations"]
  matrix = data["transition_matrix"]

  fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
  ax.hist(durations, bins=min(15, max(3, len(durations))), color="#4c78a8")
  ax.axvspan(0.30, 0.40, color="#59a14f", alpha=0.18, label="цель 0.30–0.40 с")
  ax.set(title="Распределение периода FL→FL", xlabel="Период, с", ylabel="Циклы")
  ax.legend()
  ax.grid(alpha=0.25)
  fig.savefig(output_dir / "cycle_period_distribution.png", dpi=180)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
  if len(flights):
    ax.hist(flights, bins=min(15, max(3, len(flights))), color="#f28e2b")
  else:
    ax.text(0.5, 0.5, "Полный полёт не обнаружен", ha="center", va="center")
  ax.axvline(0.04, color="#e15759", linestyle="--", label="минимальная цель 0.04 с")
  ax.set(title="Длительность полного полёта", xlabel="Время, с", ylabel="Интервалы")
  ax.legend()
  ax.grid(alpha=0.25)
  fig.savefig(output_dir / "flight_duration_distribution.png", dpi=180)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
  image = ax.imshow(matrix, cmap="Blues")
  for row in range(4):
    for column in range(4):
      ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
  ax.set_xticks(range(4), FOOT_LABELS)
  ax.set_yticks(range(4), FOOT_LABELS)
  ax.set(
    xlabel="Следующее касание",
    ylabel="Предыдущее касание",
    title="Матрица переходов касаний",
  )
  fig.colorbar(image, ax=ax, label="Количество")
  fig.savefig(output_dir / "touchdown_transition_matrix.png", dpi=180)
  plt.close(fig)


def plot_body_velocity(data: dict, bins: int, output_dir: Path) -> None:
  x, forward = phase_average(data["phase"], data["forward_speed"][:, None], bins)
  _, lateral = phase_average(data["phase"], data["lateral_speed"][:, None], bins)
  fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
  add_stage_background(ax)
  ax.plot(x, forward[:, 0], label="вперёд", linewidth=2)
  ax.plot(x, lateral[:, 0], label="вбок", linewidth=2)
  ax.axhline(0.0, color="0.3", linewidth=0.8)
  ax.set(title="Скорость корпуса внутри цикла", xlabel="Фаза цикла, %", ylabel="м/с")
  ax.legend()
  fig.savefig(output_dir / "cycle_body_velocity.png", dpi=180)
  plt.close(fig)


def main() -> None:
  args = parse_args()
  if args.video and any(
    size <= 0 or size % 2 for size in (args.video_width, args.video_height)
  ):
    raise ValueError("Video width and height must be positive even numbers.")
  if args.video and args.phase_source != "clock":
    raise ValueError("Use --phase-source clock to synchronize video and samples.")
  if args.duration <= 0:
    raise ValueError("duration must be positive")
  if (
    args.cycles < 1 or args.warmup_cycles < 0 or args.bins < 4 or args.gait_period <= 0
  ):
    raise ValueError(
      "cycles >= 1, warmup-cycles >= 0, bins >= 4 and gait-period > 0 are required."
    )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  data = collect_rollout(args)
  np.savez(
    args.output_dir / "rollout_data.npz",
    phase=data["phase"],
    torque=data["torque"],
    velocity=data["velocity"],
    time=data["time"],
    power=data["power"],
    mechanical_power=data["mechanical_power"],
    position=data["position"],
    energy_trace=data["energy_trace"],
    com_position=data["com_position"],
    com_velocity=data["com_velocity"],
    contact=data["contact"],
    forward_speed=data["forward_speed"],
    lateral_speed=data["lateral_speed"],
    joint_names=np.asarray(data["joint_names"], dtype=str),
    total_energy=data.get("total_energy", np.nan),
    total_distance=data.get("total_distance", np.nan),
    measurement_duration=args.duration,
    training_path=str(args.checkpoint.parent),
    checkpoint=str(args.checkpoint),
  )
  energy = data["total_energy"]
  distance = data["total_distance"]
  cot_scalar = energy / (MASS * GRAVITY * max(distance, 1e-9))
  print(f"Measurement duration: {args.duration:.3f} s")
  print(f"Mechanical energy: {energy:.6f} J")
  print(f"Distance: {distance:.6f} m")
  print(f"Mean speed: {distance / args.duration:.6f} m/s")
  print(f"Mechanical Cost of Transport: {cot_scalar:.6f}")
  print_cot_summary(data)
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
  plot_foot_contacts(data, args.bins, args.output_dir)
  plot_body_velocity(data, args.bins, args.output_dir)
  plot_cycle_diagnostics(data, args.output_dir)
  for path in sorted(args.output_dir.glob("*.png")):
    print(path)


if __name__ == "__main__":
  main()
