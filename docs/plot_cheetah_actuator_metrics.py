"""Plot Cheetah energy and actuator metrics from a TensorBoard run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run", type=Path, help="Training run directory")
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("docs/assets/actuator_metrics"),
    help="Directory for generated PNG files",
  )
  return parser.parse_args()


def load_scalars(run: Path) -> dict[str, list[tuple[int, float]]]:
  accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
  accumulator.Reload()
  return {
    tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
    for tag in accumulator.Tags()["scalars"]
  }


def short_joint_name(tag: str) -> str:
  return tag.rsplit("/", maxsplit=1)[-1].removesuffix("_joint")


def plot_cost_of_transport(
  scalars: dict[str, list[tuple[int, float]]], output_dir: Path
) -> None:
  tag = "Episode_Metrics/mechanical_cost_of_transport"
  if tag not in scalars:
    raise KeyError(f"Metric '{tag}' is absent; start training with the new config.")
  points = scalars[tag]
  fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
  ax.plot([p[0] for p in points], [p[1] for p in points], color="#2678b2")
  ax.set(title="Mechanical Cost of Transport", xlabel="Iteration", ylabel="CoT")
  ax.grid(alpha=0.3)
  fig.savefig(output_dir / "cost_of_transport.png", dpi=180)
  plt.close(fig)


def plot_actuators(
  scalars: dict[str, list[tuple[int, float]]], output_dir: Path
) -> None:
  torque_prefix = "Episode_Metrics/actuator_torque_abs/"
  velocity_prefix = "Episode_Metrics/actuator_velocity_abs/"
  torque_tags = sorted(tag for tag in scalars if tag.startswith(torque_prefix))
  velocity_tags = sorted(tag for tag in scalars if tag.startswith(velocity_prefix))
  if not torque_tags or not velocity_tags:
    raise KeyError("Actuator metrics are absent; start training with the new config.")

  fig, (torque_ax, velocity_ax) = plt.subplots(
    2, 1, figsize=(14, 10), sharex=True, constrained_layout=True
  )
  for tag in torque_tags:
    points = scalars[tag]
    torque_ax.plot(
      [p[0] for p in points],
      [p[1] for p in points],
      label=short_joint_name(tag),
      linewidth=1.3,
    )
  for tag in velocity_tags:
    points = scalars[tag]
    velocity_ax.plot(
      [p[0] for p in points],
      [p[1] for p in points],
      label=short_joint_name(tag),
      linewidth=1.3,
    )

  torque_ax.set(title="Mean absolute actuator torque", ylabel="Torque, N·m")
  velocity_ax.set(
    title="Mean absolute actuator velocity",
    xlabel="Iteration",
    ylabel="Angular velocity, rad/s",
  )
  for ax in (torque_ax, velocity_ax):
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
  fig.savefig(output_dir / "actuator_torque_velocity.png", dpi=180)
  plt.close(fig)


def main() -> None:
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  scalars = load_scalars(args.run)
  plot_cost_of_transport(scalars, args.output_dir)
  plot_actuators(scalars, args.output_dir)
  print(args.output_dir / "cost_of_transport.png")
  print(args.output_dir / "actuator_torque_velocity.png")


if __name__ == "__main__":
  main()
