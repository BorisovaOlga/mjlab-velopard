"""Plot the target feline-gallop timing used by the Cheetah environment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PERIOD = 0.4
LEGS = (
  ("FL — левая передняя", 0.16, 0.34, "#2678b2"),
  ("FR — правая передняя", 0.22, 0.40, "#55a868"),
  ("RR — правая задняя", 0.54, 0.73, "#c44e52"),
  ("RL — левая задняя", 0.61, 0.80, "#8172b3"),
)
FLIGHTS = ((0.0, 0.16), (0.40, 0.54), (0.80, 1.0))


def swing_height(
  phase: np.ndarray, stance_start: float, stance_end: float
) -> np.ndarray:
  """Return a smooth normalized swing arc outside the stance interval."""
  swing_length = (stance_start + 1.0 - stance_end) % 1.0
  progress = (phase - stance_end) % 1.0
  in_swing = progress < swing_length
  height = np.zeros_like(phase)
  height[in_swing] = np.sin(np.pi * progress[in_swing] / swing_length)
  return height


def add_flight_shading(ax: plt.Axes) -> None:
  for index, (start, end) in enumerate(FLIGHTS):
    ax.axvspan(
      start * PERIOD,
      end * PERIOD,
      color="#f2c14e",
      alpha=0.22,
      label="фаза полёта" if index == 0 else None,
    )


def main() -> None:
  phase = np.linspace(0.0, 1.0, 1201)
  time = phase * PERIOD
  fig, (contact_ax, swing_ax) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
  )

  add_flight_shading(contact_ax)
  for row, (_name, start, end, color) in enumerate(LEGS):
    contact = ((phase >= start) & (phase < end)).astype(float)
    contact_ax.fill_between(
      time, row, row + 0.72 * contact, step="post", color=color, alpha=0.9
    )
  contact_ax.set_yticks(np.arange(len(LEGS)) + 0.36, [leg[0] for leg in LEGS])
  contact_ax.set_ylim(-0.08, len(LEGS) + 0.05)
  contact_ax.invert_yaxis()
  contact_ax.set_title("Целевая последовательность контактов лап")
  contact_ax.set_ylabel("Лапа")
  contact_ax.legend(loc="upper right")
  contact_ax.grid(axis="x", alpha=0.25)

  add_flight_shading(swing_ax)
  for name, start, end, color in LEGS:
    swing_ax.plot(time, swing_height(phase, start, end), label=name, color=color, lw=2)
  swing_ax.set_title("Условная высота лапы в фазе переноса")
  swing_ax.set_xlabel("Время внутри цикла, с")
  swing_ax.set_ylabel("Высота (нормированная)")
  swing_ax.set_xlim(0.0, PERIOD)
  swing_ax.set_ylim(-0.04, 1.08)
  swing_ax.grid(alpha=0.25)
  swing_ax.legend(ncol=2, loc="upper center")

  fig.suptitle(
    "Целевой галоп Cheetah: цикл 0.4 с, FL → FR → RR → RL",
    fontsize=15,
  )
  output = Path(__file__).with_name("assets") / "cheetah_gait_timing.png"
  fig.savefig(output, dpi=180)
  print(output)


if __name__ == "__main__":
  main()
