# Flexible-spine Cheetah — evaluation report

## Checkpoint

- Run: `2026-09-10_14-15-33_straight_stable_from_10994`
- Checkpoint: `model_11293.pt`
- Task: `Mjlab-Velocity-Flat-Cheetah-Duration-Finetune`
- Measurement window: 10 s

## Kinematic and energetic results

| Metric | Value |
|---|---:|
| Correct FL → RR → FR → RL cycles | 30/30 (100%) |
| Mean cycle period | 0.221 s |
| Mean full-flight interval | 0.040 s |
| Maximum full-flight interval | 0.100 s |
| Mechanical energy | 1012.96 J |
| Distance | 39.71 m |
| Mean speed | 3.971 m/s |
| Mechanical CoT | 0.4794 |

The scalar CoT is calculated as `E / (m g D)`, with robot mass `5.424414725 kg`.

## Reproduce the plots

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python docs/plot_cheetah_cycle_metrics.py \
  logs/rsl_rl/cheetah_go2_baseline/2026-09-10_14-15-33_straight_stable_from_10994/model_11293.pt \
  --task-id Mjlab-Velocity-Flat-Cheetah-Duration-Finetune \
  --duration 10 \
  --cycles 30 \
  --output-dir docs/assets/flexible_spine_11293_metrics
```

The output directory contains plots of actuator torques, angular velocities,
power, contacts, body velocity, cycle timing and flight intervals.

## Video

Run the checkpoint with the standard viewer:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run play \
  Mjlab-Velocity-Flat-Cheetah-Duration-Finetune \
  --checkpoint-file logs/rsl_rl/cheetah_go2_baseline/2026-09-10_14-15-33_straight_stable_from_10994/model_11293.pt
```

Record the viewer output as `flexible_spine_11293.mp4` and place it beside
this report. The report should be compared with the fixed-spine experiment
using the same 10-second measurement window and actual mean speed.

## Interpretation

The checkpoint demonstrates the required footfall order and short flight
phases. The comparison with the fixed-spine model must use equal measured
speed; otherwise CoT values are not directly comparable.
