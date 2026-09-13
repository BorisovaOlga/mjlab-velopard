"""Convert video-extracted trajectories to a reference motion NPZ.

This script expects an input NPZ with at least a 1-D `time` array and
either:
 - `feet` shape (T,4,3) with foot world positions in order FL,FR,RL,RR, and
   optionally `spine_flexion` or `body_pitch` (T,), or
 - precomputed 1-D signals named like `FL_x`, `FL_y`, etc.

Output is an NPZ suitable for `ReferenceMotion`/`ReferenceFeetTracking` and
`ReferenceSpineTracking` (keys: `time`, `spine_flexion`, `FL_x`, `FL_y`, ...).
"""
from pathlib import Path
import argparse
import numpy as np


def to_1d(arr):
  arr = np.asarray(arr)
  if arr.ndim == 1:
    return arr
  if arr.ndim > 1 and arr.shape[1] == 1:
    return arr.ravel()
  raise ValueError("Cannot coerce array to 1-D signal")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("input", type=Path)
  p.add_argument("output", type=Path)
  args = p.parse_args()

  data = np.load(args.input)
  if "time" not in data:
    raise SystemExit("Input must contain 'time' array")
  time = np.asarray(data["time"]).astype(np.float32)
  T = len(time)

  out = {"time": time}

  # Spine flexion: prefer explicit key, then body_pitch
  spine_keys = ["spine_flexion", "spine_pitch", "body_pitch"]
  for k in spine_keys:
    if k in data:
      out["spine_flexion"] = to_1d(data[k]).astype(np.float32)
      break

  # Feet: either provided as per-frame 3D positions or already as signals.
  foot_names = ("FL", "FR", "RL", "RR")
  if "feet" in data:
    feet = np.asarray(data["feet"])  # expect (T,4,3)
    if feet.ndim != 3 or feet.shape[1] != 4 or feet.shape[2] != 3:
      raise SystemExit("'feet' must have shape (T,4,3) in order FL,FR,RL,RR")
    # store sagittal (x) and vertical (z) signals for each foot
    for i, name in enumerate(foot_names):
      out[f"{name}_x"] = feet[:, i, 0].astype(np.float32)
      out[f"{name}_y"] = feet[:, i, 2].astype(np.float32)
  else:
    # look for individual FL_x etc.
    found = False
    for name in foot_names:
      if f"{name}_x" in data and f"{name}_y" in data:
        out[f"{name}_x"] = to_1d(data[f"{name}_x"]).astype(np.float32)
        out[f"{name}_y"] = to_1d(data[f"{name}_y"]).astype(np.float32)
        found = True
    if not found:
      # no foot signals found — emit zeros to avoid KeyError downstream
      for name in foot_names:
        out[f"{name}_x"] = np.zeros(T, dtype=np.float32)
        out[f"{name}_y"] = np.zeros(T, dtype=np.float32)

  # Relative chest coordinates: try to form `relative_chest_x/y` if chest and pelvis present
  if "chest_pos" in data and "pelvis_pos" in data:
    chest = np.asarray(data["chest_pos"]).astype(np.float32)
    pelvis = np.asarray(data["pelvis_pos"]).astype(np.float32)
    if chest.shape[0] == T and pelvis.shape[0] == T:
      rel = chest - pelvis
      out["relative_chest_x"] = rel[:, 0].astype(np.float32)
      out["relative_chest_y"] = rel[:, 2].astype(np.float32)

  # Ensure all signals are 1-D arrays of length T
  for k, v in list(out.items()):
    arr = np.asarray(v)
    if arr.ndim != 1 or len(arr) != T:
      raise SystemExit(f"Signal {k} has wrong shape {arr.shape}; expected ({T},)")

  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.output, **out)
  print(f"Saved reference motion: {args.output} (signals: {', '.join(sorted(out.keys()))})")


if __name__ == "__main__":
  main()
