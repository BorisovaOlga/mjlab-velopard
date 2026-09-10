"""Reference-motion loading and phase sampling utilities."""

from pathlib import Path
import numpy as np
import torch


class ReferenceMotion:
  """Load a normalized gait cycle and sample it for batched environments."""

  def __init__(self, path: str | Path, device: str | torch.device = "cpu"):
    self.path = Path(path)
    if not self.path.exists():
      raise FileNotFoundError(f"Reference motion not found: {self.path}")
    data = np.load(self.path)
    if "time" not in data:
      raise ValueError("Reference file must contain a 'time' array")
    self.device = torch.device(device)
    time = np.asarray(data["time"], dtype=np.float32)
    if time.ndim != 1 or len(time) < 2:
      raise ValueError("Reference time must be a 1-D array with at least 2 samples")
    self.phase = torch.linspace(0.0, 1.0, len(time), device=self.device)
    self.duration = float(time[-1] - time[0])
    self.values: dict[str, torch.Tensor] = {}
    for name in data.files:
      if name == "time":
        continue
      array = np.asarray(data[name], dtype=np.float32).squeeze()
      if array.ndim != 1 or len(array) != len(time):
        continue
      self.values[name] = torch.as_tensor(array, device=self.device)

  def sample(self, phase: torch.Tensor, name: str) -> torch.Tensor:
    """Linearly interpolate one signal at phases in [0, 1)."""
    if name not in self.values:
      raise KeyError(f"Signal '{name}' is not present in {self.path}")
    p = torch.remainder(phase, 1.0).to(self.device)
    position = p * (len(self.phase) - 1)
    left = torch.floor(position).long()
    right = torch.clamp(left + 1, max=len(self.phase) - 1)
    alpha = position - left.float()
    signal = self.values[name]
    return signal[left] * (1.0 - alpha) + signal[right] * alpha

  def sample_many(self, phase: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
    """Return sampled signals stacked in the requested order."""
    return torch.stack([self.sample(phase, name) for name in names], dim=-1)
