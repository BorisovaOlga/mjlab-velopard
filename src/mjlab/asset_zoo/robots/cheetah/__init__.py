"""Cheetah quadruped."""

from .cheetah_constants import (
  CHEETAH_ACTION_SCALE as CHEETAH_ACTION_SCALE,
)
from .cheetah_constants import (
  CHEETAH_FIXED_SPINE_ACTION_SCALE as CHEETAH_FIXED_SPINE_ACTION_SCALE,
)
from .cheetah_constants import (
  get_cheetah_robot_cfg as get_cheetah_robot_cfg,
)
from .cheetah_constants import (
  get_fixed_spine_cheetah_robot_cfg as get_fixed_spine_cheetah_robot_cfg,
)

__all__ = [
  "CHEETAH_ACTION_SCALE",
  "CHEETAH_FIXED_SPINE_ACTION_SCALE",
  "get_cheetah_robot_cfg",
  "get_fixed_spine_cheetah_robot_cfg",
]
