"""Cheetah quadruped."""

from .cheetah_constants import (
  CHEETAH_ACTION_SCALE as CHEETAH_ACTION_SCALE,
)
from .cheetah_constants import (
  get_cheetah_robot_cfg as get_cheetah_robot_cfg,
)
from .cheetah_constants import (
  get_reborn_cheetah_robot_cfg as get_reborn_cheetah_robot_cfg,
)

__all__ = [
  "CHEETAH_ACTION_SCALE",
  "get_cheetah_robot_cfg",
  "get_reborn_cheetah_robot_cfg",
]
