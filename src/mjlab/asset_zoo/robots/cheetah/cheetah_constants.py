"""Cheetah robot constants.

Uses the provided `cheetah.xml` MJCF and defines lightweight actuator
parameters. This exposes a simple spine actuator configuration in addition
to hip/knee actuators. Tune these values with the real hardware specs.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

CHEETAH_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "cheetah" / "xml" / "cheetah.xml"
)
assert CHEETAH_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(CHEETAH_XML))


##
# Actuator config (lightweight)
##

# Approximate rotor inertia (smaller than Go1 to reflect light design).
ROTOR_INERTIA = 5.0e-05

# Simple gearbox assumptions.
HIP_GEAR_RATIO = 6
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5
SPINE_GEAR_RATIO = 6

HIP_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIP_GEAR_RATIO),
  velocity_limit=40.0,
  effort_limit=12.0,
)
KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO),
  velocity_limit=30.0,
  effort_limit=18.0,
)
SPINE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, SPINE_GEAR_RATIO),
  velocity_limit=20.0,
  effort_limit=6.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP = HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_HIP = 2 * DAMPING_RATIO * HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_KNEE = KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_KNEE = 2 * DAMPING_RATIO * KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_SPINE = SPINE_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_SPINE = 2 * DAMPING_RATIO * SPINE_ACTUATOR.reflected_inertia * NATURAL_FREQ


CHEETAH_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*hip_pitch_joint", ".*hip_roll_joint"),
  stiffness=STIFFNESS_HIP,
  damping=DAMPING_HIP,
  effort_limit=HIP_ACTUATOR.effort_limit,
  armature=HIP_ACTUATOR.reflected_inertia,
)
CHEETAH_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*knee_pitch_joint",),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
)
CHEETAH_SPINE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("body_pitch_joint",),
  stiffness=STIFFNESS_SPINE,
  damping=DAMPING_SPINE,
  effort_limit=SPINE_ACTUATOR.effort_limit,
  armature=SPINE_ACTUATOR.reflected_inertia,
)

##
# Keyframes / initial state
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.20),
  joint_pos={
    ".*hip_pitch_joint": 0.0,
    ".*knee_pitch_joint": 0.0,
    "body_pitch_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config (reuse Go1-style selectors)
##

geom_names = (
  r"^left_front_.*_collision\d*$",
  r"^right_front_.*_collision\d*$",
  r"^left_(?!front).*_collision\d*$",
  r"^right_(?!front).*_collision\d*$",
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=geom_names,
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# Combined foot regex used by FULL_COLLISION mappings (matches front/hind left/right)
_foot_regex = (
  r"^(?:left_front_.*_collision\d*|right_front_.*_collision\d*|"
  r"left_(?!front).*_collision\d*|right_(?!front).*_collision\d*)$"
)

_collision_regex = r".*_collision\d*"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_collision_regex,),
  contype=1,
  conaffinity=1,
  solref=(0.01, 1),
  condim={_foot_regex: 6, _collision_regex: 1},
  priority={_foot_regex: 1, ".*": 0},
  friction={_foot_regex: (1, 5e-3, 5e-4)},
)


##
# Final config.
##

CHEETAH_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    CHEETAH_HIP_ACTUATOR_CFG,
    CHEETAH_KNEE_ACTUATOR_CFG,
    CHEETAH_SPINE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_cheetah_robot_cfg() -> EntityCfg:
  """Get a fresh Cheetah robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=CHEETAH_ARTICULATION,
  )


CHEETAH_ACTION_SCALE: dict[str, float] = {}
for a in CHEETAH_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    CHEETAH_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_cheetah_robot_cfg())

  viewer.launch(robot.spec.compile())
