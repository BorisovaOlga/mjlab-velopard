import numpy as np

AK40_TORQUE = np.array([
    0.0, 0.25, 0.45, 0.75, 1.2, 1.8, 2.25, 2.7, 3.15, 3.6, 4.05
])

AK40_EFFICIENCY = np.array([
    0.0, 0.68, 0.78, 0.80, 0.76, 0.70, 0.65, 0.58, 0.50, 0.42, 0.34
])

AK45_TORQUE = np.array([
    0.0, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.0, 6.0, 7.0
])

AK45_EFFICIENCY = np.array([
    0.0, 0.65, 0.70, 0.68, 0.62, 0.58, 0.54, 0.50, 0.46, 0.40
])


def efficiency(torque, actuator_type):
    torque = np.abs(torque)

    if actuator_type == "AK40-10":
        value = np.interp(torque, AK40_TORQUE, AK40_EFFICIENCY)
    elif actuator_type == "AK45-10":
        value = np.interp(torque, AK45_TORQUE, AK45_EFFICIENCY)
    else:
        raise ValueError(f"Unknown actuator type: {actuator_type}")

    return np.clip(value, 0.05, 1.0)