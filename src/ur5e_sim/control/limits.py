"""Shared simulator motion limits for config loading and server requests."""
import math
from numbers import Real

DEFAULT_CARTESIAN_SPEED_M_S = .005
MAX_CARTESIAN_SPEED_M_S = .020


def cartesian_speed(value, *, field='Cartesian speed'):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 < value <= MAX_CARTESIAN_SPEED_M_S):
        raise ValueError(f'{field} must be a finite number in (0, {MAX_CARTESIAN_SPEED_M_S:g}] m/s; got {value!r}')
    return float(value)
