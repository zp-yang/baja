"""
JAX EKF/UKF Toolbox
===================

A modern Python library for nonlinear Kalman filtering and smoothing, migrated from the 
MATLAB `ekfukf` toolbox. Built on JAX and Equinox for JIT compilation, 
automatic differentiation, and hardware acceleration.

Features:
- Kalman Filter (KF)
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Cubature Kalman Filter (CKF)
- Gauss-Hermite Kalman Filter (GHKF)
- Particle Filter (PF) (Bootstrap Filter)
- Particle Flow Filters: Exact Daum-Huang (EDH) & Localized EDH (LEDH)
- Interacting Multiple Model (IMM) Filter & Smoother
- Forward filtering and backward RTS smoothing
- Fast sequence processing via `jax.lax.scan`

For a quickstart, see the documentation in `README.md`.
"""

from .base import AbstractFilter, GaussianState
from .kf import KalmanFilter
from .ekf import ExtendedKalmanFilter
from .ukf import UnscentedKalmanFilter
from .ckf import CubatureKalmanFilter
from .ghkf import GaussHermiteKalmanFilter
from .imm import IMMState, IMMFilter
from .pff import EDHFilter, LEDHKNNFilter, LEDHKDEFilter, LEDHGMMFilter
from .pf import ParticleState, ParticleFilter
from .utils import lti_disc, rk4
from .transforms import compute_ut_weights, generate_sigmas, unscented_transform

__all__ = [
    "GaussianState",
    "AbstractFilter",
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "CubatureKalmanFilter",
    "GaussHermiteKalmanFilter",
    "IMMState",
    "IMMFilter",
    "ParticleState",
    "ParticleFilter",
    "EDHFilter",
    "LEDHKNNFilter",
    "LEDHKDEFilter",
    "LEDHGMMFilter",
    "lti_disc",
    "rk4",
    "compute_ut_weights",
    "generate_sigmas",
    "unscented_transform"
]
