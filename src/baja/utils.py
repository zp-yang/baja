import jax.numpy as jnp
import jax.scipy.linalg
from functools import partial
import numpy as np


def lti_disc(
    F: jax.Array, L: jax.Array, Qc: jax.Array, dt: float
) -> tuple[jax.Array, jax.Array]:
    """
    Discretize LTI ODE with Gaussian Noise.

    The original ODE model is:
        dx/dt = F x + L w,  w ~ N(0,Qc)

    Resulting discrete model is:
        x[k] = A x[k-1] + q, q ~ N(0,Q)

    Args:
        F: (N, N) Feedback matrix
        L: (N, L) Noise effect matrix
        Qc: (L, L) Diagonal Spectral Density
        dt: Time Step

    Returns:
        A: (N, N) Transition matrix
        Q: (N, N) Discrete Process Covariance
    """
    n = F.shape[0]

    # 1. Closed form integration of transition matrix
    A = jax.scipy.linalg.expm(F * dt)

    # 2. Closed form integration of covariance by matrix fraction decomposition
    # Phi = [F    L*Qc*L']
    #       [0    -F']
    Phi_top = jnp.hstack([F, L @ Qc @ L.T])
    Phi_bottom = jnp.hstack([jnp.zeros((n, n)), -F.T])
    Phi = jnp.vstack([Phi_top, Phi_bottom])

    AB = jax.scipy.linalg.expm(Phi * dt) @ jnp.vstack([jnp.zeros((n, n)), jnp.eye(n)])

    # Q = AB(1:n,:) / AB((n+1):(2*n),:)
    top_AB = AB[:n, :]
    bottom_AB = AB[n:, :]
    Q = jax.scipy.linalg.solve(bottom_AB.T, top_AB.T).T

    return A, Q


def rk4(f, dt: float, x: jax.Array, *args) -> jax.Array:
    """
    4th order Runge-Kutta integration.

    Args:
        f: function representing dx/dt = f(x, *args)
        dt: Delta time
        x: Current value of x
        *args: Additional arguments passed to f

    Returns:
        x_next: Next value of x after dt
    """
    k1 = f(x, *args) * dt
    k2 = f(x + 0.5 * k1, *args) * dt
    k3 = f(x + 0.5 * k2, *args) * dt
    k4 = f(x + k3, *args) * dt

    x_next = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return x_next
