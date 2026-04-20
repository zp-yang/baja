import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Any, Callable
from .base import AbstractFilter, GaussianState

class ExtendedKalmanFilter(AbstractFilter):
    """
    Extended Kalman Filter (EKF).
    
    This implementation utilizes JAX auto-differentiation (`jax.jacfwd`) 
    to automatically compute the Jacobians $F_k$ and $H_k$, eliminating 
    the need to manually derive and provide them.
    """
    
    # Static parameters (can be overridden by params argument)
    f: Callable
    h: Callable
    Q: Optional[jax.Array] = None
    R: Optional[jax.Array] = None
    
    def __init__(self, f: Callable, h: Callable, Q: Optional[jax.Array] = None, R: Optional[jax.Array] = None):
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R

    @eqx.filter_jit
    def predict(
        self, state: GaussianState, u: Optional[jax.Array] = None, params: Any = None
    ) -> GaussianState:
        """
        EKF prediction step.
        """
        Q = params.Q if params is not None and hasattr(params, 'Q') else self.Q
        assert Q is not None, "Process noise covariance Q must be provided."
        
        # Evaluate the nonlinear transition function f(x, u)
        # We pass u as an argument to f if u is provided, else f(x)
        if u is not None and u.size > 0:
            m = self.f(state.mean, u)
            # Jacobian of f with respect to state (argnums=0)
            F = jax.jacfwd(self.f, argnums=0)(state.mean, u)
        else:
            m = self.f(state.mean)
            F = jax.jacfwd(self.f)(state.mean)
            
        P = F @ state.cov @ F.T + Q
        return GaussianState(mean=m, cov=P)

    @eqx.filter_jit
    def update(
        self, state: GaussianState, y: jax.Array, params: Any = None, return_likelihood: bool = False
    ) -> GaussianState:
        """
        EKF update step.
        """
        R = params.R if params is not None and hasattr(params, 'R') else self.R
        assert R is not None, "Measurement noise covariance R must be provided."
        
        # Evaluate the nonlinear measurement function h(x)
        z = self.h(state.mean)
        
        # Jacobian of h with respect to state
        H = jax.jacfwd(self.h)(state.mean)
        
        # Innovation (residual)
        v = y - z
        
        # Innovation covariance
        S = H @ state.cov @ H.T + R
        
        # Kalman gain (using solve to avoid explicit inversion, S * K^T = H * P => K = (P @ H.T) @ inv(S))
        K = jax.scipy.linalg.solve(S.T, H @ state.cov).T
        
        # Update mean and covariance
        mean = state.mean + K @ v
        # Joseph form for numerical stability
        I_KH = jnp.eye(state.cov.shape[0]) - K @ H
        cov = I_KH @ state.cov @ I_KH.T + K @ R @ K.T
        
        if return_likelihood:
            ll = jax.scipy.stats.multivariate_normal.logpdf(y, mean=z, cov=S)
            likelihood = jnp.exp(ll)
            return GaussianState(mean, cov), likelihood
        return GaussianState(mean, cov)

    @eqx.filter_jit
    def smooth_step(
        self, 
        state: GaussianState, 
        next_smoothed_state: GaussianState, 
        u: Optional[jax.Array] = None, 
        params: Any = None
    ) -> GaussianState:
        """
        EKF backward smoothing step (ERTS).
        """
        Q = params.Q if params is not None and hasattr(params, 'Q') else self.Q
        assert Q is not None, "Process noise covariance Q must be provided."
        
        # 1. Forward predict state
        if u is not None and u.size > 0:
            m_pred = self.f(state.mean, u)
            F = jax.jacfwd(self.f, argnums=0)(state.mean, u)
        else:
            m_pred = self.f(state.mean)
            F = jax.jacfwd(self.f)(state.mean)
            
        P_pred = F @ state.cov @ F.T + Q
        
        # 2. Smoother gain
        C = state.cov @ F.T
        D = jax.scipy.linalg.solve(P_pred.T, C.T).T
        
        # 3. Update mean and covariance
        mean = state.mean + D @ (next_smoothed_state.mean - m_pred)
        cov = state.cov + D @ (next_smoothed_state.cov - P_pred) @ D.T
        
        return GaussianState(mean=mean, cov=cov)
