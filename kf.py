import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Any
from .base import AbstractFilter, GaussianState

class KalmanFilter(AbstractFilter):
    """
    Standard Linear Kalman Filter.
    """
    # Define default static parameters (if available)
    A: Optional[jax.Array] = None
    B: Optional[jax.Array] = None
    H: Optional[jax.Array] = None
    Q: Optional[jax.Array] = None
    R: Optional[jax.Array] = None

    @eqx.filter_jit
    def predict(
        self, state: GaussianState, u: Optional[jax.Array] = None, params: Any = None
    ) -> GaussianState:
        """
        Kalman Filter prediction step.
        """
        # Resolve A, B, Q from params if provided, otherwise use class attributes
        A = params.A if params is not None and hasattr(params, 'A') else self.A
        B = params.B if params is not None and hasattr(params, 'B') else self.B
        Q = params.Q if params is not None and hasattr(params, 'Q') else self.Q
        
        assert A is not None, "Transition matrix A must be provided via class init or params."
        assert Q is not None, "Process noise covariance Q must be provided via class init or params."

        mean = A @ state.mean
        if u is not None and u.size > 0:
            if B is not None:
                mean = mean + B @ u
            else:
                mean = mean + u # Assume direct additive input if B not provided
                
        cov = A @ state.cov @ A.T + Q
        return GaussianState(mean=mean, cov=cov)

    @eqx.filter_jit
    def update(
        self, state: GaussianState, y: jax.Array, params: Any = None, return_likelihood: bool = False
    ) -> GaussianState:
        """
        Kalman Filter update step.
        """
        # Resolve H, R from params if provided, otherwise use class attributes
        H = params.H if params is not None and hasattr(params, 'H') else self.H
        R = params.R if params is not None and hasattr(params, 'R') else self.R
        
        assert H is not None, "Measurement matrix H must be provided via class init or params."
        assert R is not None, "Measurement noise covariance R must be provided via class init or params."

        # Innovation (residual)
        v = y - H @ state.mean
        
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
            ll = jax.scipy.stats.multivariate_normal.logpdf(y, mean=H @ state.mean, cov=S)
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
        Kalman Filter backward smoothing step.
        """
        A = params.A if params is not None and hasattr(params, 'A') else self.A
        B = params.B if params is not None and hasattr(params, 'B') else self.B
        Q = params.Q if params is not None and hasattr(params, 'Q') else self.Q
        
        assert A is not None, "Transition matrix A must be provided via class init or params."
        assert Q is not None, "Process noise covariance Q must be provided via class init or params."
        
        # 1. Forward predict state
        m_pred = A @ state.mean
        if u is not None and u.size > 0:
            if B is not None:
                m_pred = m_pred + B @ u
            else:
                m_pred = m_pred + u
                
        P_pred = A @ state.cov @ A.T + Q
        
        # 2. Smoother gain
        # D_k = P_{u,k} A^T P_{p, k+1}^{-1}
        # We avoid explicit inversion using solve: D_k^T = P_{p, k+1}^{-T} (A P_{u,k})^T => D_k = solve(P_pred.T, A @ P_u).T
        C = state.cov @ A.T
        D = jax.scipy.linalg.solve(P_pred.T, C.T).T
        
        # 3. Update mean and covariance
        mean = state.mean + D @ (next_smoothed_state.mean - m_pred)
        cov = state.cov + D @ (next_smoothed_state.cov - P_pred) @ D.T
        
        return GaussianState(mean=mean, cov=cov)
