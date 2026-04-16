import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Any, Callable
from .base import AbstractFilter, GaussianState
from .transforms import compute_ckf_weights, generate_ckf_sigmas, unscented_transform

class CubatureKalmanFilter(AbstractFilter):
    """
    Standard Additive Cubature Kalman Filter (CKF).
    
    This implementation utilizes the spherical-radial cubature rule 
    to propagate the state mean and covariance through nonlinear 
    transition and measurement functions.
    """
    
    # Static parameters (can be overridden by params argument)
    f: Callable
    h: Callable
    Q: Optional[jax.Array] = None
    R: Optional[jax.Array] = None
    
    def __init__(
        self, 
        f: Callable, 
        h: Callable, 
        Q: Optional[jax.Array] = None, 
        R: Optional[jax.Array] = None, 
    ):
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R

    @eqx.filter_jit
    def predict(
        self, state: GaussianState, u: Optional[jax.Array] = None, params: Any = None
    ) -> GaussianState:
        """
        CKF prediction step.
        """
        Q = params.Q if params is not None and hasattr(params, 'Q') else self.Q
        assert Q is not None, "Process noise covariance Q must be provided."
        
        n = state.mean.shape[0]
        
        # 1. Compute CKF weights and scaling
        WM, WC, c = compute_ckf_weights(n)
        
        # 2. Generate sigma points
        sigmas = generate_ckf_sigmas(state.mean, state.cov, c)
        
        # 3. Propagate sigma points through nonlinear f
        def f_with_u(x):
            if u is not None and u.size > 0:
                return self.f(x, u)
            return self.f(x)
            
        mean, cov, _, _ = unscented_transform(sigmas, WM, WC, f_with_u, Q)
        
        return GaussianState(mean=mean, cov=cov)

    @eqx.filter_jit
    def update(
        self, state: GaussianState, y: jax.Array, params: Any = None, return_likelihood: bool = False
    ) -> GaussianState:
        """
        CKF update step.
        """
        R = params.R if params is not None and hasattr(params, 'R') else self.R
        assert R is not None, "Measurement noise covariance R must be provided."
        
        n = state.mean.shape[0]
        
        # 1. Compute CKF weights and scaling
        WM, WC, c = compute_ckf_weights(n)
        
        # 2. Generate sigma points from the predicted state
        sigmas = generate_ckf_sigmas(state.mean, state.cov, c)
        
        # 3. Propagate sigma points through nonlinear h
        z_mean, S, Z_sigmas, Z_diff = unscented_transform(sigmas, WM, WC, self.h, R)
        
        # 4. Compute cross-covariance C
        X_diff = sigmas - state.mean[None, :]
        C = X_diff.T @ (WC[:, None] * Z_diff)
        
        # 5. Kalman gain
        K = jax.scipy.linalg.solve(S.T, C.T).T
        
        # 6. Update mean and covariance
        mean = state.mean + K @ (y - z_mean)
        cov = state.cov - K @ S @ K.T
        
        if return_likelihood:
            ll = jax.scipy.stats.multivariate_normal.logpdf(y, mean=z_mean, cov=S)
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
        CKF backward smoothing step (CRTS).
        """
        Q = params.Q if params is not None and hasattr(params, 'Q') else self.Q
        assert Q is not None, "Process noise covariance Q must be provided."
        
        n = state.mean.shape[0]
        
        # 1. Compute CKF weights and scaling
        WM, WC, c = compute_ckf_weights(n)
        
        # 2. Generate sigma points
        sigmas = generate_ckf_sigmas(state.mean, state.cov, c)
        
        # 3. Propagate sigma points through nonlinear f
        def f_with_u(x):
            if u is not None and u.size > 0:
                return self.f(x, u)
            return self.f(x)
            
        m_pred, P_pred, mapped_sigmas, Y_diff = unscented_transform(sigmas, WM, WC, f_with_u, Q)
        
        # 4. Compute cross-covariance C
        X_diff = sigmas - state.mean[None, :]
        C = X_diff.T @ (WC[:, None] * Y_diff)
        
        # 5. Smoother gain
        D = jax.scipy.linalg.solve(P_pred.T, C.T).T
        
        # 6. Update mean and covariance
        mean = state.mean + D @ (next_smoothed_state.mean - m_pred)
        cov = state.cov + D @ (next_smoothed_state.cov - P_pred) @ D.T
        
        return GaussianState(mean=mean, cov=cov)
