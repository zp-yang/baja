import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Any, Callable
from .base import AbstractFilter, GaussianState
from .transforms import compute_ghkf_weights_and_points, generate_ghkf_sigmas, unscented_transform

class GaussHermiteKalmanFilter(AbstractFilter):
    """
    Standard Additive Gauss-Hermite Kalman Filter (GHKF).
    
    This implementation utilizes the Gauss-Hermite quadrature rule 
    to propagate the state mean and covariance through nonlinear 
    transition and measurement functions.
    """
    
    # Static parameters (can be overridden by params argument)
    f: Callable
    h: Callable
    p: int
    Q: Optional[jax.Array] = None
    R: Optional[jax.Array] = None
    _X_base: jax.Array = eqx.field(static=False)
    _W: jax.Array = eqx.field(static=False)
    _c: float = eqx.field(static=True)
    
    def __init__(
        self, 
        n: int,
        f: Callable, 
        h: Callable, 
        p: int = 3,
        Q: Optional[jax.Array] = None, 
        R: Optional[jax.Array] = None, 
    ):
        """
        Initializes the GHKF.
        
        Args:
            n: State dimension (required to pre-calculate quadrature points).
            f: Nonlinear state transition function.
            h: Nonlinear measurement function.
            p: Degree of approximation (number of quadrature points per dimension).
            Q: Process noise covariance.
            R: Measurement noise covariance.
        """
        self.f = f
        self.h = h
        self.p = p
        self.Q = Q
        self.R = R
        
        # Pre-calculate base points and weights since they only depend on n and p
        X_base, W, c = compute_ghkf_weights_and_points(n, p)
        self._X_base = X_base
        self._W = W
        self._c = c

    @eqx.filter_jit
    def predict(
        self, state: GaussianState, u: Optional[jax.Array] = None, params: Any = None
    ) -> GaussianState:
        """
        GHKF prediction step.
        """
        Q = params.Q if params is not None and hasattr(params, 'Q') else self.Q
        assert Q is not None, "Process noise covariance Q must be provided."
        
        # 1. Generate sigma points
        sigmas = generate_ghkf_sigmas(state.mean, state.cov, self._X_base, self._c)
        
        # 2. Propagate sigma points through nonlinear f
        def f_with_u(x):
            if u is not None and u.size > 0:
                return self.f(x, u)
            return self.f(x)
            
        mean, cov, _, _ = unscented_transform(sigmas, self._W, self._W, f_with_u, Q)
        
        return GaussianState(mean=mean, cov=cov)

    @eqx.filter_jit
    def update(
        self, state: GaussianState, y: jax.Array, params: Any = None, return_likelihood: bool = False
    ) -> GaussianState:
        """
        GHKF update step.
        """
        R = params.R if params is not None and hasattr(params, 'R') else self.R
        assert R is not None, "Measurement noise covariance R must be provided."
        
        # 1. Generate sigma points from the predicted state
        sigmas = generate_ghkf_sigmas(state.mean, state.cov, self._X_base, self._c)
        
        # 2. Propagate sigma points through nonlinear h
        z_mean, S, Z_sigmas, Z_diff = unscented_transform(sigmas, self._W, self._W, self.h, R)
        
        # 3. Compute cross-covariance C
        X_diff = sigmas - state.mean[None, :]
        C = X_diff.T @ (self._W[:, None] * Z_diff)
        
        # 4. Kalman gain
        K = jax.scipy.linalg.solve(S.T, C.T).T
        
        # 5. Update mean and covariance
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
        GHKF backward smoothing step (GHRTS).
        """
        Q = params.Q if params is not None and hasattr(params, 'Q') else self.Q
        assert Q is not None, "Process noise covariance Q must be provided."
        
        # 1. Generate sigma points
        sigmas = generate_ghkf_sigmas(state.mean, state.cov, self._X_base, self._c)
        
        # 2. Propagate sigma points through nonlinear f
        def f_with_u(x):
            if u is not None and u.size > 0:
                return self.f(x, u)
            return self.f(x)
            
        m_pred, P_pred, mapped_sigmas, Y_diff = unscented_transform(sigmas, self._W, self._W, f_with_u, Q)
        
        # 3. Compute cross-covariance C
        X_diff = sigmas - state.mean[None, :]
        C = X_diff.T @ (self._W[:, None] * Y_diff)
        
        # 4. Smoother gain
        D = jax.scipy.linalg.solve(P_pred.T, C.T).T
        
        # 5. Update mean and covariance
        mean = state.mean + D @ (next_smoothed_state.mean - m_pred)
        cov = state.cov + D @ (next_smoothed_state.cov - P_pred) @ D.T
        
        return GaussianState(mean=mean, cov=cov)
