import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Tuple, Optional
import itertools

def compute_ut_weights(
    n: int, alpha: float = 1.0, beta: float = 0.0, kappa: Optional[float] = None
) -> Tuple[jax.Array, jax.Array, float]:
    """
    Computes unscented transformation weights.
    
    Args:
        n: Dimensionality of the random variable.
        alpha: Spread of the sigma points. Defaults to 1.0.
        beta: Prior knowledge of the distribution (2.0 is optimal for Gaussian). Defaults to 0.0.
        kappa: Secondary scaling parameter. Defaults to 3.0 - n.
        
    Returns:
        WM: Weights for mean calculation (2n+1,)
        WC: Weights for covariance calculation (2n+1,)
        c: Scaling constant
    """
    if kappa is None:
        kappa = 3.0 - n
        
    lambda_ = alpha**2 * (n + kappa) - n
    c = n + lambda_
    
    # Weights for mean
    wm_0 = lambda_ / c
    # Weights for covariance
    wc_0 = lambda_ / c + (1.0 - alpha**2 + beta)
    
    # Weights for the rest of the sigma points
    w_rest = 1.0 / (2.0 * c)
    
    WM = jnp.full(2 * n + 1, w_rest)
    WM = WM.at[0].set(wm_0)
    
    WC = jnp.full(2 * n + 1, w_rest)
    WC = WC.at[0].set(wc_0)
    
    return WM, WC, c

def generate_sigmas(mean: jax.Array, cov: jax.Array, c: float) -> jax.Array:
    """
    Generates Sigma Points for the Unscented Transformation.
    
    Args:
        mean: State mean vector (N,)
        cov: State covariance matrix (N, N)
        c: Scaling constant from compute_ut_weights
        
    Returns:
        sigmas: Matrix of sigma points (2N+1, N)
    """
    # L @ L.T = cov
    L = jax.scipy.linalg.cholesky(cov, lower=True)
    
    # The i-th column of L is L[:, i].
    # L.T has shape (n, n), so its i-th row is L[:, i].
    offsets = jnp.sqrt(c) * L.T
    
    sigmas = jnp.vstack([
        mean[None, :],
        mean[None, :] + offsets,
        mean[None, :] - offsets
    ])
    
    return sigmas

def compute_ckf_weights(n: int) -> Tuple[jax.Array, jax.Array, float]:
    """
    Computes Cubature Kalman Filter weights.
    
    Args:
        n: Dimensionality of the random variable.
        
    Returns:
        W: Weights for mean and covariance calculation (2n,)
        W: (same as above)
        c: Scaling constant (n)
    """
    W = jnp.full(2 * n, 1.0 / (2.0 * n))
    c = float(n)
    return W, W, c

def generate_ckf_sigmas(mean: jax.Array, cov: jax.Array, c: float) -> jax.Array:
    """
    Generates Sigma Points for the Cubature Transformation.
    
    Args:
        mean: State mean vector (N,)
        cov: State covariance matrix (N, N)
        c: Scaling constant (usually n)
        
    Returns:
        sigmas: Matrix of sigma points (2N, N)
    """
    L = jax.scipy.linalg.cholesky(cov, lower=True)
    offsets = jnp.sqrt(c) * L.T
    
    sigmas = jnp.vstack([
        mean[None, :] + offsets,
        mean[None, :] - offsets
    ])
    return sigmas

def compute_ghkf_weights_and_points(n: int, p: int) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Computes Gauss-Hermite Kalman Filter base points and weights.
    
    Args:
        n: Dimensionality of the random variable.
        p: Number of points per dimension.
        
    Returns:
        X_base: Base Cartesian product of 1D roots (p^n, n)
        W: ND weights (p^n,)
        c: Scaling constant (sqrt(2))
    """
    # Use numpy to get 1D roots and weights, since this is a static setup
    x_1d, w_1d = np.polynomial.hermite.hermgauss(p)
    
    # Cartesian product for n dimensions
    # list of p^n tuples, each of length n
    X_base = np.array(list(itertools.product(x_1d, repeat=n)))
    
    # Same for weights, then multiply them
    W_base = np.array(list(itertools.product(w_1d, repeat=n)))
    W = np.prod(W_base, axis=1) / (np.pi ** (n / 2.0))
    
    return jnp.array(X_base), jnp.array(W), jnp.sqrt(2.0)

def generate_ghkf_sigmas(mean: jax.Array, cov: jax.Array, X_base: jax.Array, c: float) -> jax.Array:
    """
    Generates Sigma Points for the Gauss-Hermite Transformation.
    
    Args:
        mean: State mean vector (N,)
        cov: State covariance matrix (N, N)
        X_base: Base Cartesian points (p^n, n)
        c: Scaling constant (sqrt(2))
        
    Returns:
        sigmas: Matrix of sigma points (p^n, N)
    """
    L = jax.scipy.linalg.cholesky(cov, lower=True)
    # X_base is (p^n, n). We need (L @ X_base.T).T => X_base @ L.T
    offsets = c * (X_base @ L.T)
    sigmas = mean[None, :] + offsets
    return sigmas

def unscented_transform(
    sigmas: jax.Array, 
    WM: jax.Array, 
    WC: jax.Array, 
    f: Callable, 
    noise_cov: Optional[jax.Array] = None, 
    *args
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Applies the Unscented Transformation given a nonlinear function.
    Works for UKF, CKF, and GHKF.
    
    Args:
        sigmas: Sigma points (num_sigmas, D_in)
        WM: Mean weights (num_sigmas,)
        WC: Covariance weights (num_sigmas,)
        f: Nonlinear function to map sigma points f(x, *args)
        noise_cov: Optional additive noise covariance matrix (D_out, D_out)
        *args: Additional arguments to f
        
    Returns:
        mean: Transformed mean (D_out,)
        cov: Transformed covariance (D_out, D_out)
        mapped_sigmas: The sigmas passed through f (num_sigmas, D_out)
        diff: The centered mapped sigmas (num_sigmas, D_out)
    """
    # Map f over each sigma point (row by row)
    mapped_sigmas = jax.vmap(lambda x: f(x, *args))(sigmas)
    
    # Compute transformed mean
    mean = jnp.sum(WM[:, None] * mapped_sigmas, axis=0)
    
    # Compute transformed covariance
    diff = mapped_sigmas - mean[None, :]
    cov = diff.T @ (WC[:, None] * diff)
    
    if noise_cov is not None:
        cov += noise_cov
        
    return mean, cov, mapped_sigmas, diff
