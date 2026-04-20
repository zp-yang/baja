import jax
import jax.numpy as jnp
import pytest
from baja import KalmanFilter, GaussianState, lti_disc

def test_kf_sine_demo():
    # Setup the same scenario as kf_sine_demo.m
    sd = 0.1
    dt = 0.1
    w = 1.0
    T = jnp.arange(0, 30 + dt, dt)
    
    # Generate true signal and measurements
    # Fix random seed for reproducibility
    key = jax.random.PRNGKey(0)
    X = jnp.sin(w * T)
    Y = X + sd * jax.random.normal(key, X.shape)
    
    # Initialize KF to values
    # M = [0;0], P = diag([0.1 2]), R = sd^2
    M = jnp.array([0.0, 0.0])
    P = jnp.diag(jnp.array([0.1, 2.0]))
    R = jnp.array([[sd**2]])
    H = jnp.array([[1.0, 0.0]])
    q = 0.1
    F = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    
    # LTI Discretization
    A, Q = lti_disc(F, jnp.eye(2), jnp.diag(jnp.array([0.0, q])), dt)
    
    initial_state = GaussianState(mean=M, cov=P)
    
    # Setup KF
    kf = KalmanFilter(A=A, Q=Q, H=H, R=R)
    
    # Run the filter sequence (offline mode)
    # We need to reshape Y to (Time, measurement_dim)
    Y_reshaped = Y[:, None]
    
    final_state, history_pred, history_update = kf.filter_sequence(initial_state, Y_reshaped)
    
    # The estimated signal is the first dimension of the mean history
    MM_0 = history_update.mean[:, 0]
    
    # Calculate RMS error
    rmse_kf = jnp.sqrt(jnp.mean((MM_0 - X)**2))
    print(f"KF RMSE: {rmse_kf:.4f}")
    
    # Assert RMSE is reasonably low, as it should be filtering
    assert rmse_kf < 0.15, "RMSE is too high, filtering failed"
    
    # Apply Smoother
    history_smooth = kf.smooth_sequence(history_update)
    SM_0 = history_smooth.mean[:, 0]
    rmse_rts = jnp.sqrt(jnp.mean((SM_0 - X)**2))
    print(f"RTS RMSE: {rmse_rts:.4f}")
    
    assert rmse_rts < rmse_kf, "Smoother should perform better than filter"

if __name__ == "__main__":
    test_kf_sine_demo()
