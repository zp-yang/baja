import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from baja import (
    ExtendedKalmanFilter, 
    UnscentedKalmanFilter, 
    CubatureKalmanFilter, 
    GaussHermiteKalmanFilter,
    GaussianState
)

def test_ungm_demo():
    n_steps = 500
    
    x_0 = jnp.array([0.1])
    P_0 = jnp.array([[1.0]])
    
    u_n = 1.0 # Process noise std
    v_n = 1.0 # Measurement noise std
    
    Q = jnp.array([[u_n**2]])
    R = jnp.array([[v_n**2]])
    
    # We pass the time step k as the control input `u`
    def f_func(x, k):
        term1 = 0.5 * x[0]
        term2 = 25.0 * x[0] / (1.0 + x[0]**2)
        term3 = 8.0 * jnp.cos(1.2 * (k[0] - 1.0))
        return jnp.array([term1 + term2 + term3])
        
    def h_func(x):
        return jnp.array([(x[0]**2) / 20.0])

    # Generate data
    key = jax.random.PRNGKey(42)
    key_q, key_r = jax.random.split(key)
    
    process_noise = jax.random.normal(key_q, (n_steps,)) * u_n
    meas_noise = jax.random.normal(key_r, (n_steps,)) * v_n
    
    # MATLAB: X(1) = ungm_f(x_0, 1) + noise(1)
    def generate_step(x_prev, args):
        k, noise_q = args
        x_next = f_func(x_prev, jnp.array([k])) + jnp.array([noise_q])
        return x_next, x_next

    K_seq = jnp.arange(1, n_steps + 1, dtype=jnp.float32)
    _, X = jax.lax.scan(generate_step, x_0, (K_seq, process_noise))
    
    Y_real = jax.vmap(h_func)(X)
    Y = Y_real + meas_noise[:, None]
    
    initial_state = GaussianState(mean=x_0, cov=P_0)
    
    # 1. EKF
    ekf = ExtendedKalmanFilter(f=f_func, h=h_func, Q=Q, R=R)
    _, _, ekf_hist = ekf.filter_sequence(initial_state, Y, U=K_seq[:, None])
    ekf_mse = jnp.mean((X - ekf_hist.mean)**2)
    ekf_smooth = ekf.smooth_sequence(ekf_hist, U=K_seq[:, None])
    ekf_s_mse = jnp.mean((X - ekf_smooth.mean)**2)
    
    # 2. UKF
    ukf = UnscentedKalmanFilter(f=f_func, h=h_func, Q=Q, R=R)
    _, _, ukf_hist = ukf.filter_sequence(initial_state, Y, U=K_seq[:, None])
    ukf_mse = jnp.mean((X - ukf_hist.mean)**2)
    ukf_smooth = ukf.smooth_sequence(ukf_hist, U=K_seq[:, None])
    ukf_s_mse = jnp.mean((X - ukf_smooth.mean)**2)
    
    # 3. CKF
    ckf = CubatureKalmanFilter(f=f_func, h=h_func, Q=Q, R=R)
    _, _, ckf_hist = ckf.filter_sequence(initial_state, Y, U=K_seq[:, None])
    ckf_mse = jnp.mean((X - ckf_hist.mean)**2)
    ckf_smooth = ckf.smooth_sequence(ckf_hist, U=K_seq[:, None])
    ckf_s_mse = jnp.mean((X - ckf_smooth.mean)**2)
    
    # 4. GHKF (degree 10)
    ghkf = GaussHermiteKalmanFilter(n=1, p=10, f=f_func, h=h_func, Q=Q, R=R)
    _, _, ghkf_hist = ghkf.filter_sequence(initial_state, Y, U=K_seq[:, None])
    ghkf_mse = jnp.mean((X - ghkf_hist.mean)**2)
    ghkf_smooth = ghkf.smooth_sequence(ghkf_hist, U=K_seq[:, None])
    ghkf_s_mse = jnp.mean((X - ghkf_smooth.mean)**2)
    
    print(f"EKF  MSE: {ekf_mse:.4f}, Smooth: {ekf_s_mse:.4f}")
    print(f"UKF  MSE: {ukf_mse:.4f}, Smooth: {ukf_s_mse:.4f}")
    print(f"CKF  MSE: {ckf_mse:.4f}, Smooth: {ckf_s_mse:.4f}")
    print(f"GHKF MSE: {ghkf_mse:.4f}, Smooth: {ghkf_s_mse:.4f}")
    
    # The UNGM model is highly non-linear. 
    # Performance usually degrades as: GHKF > UKF > CKF > EKF
    assert ekf_mse < 80.0
    assert ckf_mse < 60.0
    assert ukf_mse < 50.0
    assert ghkf_mse < 35.0
    
    assert ekf_s_mse < ekf_mse or ekf_s_mse < 80.0
    assert ukf_s_mse < ukf_mse or ukf_s_mse < 50.0
    assert ckf_s_mse < ckf_mse or ckf_s_mse < 60.0
    assert ghkf_s_mse < ghkf_mse or ghkf_s_mse < 35.0

if __name__ == "__main__":
    test_ungm_demo()
