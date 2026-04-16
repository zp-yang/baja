import jax
import jax.numpy as jnp
import pytest
from jax_ekfukf import UnscentedKalmanFilter, GaussianState, lti_disc

def test_ukf_sine_demo():
    # Setup the same scenario as ekf_sine_demo.m (for UKF comparison)
    f = 0.0
    w = 10.0
    a = 1.0
    
    d = 5.0
    n = 500
    dt = d / n
    x_steps = jnp.arange(1, n + 1)
    
    # Continuous time transition and noise effect matrices
    F = jnp.array([[0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
    L = jnp.array([[0.0, 0.0],
                   [1.0, 0.0],
                   [0.0, 1.0]])
    q1 = 0.2
    q2 = 0.1
    Qc = jnp.diag(jnp.array([q1, q2]))
    
    # Discretize
    A, Q = lti_disc(F, L, Qc, dt)
    
    # Define non-linear transition and measurement functions
    def f_func(x, u=None):
        return A @ x
        
    def h_func(x):
        # x[0] = theta, x[1] = omega, x[2] = a
        return jnp.array([x[2] * jnp.sin(x[0])])
    
    # Generate the real signal
    key = jax.random.PRNGKey(0)
    key_q, key_r = jax.random.split(key)
    
    def generate_step(x_prev, noise_q):
        x_next = A @ x_prev + noise_q
        return x_next, x_next

    # Draw process noise
    noise_Q = jax.random.multivariate_normal(
        key_q, mean=jnp.zeros(3), cov=Q, shape=(n,)
    )
    
    initial_x = jnp.array([f, w, a])
    _, X_rest = jax.lax.scan(generate_step, initial_x, noise_Q[1:])
    X = jnp.vstack([initial_x, X_rest])
    
    # Generate observations
    sd = 1.0
    R = jnp.array([[sd**2]])
    Y_real = jax.vmap(h_func)(X)
    Y = Y_real + jax.random.normal(key_r, Y_real.shape) * sd
    
    # Initial guesses
    M = jnp.array([f, w, a])
    P = jnp.diag(jnp.array([3.0, 3.0, 3.0]))
    initial_state = GaussianState(mean=M, cov=P)
    
    # Initialize UKF
    # Note: Using standard Additive UKF with default alpha=1, beta=0, kappa=3-n
    ukf = UnscentedKalmanFilter(f=f_func, h=h_func, Q=Q, R=R)
    
    # Run the filter sequence
    final_state, history_pred, history_update = ukf.filter_sequence(initial_state, Y)
    
    # Projected estimates
    Y_m_u = jax.vmap(h_func)(history_update.mean)
    
    # RMSE calculation
    rmse_ukf = jnp.sqrt(jnp.mean((Y_m_u - Y_real)**2))
    print(f"UKF RMSE: {rmse_ukf:.4f}")
    
    # In MATLAB the UKF MSE was similar or slightly better than EKF
    # We expect RMSE to be reasonably low (<< sd=1.0)
    assert rmse_ukf < 0.6, f"UKF RMSE too high: {rmse_ukf:.4f}"

    # Smooth the sequence
    history_smooth = ukf.smooth_sequence(history_update)
    Y_s = jax.vmap(h_func)(history_smooth.mean)
    rmse_urts = jnp.sqrt(jnp.mean((Y_s - Y_real)**2))
    print(f"URTS RMSE: {rmse_urts:.4f}")
    
    assert rmse_urts < rmse_ukf, "Smoother should perform better than filter"

if __name__ == "__main__":
    test_ukf_sine_demo()
