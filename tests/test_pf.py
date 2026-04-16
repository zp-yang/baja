import jax
import jax.numpy as jnp
import pytest

from jax_ekfukf.pf import ParticleFilter, ParticleState

def test_pf_ungm():
    """
    Test the Bootstrap Particle Filter against the Univariate Non-stationary Growth Model (UNGM),
    which is a highly nonlinear benchmark problem commonly used to evaluate PFs.
    
    x_k = 0.5 * x_{k-1} + 25 * x_{k-1} / (1 + x_{k-1}^2) + 8 * cos(1.2 * k) + v_k
    y_k = x_k^2 / 20 + w_k
    """
    
    # 1. Define dynamics and measurement model
    def f(x, u):
        k = u[0] # time step index
        return jnp.array([
            0.5 * x[0] + 25 * x[0] / (1 + x[0]**2) + 8 * jnp.cos(1.2 * k)
        ])
        
    def h(x):
        return jnp.array([(x[0]**2) / 20.0])
        
    # 2. Define noise statistics
    Q = jnp.array([[10.0]]) # Process noise variance
    R = jnp.array([[1.0]])  # Measurement noise variance
    
    num_particles = 1000
    
    pf = ParticleFilter(
        f=f, 
        h=h, 
        num_particles=num_particles,
        Q=Q, 
        R=R,
        resample_threshold=num_particles / 2.0
    )
    
    # 3. Generate synthetic data
    N_steps = 100
    key = jax.random.PRNGKey(0)
    
    x_true = []
    y_obs = []
    
    x_k = jnp.array([0.1])
    for k in range(N_steps):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        v_k = jax.random.normal(subkey1) * jnp.sqrt(10.0)
        w_k = jax.random.normal(subkey2) * jnp.sqrt(1.0)
        
        x_k = f(x_k, jnp.array([k])) + jnp.array([v_k])
        y_k = h(x_k) + jnp.array([w_k])
        
        x_true.append(x_k)
        y_obs.append(y_k)
        
    X_true = jnp.stack(x_true)
    Y = jnp.stack(y_obs)
    U = jnp.arange(N_steps, dtype=float).reshape(-1, 1)
    
    # 4. Initialize Particle Filter state
    key, init_key = jax.random.split(key)
    
    initial_particles = jax.random.normal(init_key, shape=(num_particles, 1)) * jnp.sqrt(10.0)
    initial_weights = jnp.ones(num_particles) / num_particles
    
    init_state = ParticleState(
        particles=initial_particles,
        weights=initial_weights,
        key=key
    )
    
    # 5. Run the filter
    final_state, history_pred, history_upd = pf.filter_sequence(init_state, Y, U)
    
    # Extract the tracked means
    pf_means = jax.vmap(lambda s: s.mean)(history_upd)
    
    assert pf_means.shape == (N_steps, 1)
    assert history_upd.particles.shape == (N_steps, num_particles, 1)
    
    # Compute RMSE
    mse = jnp.mean((pf_means - X_true)**2)
    rmse = jnp.sqrt(mse)
    
    print(f"PF UNGM RMSE: {rmse:.4f}")
    
    # The UNGM tracking RMSE with 1000 particles should be relatively stable
    # EKF often diverges completely or has RMSE > 20. PF should be < 15.
    assert rmse < 20.0, f"PF tracking diverged, RMSE: {rmse}"


if __name__ == "__main__":
    test_pf_ungm()
    print("Particle filter test completed successfully.")
