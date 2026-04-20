import jax
import jax.numpy as jnp
import pytest

from baja.pff import EDHFilter, LEDHKNNFilter
from baja.pf import ParticleState

def test_pff_linear():
    """
    Test EDH on a simpler linear/mildly-nonlinear problem where it shouldn't diverge.
    """
    def f(x, u=None):
        return jnp.array([x[0] + 0.1 * x[1], x[1]])
        
    def h(x):
        return jnp.array([x[0]])
        
    Q = jnp.eye(2) * 0.1
    R = jnp.eye(1) * 1.0
    num_particles = 100
    
    edh = EDHFilter(f=f, h=h, num_particles=num_particles, Q=Q, R=R, flow_steps=10)
    
    N_steps = 20
    key = jax.random.PRNGKey(42)
    Y = jax.random.normal(key, (N_steps, 1))
    
    init_particles = jax.random.normal(key, (num_particles, 2))
    init_state = ParticleState(particles=init_particles, weights=jnp.ones(num_particles)/num_particles, key=key)
    
    _, _, history = edh.filter_sequence(init_state, Y)
    means = jax.vmap(lambda s: s.mean)(history)
    
    assert not jnp.isnan(means).any(), "EDH diverged on simple problem"

def test_pff_ungm():
    """
    Test the Particle Flow Filters (LEDH) against the UNGM benchmark.
    EDH typically diverges on UNGM without careful step-size tuning due to 
    the highly multi-modal prior, showcasing the necessity of LEDH localization.
    """
    def f(x, u=None):
        k = u[0]
        return jnp.array([
            0.5 * x[0] + 25 * x[0] / (1 + x[0]**2) + 8 * jnp.cos(1.2 * k)
        ])
        
    def h(x):
        return jnp.array([(x[0]**2) / 20.0])
        
    Q = jnp.array([[10.0]])
    R = jnp.array([[1.0]])
    
    num_particles = 200
    
    ledh = LEDHKNNFilter(
        f=f, h=h, num_particles=num_particles,
        Q=Q, R=R, flow_steps=10, knn_fraction=0.1
    )
    
    N_steps = 50
    key = jax.random.PRNGKey(42)
    
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
    
    key, init_key = jax.random.split(key)
    initial_particles = jax.random.normal(init_key, shape=(num_particles, 1)) * jnp.sqrt(10.0)
    
    init_state = ParticleState(
        particles=initial_particles,
        weights=jnp.ones(num_particles) / num_particles,
        key=key
    )
    
    _, _, history_upd_ledh = ledh.filter_sequence(init_state, Y, U)
    ledh_means = jax.vmap(lambda s: s.mean)(history_upd_ledh)
    
    assert ledh_means.shape == (N_steps, 1)
    
    rmse_ledh = jnp.sqrt(jnp.mean((ledh_means - X_true)**2))
    print(f"LEDH-KNN RMSE: {rmse_ledh:.4f}")
    assert not jnp.isnan(rmse_ledh), "LEDH diverged"
    assert rmse_ledh < 30.0, f"LEDH diverged, RMSE: {rmse_ledh}"

if __name__ == "__main__":
    test_pff_linear()
    test_pff_ungm()
    print("Particle Flow tests completed successfully.")

def test_pff_variants_ungm():
    """
    Test KDE and GMM variants of LEDH on the UNGM benchmark.
    """
    from jax_ekfukf.pff import LEDHKDEFilter, LEDHGMMFilter
    
    def f(x, u):
        k = u[0]
        return jnp.array([
            0.5 * x[0] + 25 * x[0] / (1 + x[0]**2) + 8 * jnp.cos(1.2 * k)
        ])
        
    def h(x):
        return jnp.array([(x[0]**2) / 20.0])
        
    Q = jnp.array([[10.0]])
    R = jnp.array([[1.0]])
    
    num_particles = 200
    
    ledh_kde = LEDHKDEFilter(
        f=f, h=h, num_particles=num_particles,
        Q=Q, R=R, flow_steps=10, bandwidth=2.0
    )
    
    ledh_gmm = LEDHGMMFilter(
        f=f, h=h, num_particles=num_particles,
        Q=Q, R=R, flow_steps=10, num_components=3, em_iterations=5
    )
    
    N_steps = 50
    key = jax.random.PRNGKey(42)
    
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
    
    key, init_key = jax.random.split(key)
    initial_particles = jax.random.normal(init_key, shape=(num_particles, 1)) * jnp.sqrt(10.0)
    
    init_state = ParticleState(
        particles=initial_particles,
        weights=jnp.ones(num_particles) / num_particles,
        key=key
    )
    
    _, _, history_upd_kde = ledh_kde.filter_sequence(init_state, Y, U)
    kde_means = jax.vmap(lambda s: s.mean)(history_upd_kde)
    
    _, _, history_upd_gmm = ledh_gmm.filter_sequence(init_state, Y, U)
    gmm_means = jax.vmap(lambda s: s.mean)(history_upd_gmm)
    
    rmse_kde = jnp.sqrt(jnp.mean((kde_means - X_true)**2))
    rmse_gmm = jnp.sqrt(jnp.mean((gmm_means - X_true)**2))
    
    print(f"LEDH-KDE RMSE: {rmse_kde:.4f}")
    print(f"LEDH-GMM RMSE: {rmse_gmm:.4f}")
    
    assert rmse_kde < 30.0, f"LEDH-KDE diverged, RMSE: {rmse_kde}"
    assert rmse_gmm < 30.0, f"LEDH-GMM diverged, RMSE: {rmse_gmm}"

if __name__ == "__main__":
    test_pff_variants_ungm()
