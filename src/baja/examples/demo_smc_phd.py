import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from baja import SMCPHDState, SMCPHDFilter


def main():
    # -------------------------------------------------------------------------
    # 1. Setup Tracking Parameters
    # -------------------------------------------------------------------------
    dt = 1.0
    num_steps = 50
    state_dim = 4  # [x, vx, y, vy]
    meas_dim = 2   # [x, y]
    
    # Dynamics (Constant Velocity)
    F = jnp.array([
        [1.0,  dt, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0,  dt],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    def f(x):
        return F @ x

    # Measurement (Position only)
    H = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    def h(x):
        return H @ x
    
    # Noise Covariances
    q_scalar = 0.1
    Q = jnp.array([
        [dt**3/3, dt**2/2, 0.0, 0.0],
        [dt**2/2, dt,      0.0, 0.0],
        [0.0, 0.0, dt**3/3, dt**2/2],
        [0.0, 0.0, dt**2/2, dt]
    ]) * q_scalar
    
    R = jnp.eye(meas_dim) * 2.0
    
    # PHD Parameters
    P_S = 0.95  # Survival probability
    P_D = 0.90  # Detection probability
    lambda_c = 10.0  # expected clutter points per scan
    area = 200.0 * 200.0  # observation area [-100, 100] x [-100, 100]
    clutter_intensity = lambda_c / area
    
    num_particles = 1000
    num_birth_particles = 100

    # -------------------------------------------------------------------------
    # 2. Define Birth Model
    # -------------------------------------------------------------------------
    def birth_model(key, num_birth_particles):
        """Uniform birth over the tracking region."""
        k1, k2 = jax.random.split(key)
        # Position uniformly in [-100, 100]
        pos = jax.random.uniform(k1, (num_birth_particles, 2), minval=-100.0, maxval=100.0)
        # Velocity normally distributed around 0
        vel = jax.random.normal(k2, (num_birth_particles, 2)) * 1.0
        
        particles = jnp.column_stack([pos[:, 0], vel[:, 0], pos[:, 1], vel[:, 1]])
        
        # Expected number of new targets appearing per step
        expected_births = 0.1
        weights = jnp.full((num_birth_particles,), expected_births / num_birth_particles)
        
        return particles, weights

    # -------------------------------------------------------------------------
    # 3. Simulate Ground Truth & Measurements
    # -------------------------------------------------------------------------
    np.random.seed(42)
    key = jax.random.PRNGKey(42)
    
    # Target definitions: (start_time, end_time, initial_state)
    targets_info = [
        (0, 50, np.array([-50.0, 2.0, -50.0, 2.0])),   # Target 1
        (10, 50, np.array([50.0, -2.5, -50.0, 2.5])),  # Target 2
        (20, 40, np.array([0.0, 0.0, 50.0, -3.0]))     # Target 3 (disappears early)
    ]
    
    true_states = []    # list of lists for plotting
    measurements = []   # will be converted to JAX array later
    max_measurements_per_step = 30 # buffer size for JAX
    
    for t in range(num_steps):
        current_targets = []
        for (start_t, end_t, state) in targets_info:
            if start_t <= t < end_t:
                steps_active = t - start_t
                # True position
                current_state = np.linalg.matrix_power(F, steps_active) @ state
                current_targets.append(current_state)
        true_states.append(current_targets)
        
        # Generate measurements
        Z_t = []
        # Target detections
        for state in current_targets:
            if np.random.rand() < P_D:
                noise = np.random.multivariate_normal([0, 0], R)
                z = H @ state + noise
                Z_t.append(z)
                
        # Clutter
        num_clutter = np.random.poisson(lambda_c)
        for _ in range(num_clutter):
            z_c = np.random.uniform(-100, 100, size=2)
            Z_t.append(z_c)
            
        # Pad with NaNs for fixed shape
        Z_t = np.array(Z_t)
        padded_Z_t = np.full((max_measurements_per_step, meas_dim), np.nan)
        if len(Z_t) > 0:
            num_meas = min(len(Z_t), max_measurements_per_step)
            padded_Z_t[:num_meas] = Z_t[:num_meas]
            
        measurements.append(padded_Z_t)

    Y = jnp.array(measurements)  # Shape: (num_steps, max_measurements_per_step, meas_dim)

    # -------------------------------------------------------------------------
    # 4. Initialize and Run Filter
    # -------------------------------------------------------------------------
    filter = SMCPHDFilter(
        f=f, h=h, birth_model=birth_model,
        num_particles=num_particles, num_birth_particles=num_birth_particles,
        P_S=P_S, P_D=P_D, clutter_intensity=clutter_intensity,
        Q=Q, R=R
    )
    
    # Initialize with 0 targets
    key, subkey = jax.random.split(key)
    init_particles = jax.random.uniform(subkey, (num_particles, state_dim), minval=-100, maxval=100)
    init_weights = jnp.zeros(num_particles)
    initial_state = SMCPHDState(particles=init_particles, weights=init_weights, key=key)
    
    print("Running SMC-PHD Filter over sequence...")
    # Using JAX's fast scanner
    final_state, history_pred, history_update = filter.filter_sequence(initial_state, Y)
    
    # -------------------------------------------------------------------------
    # 5. Extract Results and Plot
    # -------------------------------------------------------------------------
    print("Plotting results...")
    
    # Get expected number of targets at each step
    expected_targets_history = jnp.sum(history_update.weights, axis=1)
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Tracking Scenario
    plt.subplot(1, 2, 1)
    plt.title("SMC-PHD Tracking Scenario")
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    
    # Plot True Tracks
    for t_idx, (start_t, end_t, _) in enumerate(targets_info):
        track_x, track_y = [], []
        for t in range(start_t, min(end_t, num_steps)):
            state = true_states[t][t_idx if t_idx < len(true_states[t]) else -1]
            track_x.append(state[0])
            track_y.append(state[2])
        plt.plot(track_x, track_y, 'k--', alpha=0.5, label='True Track' if t_idx==0 else "")
        plt.scatter(track_x[0], track_y[0], c='g', marker='o', s=100, label='Start' if t_idx==0 else "")
        plt.scatter(track_x[-1], track_y[-1], c='r', marker='x', s=100, label='End' if t_idx==0 else "")

    # Plot Measurements (Clutter + True)
    all_meas_x = Y[:, :, 0].flatten()
    all_meas_y = Y[:, :, 1].flatten()
    valid_meas = ~np.isnan(all_meas_x)
    plt.scatter(all_meas_x[valid_meas], all_meas_y[valid_meas], c='gray', s=10, alpha=0.3, label='Measurements/Clutter')
    
    # Plot Estimated Particle Cloud
    # To avoid plotting too many, we'll subsample or only plot particles with high weight
    for t in range(0, num_steps, 2):  # plot every 2 steps for clarity
        particles = history_update.particles[t]
        weights = history_update.weights[t]
        
        # Plot top 10% particles to represent estimates
        threshold = np.percentile(weights, 90)
        strong_particles = particles[weights > threshold]
        if len(strong_particles) > 0:
            plt.scatter(strong_particles[:, 0], strong_particles[:, 2], c='blue', s=2, alpha=0.5)

    plt.legend()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    
    # Plot 2: Cardinality Estimate (Number of Targets)
    plt.subplot(1, 2, 2)
    plt.title("Estimated Number of Targets")
    
    true_cardinality = [len(targets) for targets in true_states]
    plt.plot(range(num_steps), true_cardinality, 'k--', label='True Cardinality')
    plt.plot(range(num_steps), expected_targets_history, 'b-', label='Estimated Cardinality (SMC-PHD)')
    plt.xlabel("Time Step")
    plt.ylabel("Number of Targets")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = "smc_phd_demo.png"
    plt.savefig(plot_path)
    print(f"Demo complete! Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
