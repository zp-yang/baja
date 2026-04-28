import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import equinox as eqx
import time

# Assuming baja is installed or in python path as per project structure
from baja.pf import ParticleFilter, ParticleState

class AcousticTracking(eqx.Module):
    """
    Class to store parameters and functions for the acoustic multi-target tracking example.
    """
    T: int
    nTarget: int
    dimState_per_target: int
    simAreaSize: float
    sensorsPos: jnp.ndarray
    nSensor: int
    Amp: float
    invPow: float
    d0: float
    measvar_real: float
    measvar: float
    x0: jnp.ndarray
    Phi: jnp.ndarray
    Q: jnp.ndarray
    Q_real: jnp.ndarray
    sigma0: jnp.ndarray

    def __init__(self):
        # Base setup
        self.T = 40
        self.nTarget = 4
        self.dimState_per_target = 4
        self.simAreaSize = 40.0
        
        # 5x5 sensor grid
        sensors_xy = np.array([
            [0, 0], [10, 0], [20, 0], [30, 0], [40, 0],
            [0, 10], [10, 10], [20, 10], [30, 10], [40, 10],
            [0, 20], [10, 20], [20, 20], [30, 20], [40, 20],
            [0, 30], [10, 30], [20, 30], [30, 30], [40, 30],
            [0, 40], [10, 40], [20, 40], [30, 40], [40, 40]
        ], dtype=float)
        self.sensorsPos = jnp.array(sensors_xy)
        self.nSensor = self.sensorsPos.shape[0]
        
        # Measurement parameters
        self.Amp = 10.0
        self.invPow = 1.0
        self.d0 = 0.1
        self.measvar_real = 0.01
        self.measvar = 0.01
        
        # Initial state: 4 targets (x, y, dx, dy interleaved as [x, y, dx, dy, ...])
        self.x0 = jnp.array([
            12, 6, 0.001, 0.001,
            32, 32, -0.001, -0.005,
            20, 13, -0.1, 0.01,
            15, 35, 0.002, 0.002
        ], dtype=float)
        
        # Motion model parameters for a single target
        Phi_single = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        Gamma_single = np.array([
            [1/3, 0, 0.5, 0],
            [0, 1/3, 0, 0.5],
            [0.5, 0, 1, 0],
            [0, 0.5, 0, 1]
        ], dtype=float)
        
        gammavar_real = 0.05
        Qii_real = gammavar_real * Gamma_single
        Qii = np.array([
            [3, 0, 0.1, 0],
            [0, 3, 0, 0.1],
            [0.1, 0, 0.03, 0],
            [0, 0.1, 0, 0.03]
        ], dtype=float)
        
        # Block diagonalize for all targets
        self.Phi = jnp.array(scipy.linalg.block_diag(*[Phi_single]*self.nTarget))
        self.Q = jnp.array(scipy.linalg.block_diag(*[Qii]*self.nTarget))
        self.Q_real = jnp.array(scipy.linalg.block_diag(*[Qii_real]*self.nTarget))
        
        # Initial covariance
        sigma0_single = 10 * np.array([1, 1, 0.1, 0.1])
        self.sigma0 = jnp.diag(jnp.tile(sigma0_single, self.nTarget))

    def f(self, state):
        """
        Deterministic process function.
        x_{k} = Phi * x_{k-1}
        """
        return self.Phi @ state

    def h(self, state):
        """
        Non-linear measurement function for a single state vector.
        state: (16,)
        Returns: (25,) measurement vector
        """
        # Extract x and y coordinates of the targets
        x_pos = state[0::4]
        y_pos = state[1::4]
        
        target_pos = jnp.stack([x_pos, y_pos], axis=1) # Shape: (4, 2)
        
        # Compute squared distances: (targets, sensors, 2)
        diff = target_pos[:, None, :] - self.sensorsPos[None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1) # Shape: (4, 25)
        
        # Compute amplitudes and sum across targets
        amplitudes = self.Amp / (dist ** self.invPow + self.d0) # Shape: (4, 25)
        y = jnp.sum(amplitudes, axis=0) # Shape: (25,)
        
        return y

    def generate_tracks(self, key):
        """
        Generates target trajectories, resampling if any target goes out of bounds.
        """
        state_dim = self.x0.shape[0]
        lower_bound = 0.05 * self.simAreaSize
        upper_bound = 0.95 * self.simAreaSize
        
        while True:
            key, subkey = jax.random.split(key)
            noise = jax.random.multivariate_normal(
                subkey, mean=jnp.zeros(state_dim), cov=self.Q_real, shape=(self.T,)
            )
            
            tracks = np.zeros((self.T, state_dim))
            curr_state = self.Phi @ self.x0 + noise[0]
            tracks[0] = curr_state
            
            for t in range(1, self.T):
                curr_state = self.Phi @ curr_state + noise[t]
                tracks[t] = curr_state
                
            xx = tracks[:, 0::4]
            yy = tracks[:, 1::4]
            
            if (xx < lower_bound).any() or (xx > upper_bound).any() or \
               (yy < lower_bound).any() or (yy > upper_bound).any():
                continue
                
            return jnp.array(tracks)

    def generate_measurements(self, key, tracks):
        """
        Generates measurements from the tracks with added Gaussian noise.
        """
        # Vectorize measurement function over the time dimension
        h_vmap = jax.vmap(self.h)
        z_clean = h_vmap(tracks)
        
        R_real = self.measvar_real * jnp.eye(self.nSensor)
        
        key, subkey = jax.random.split(key)
        noise = jax.random.multivariate_normal(
            subkey, mean=jnp.zeros(z_clean.shape[1]), cov=R_real, shape=(tracks.shape[0],)
        )
        
        return z_clean + noise


def main():
    # 1. Setup
    print("Initializing Acoustic Tracking configuration...")
    tracker = AcousticTracking()
    key = jax.random.PRNGKey(42)
    
    # 2. Generate Data
    print("Generating tracks and measurements...")
    key, subkey1, subkey2 = jax.random.split(key, 3)
    tracks_gt = tracker.generate_tracks(subkey1)
    measurements = tracker.generate_measurements(subkey2, tracks_gt)
    
    print(f"Tracks shape: {tracks_gt.shape}")
    print(f"Measurements shape: {measurements.shape}")
    
    # 3. Setup Particle Filter
    num_particles = 500
    print(f"Setting up Particle Filter with {num_particles} particles...")
    
    R_filter = tracker.measvar * jnp.eye(tracker.nSensor)
    
    pf = ParticleFilter(
        f=tracker.f,
        h=tracker.h,
        num_particles=num_particles,
        Q=tracker.Q,
        R=R_filter,
        resample_threshold=0.5
    )
    
    # Initialize particles
    key, subkey = jax.random.split(key)
    initial_particles = jax.random.multivariate_normal(
        subkey, mean=tracker.x0, cov=tracker.sigma0, shape=(num_particles,)
    )
    initial_weights = jnp.ones(num_particles) / num_particles
    
    pf_state = ParticleState(
        particles=initial_particles,
        weights=initial_weights,
        key=key
    )
    
    # 4. Filtering Loop
    print("Starting filtering loop...")
    start_time = time.time()
    
    estimated_states = []
    
    @jax.jit
    def pf_step(state, z):
        state = pf.predict(state)
        state = pf.update(state, z)
        return state

    for t in range(tracker.T):
        z = measurements[t]
        pf_state = pf_step(pf_state, z)
        estimated_states.append(pf_state.mean)
        
        if (t + 1) % 10 == 0:
            print(f"Processed step {t + 1}/{tracker.T}")
            
    estimated_states = jnp.array(estimated_states)
    end_time = time.time()
    
    print(f"Filtering complete in {end_time - start_time:.2f} seconds.")
    
    # Calculate RMSE for the first target's position (x, y)
    err_x = estimated_states[:, 0] - tracks_gt[:, 0]
    err_y = estimated_states[:, 1] - tracks_gt[:, 1]
    rmse_t1 = jnp.sqrt(jnp.mean(err_x**2 + err_y**2))
    print(f"RMSE for Target 1 Position: {rmse_t1:.4f}")

if __name__ == "__main__":
    main()
