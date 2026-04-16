# JAX EKF/UKF Toolbox

A modern, highly-performant Python library for linear and nonlinear Kalman filtering and smoothing. This library is a complete port and reimagining of the classic MATLAB `ekfukf` toolbox, rewritten from the ground up using [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

By leveraging JAX, this library supports:
* **JIT Compilation (`jax.jit`)**: Filters and smoothers are blazing fast and compile down to XLA.
* **Automatic Differentiation (`jax.jacfwd`)**: The Extended Kalman Filter (EKF) no longer requires user-provided analytical Jacobians; they are computed automatically.
* **Hardware Acceleration**: Run your filtering pipelines on CPU, GPU, or TPU seamlessly.
* **Vectorized Sequence Processing**: Batch processing of time-series data uses `jax.lax.scan` to execute the full forward-backward sweeps entirely inside compiled XLA loops.

## Available Filters & Smoothers

Every filter comes with its corresponding Rauch-Tung-Striebel (RTS) backward smoother.

* **KF**: Linear Kalman Filter
* **EKF**: Extended Kalman Filter (Auto-diff Jacobians)
* **UKF**: Unscented Kalman Filter
* **CKF**: Cubature Kalman Filter
* **GHKF**: Gauss-Hermite Kalman Filter
* **IMM**: Interacting Multiple Model Filter & Smoother (for combining arbitrary sub-filters)

## Architecture

The framework relies on two primary abstractions:

1. **`GaussianState`**: An Equinox Module containing the `mean` and `cov` of the system.
2. **`AbstractFilter`**: The base class for all filters. It provides:
   * `.predict(state, u, params)`: Single-step prediction.
   * `.update(state, y, params)`: Single-step measurement update.
   * `.filter_sequence(initial_state, Y, U, Params)`: Unrolls the prediction and update loops over an entire sequence using `jax.lax.scan`.
   * `.smooth_sequence(history_update, history_predict, U, Params)`: Runs the backward RTS smoothing pass over the filtered sequence.

## Basic Usage

Here is a simple example demonstrating how to construct and use the Extended Kalman Filter (EKF).

```python
import jax.numpy as jnp
from jax_ekfukf import ExtendedKalmanFilter, GaussianState

# 1. Define your nonlinear dynamics and measurement functions
def f(x, u):
    # e.g., coordinated turn model, pendulum, etc.
    return jnp.array([x[0] + 0.1 * x[1], x[1] - 0.1 * x[0]]) + u

def h(x):
    # e.g., polar coordinate measurements
    return jnp.array([jnp.sqrt(x[0]**2 + x[1]**2)])

# 2. Define covariances
Q = jnp.eye(2) * 0.01  # Process noise
R = jnp.eye(1) * 0.1   # Measurement noise

# 3. Initialize the filter
ekf = ExtendedKalmanFilter(f=f, h=h, Q=Q, R=R)

# 4. Create an initial state
init_state = GaussianState(mean=jnp.array([1.0, 0.0]), cov=jnp.eye(2))

# 5. Define data (batch dimension first)
Y = jnp.ones((100, 1)) # 100 measurements of dim 1
U = jnp.zeros((100, 2)) # 100 control inputs of dim 2

# 6. Run the forward filter over the entire sequence
final_state, history_pred, history_upd = ekf.filter_sequence(init_state, Y, U)

# 7. Run the backward smoother
init_smooth, history_smooth = ekf.smooth_sequence(history_upd, history_pred, U)

# history_smooth.mean contains the smoothed sequence of shape (100, 2)
```

## Interacting Multiple Models (IMM)

The IMM filter allows you to run a bank of filters simultaneously to track targets or states that switch between different dynamic models.

```python
from jax_ekfukf import IMMFilter, IMMState

# Create your sub-filters (can be combinations of KF, EKF, UKF, etc.)
filter1 = KalmanFilter(A=A1, H=H, Q=Q1, R=R)
filter2 = ExtendedKalmanFilter(f=f_nonlin, h=h_nonlin, Q=Q2, R=R)

# Define Markov transition probabilities between models
p_ij = jnp.array([
    [0.98, 0.02],
    [0.05, 0.95]
])

imm = IMMFilter(filters=(filter1, filter2), transition_matrix=p_ij)

# initial model probabilities
mu_0 = jnp.array([0.9, 0.1])
init_imm_state = IMMState(states=(state1, state2), mu=mu_0)

final_state, hist_pred, hist_upd = imm.filter_sequence(init_imm_state, Y)
```

## Particle Filter (PF)
The `ParticleFilter` implements a standard Bootstrap/SIR Filter with systematic resampling conditioned on the Effective Sample Size (ESS). 
It relies on a newly introduced `ParticleState` which tracks a JAX `PRNGKey` internally to support deterministic PRNG splits during `jax.lax.scan` operations.

```python
from jax_ekfukf import ParticleFilter, ParticleState
import jax.random

num_particles = 1000
pf = ParticleFilter(
    f=f, 
    h=h, 
    num_particles=num_particles,
    Q=Q, 
    R=R,
    resample_threshold=num_particles / 2.0 # Resample when ESS < 500
)

# Particle state initialization
key = jax.random.PRNGKey(0)
initial_particles = jax.random.multivariate_normal(key, mean=jnp.zeros(2), cov=jnp.eye(2), shape=(num_particles,))
initial_weights = jnp.ones(num_particles) / num_particles

init_state = ParticleState(
    particles=initial_particles,
    weights=initial_weights,
    key=key
)

final_state, hist_pred, hist_upd = pf.filter_sequence(init_state, Y)
```
*Note: Particle smoothing (`smooth_sequence`) is highly computationally expensive and is not currently implemented. `ParticleFilter.smooth_sequence()` will raise a `NotImplementedError`.*

## Particle Flow Filters (EDH & LEDH)
Particle Flow Filters seamlessly migrate particles from the prior distribution to the posterior distribution by integrating an Ordinary Differential Equation (ODE), avoiding the need for weight-based resampling entirely.

The **Exact Daum-Huang (EDH)** filter assumes a global Gaussian prior to compute the continuous flow ODE. However, highly non-Gaussian or multi-modal priors cause EDH to diverge. 

To resolve this, the **Localized Exact Daum-Huang (LEDH)** filter estimates unique local covariance matrices for every particle. The library includes three localization variants:
1. `LEDHKNNFilter`: Uses K-Nearest Neighbors (fastest, most robust for JAX `lax.scan`).
2. `LEDHKDEFilter`: Uses Gaussian Kernel Density Estimation (KDE) with a configurable bandwidth $h$.
3. `LEDHGMMFilter`: Uses Gaussian Mixture Models (Hard-EM/K-Means) to group particles into $M$ components.

```python
from jax_ekfukf import EDHFilter, LEDHKNNFilter, ParticleState

num_particles = 200

# 1. The Global EDH Filter
edh = EDHFilter(
    f=f, 
    h=h, 
    num_particles=num_particles,
    Q=Q, 
    R=R,
    flow_steps=20 # ODE integration steps
)

# 2. The Localized LEDH Filters (choose your localization strategy)
from jax_ekfukf import LEDHKNNFilter, LEDHKDEFilter, LEDHGMMFilter

ledh_knn = LEDHKNNFilter(f=f, h=h, num_particles=N, Q=Q, R=R, knn_fraction=0.1)
ledh_kde = LEDHKDEFilter(f=f, h=h, num_particles=N, Q=Q, R=R, bandwidth=2.0)
ledh_gmm = LEDHGMMFilter(f=f, h=h, num_particles=N, Q=Q, R=R, num_components=3)

# Usage is identical to the standard Particle Filter:
final_state, hist_pred, hist_upd = ledh_knn.filter_sequence(init_state, Y)
```
