# JAX/Equinox Migration Plan for ekfukf Toolkit

## Core Architecture & Philosophy
1. **State as PyTrees:** Encapsulate state in an Equinox module (essentially a JAX-compatible dataclass) called `GaussianState(eqx.Module)`.
2. **Filters as Equinox Modules:** Each filter will be an `eqx.Module` base class. This enables storing static parameters while exposing them correctly to JAX's JIT compiler.
3. **No Manual Jacobians:** For Extended Kalman Filters (EKF), we will use `jax.jacfwd` and `jax.jacrev` to compute Jacobians automatically.
4. **Time Series Vectorization (`jax.lax.scan`):** Provide a `filter_sequence()` method that uses `jax.lax.scan` for drastically faster, compiled execution over time steps.

## Dual-Mode Strategy (Online vs Offline)
1. **The Core API (Single-Step):** The base `eqx.Module` will define `predict` and `update` functions operating strictly on a single time step. Decorated with `@eqx.filter_jit`, they can be used effortlessly in real-time callbacks.
2. **The Batch API (Vectorized Sequence):** For post-analysis, a `filter_sequence` method will wrap the single-step methods inside a `jax.lax.scan`. This avoids Python `for` loops and compiles the entire sequence execution directly in XLA.
3. **Time-varying parameters:** All single-step methods and sequences will accept a generic `params` argument (or sequence of `Params`) to allow passing time-varying matrices (like $A_k, Q_k, R_k$) without recompilation.

## Phased Implementation

### Phase 1: Base Structures and Utilities
*   Create `GaussianState` and `AbstractFilter`.
*   Implement standard `filter_sequence` loop over `jax.lax.scan`.
*   Port utilities like `lti_disc` (using `jax.scipy.linalg.expm`) and `rk4` integrator.

### Phase 2: Linear & Extended Kalman Filters (KF & EKF)
*   **KF:** Maps directly to standard Matrix operations (`jnp.dot`, `jnp.linalg.solve`).
*   **EKF:** Uses pure python functions $f(x, u)$ and $h(x)$, evaluating Jacobians dynamically with JAX autodiff.

### Phase 3: Sigma-Point Filters (UKF, CKF, GHKF)
*   Consolidate `ut_transform`, `ckf_transform`, etc., into a generic `transforms.py` module.
*   Use `jax.vmap` to map the nonlinear function $f(x)$ over all sigma points simultaneously.
*   Precompute static weights.

### Phase 4: Interacting Multiple Model (IMM) & Smoothers
*   **IMM:** An `IMMFilter` wrapping a tuple of filters, applying `jax.vmap` or functional comprehension across the bank.
*   **Smoothers:** Introduce a `Smoother` mixin that takes the forward-pass history and runs `jax.lax.scan(reverse=True)` for efficient RTS smoothing.

## Testing Strategy
To verify correctness against the original `ekfukf` MATLAB implementation, all Python module equivalents must be thoroughly tested against the logic and dataset generation used in the MATLAB `demos/` directory.

### Completed Tests
*   **Linear Kalman Filter (KF):** Verified via `test_kf.py`. Uses `demos/kf_sine_demo.m` (tracking a linear sine wave).
*   **Extended Kalman Filter (EKF):** Verified via `test_ekf.py`. Uses `demos/ekf_sine_demo.m` (tracking a random single-component sinusoid, $x_k = a_k \cdot \sin(\theta_k)$).

### Planned Tests for Future Implementations
1. **Unscented Kalman Filter (UKF):**
    *   **Target Demo:** Replicate the UKF section of `demos/ekf_sine_demo.m`. Ensure the Unscented Transform (UT) weights and sigma points match MATLAB precisely.
2. **Cubature & Gauss-Hermite Kalman Filters (CKF, GHKF):**
    *   **Target Demo:** Replicate logic from `demos/ungm_demo.m` (Univariate Non-stationary Growth Model) to benchmark the exact transformations against the EKF/UKF.
3. **Interacting Multiple Model (IMM):**
    *   **Target Demo:** Use `demos/imm_demo.m` and `demos/eimm_demo.m`. The model must successfully filter the piecewise trajectory by switching between a constant velocity (CV) and a coordinated turn (CT) model.
4. **Smoothers (RTS, URTS, ERTS):**
    *   **Target Strategy:** Every single filter test above (KF, EKF, UKF) should be augmented with a smoothing pass. The RMSE of the smoothed sequence must strictly be lower than the forward-filtered RMSE (e.g., in `test_kf.py`, RTS smoothed sequence vs KF sequence).

---

**Initial Steps:** Start by implementing Phase 1 and the Phase 2 Linear KF.