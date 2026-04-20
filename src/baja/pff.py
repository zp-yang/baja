import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any, Tuple

from .base import AbstractFilter
from .pf import ParticleState


class EDHFilter(AbstractFilter):
    """
    Exact Daum-Huang (EDH) Particle Flow Filter.

    Propagates particles continuously from the prior to the posterior
    using an ODE over pseudo-time lambda, assuming a global Gaussian prior covariance.
    """

    f: Callable
    h: Callable
    Q: Optional[jax.Array] = None
    R: Optional[jax.Array] = None
    num_particles: int = eqx.field(static=True)
    flow_steps: int = eqx.field(static=True)

    def __init__(
        self,
        f: Callable,
        h: Callable,
        num_particles: int,
        Q: Optional[jax.Array] = None,
        R: Optional[jax.Array] = None,
        flow_steps: int = 20,
    ):
        self.f = f
        self.h = h
        self.Q = jnp.asarray(Q) if Q is not None else None
        self.R = jnp.asarray(R) if R is not None else None
        self.num_particles = num_particles
        self.flow_steps = flow_steps

    @eqx.filter_jit
    def predict(
        self, state: ParticleState, u: Optional[jax.Array] = None, params: Any = None
    ) -> ParticleState:
        """
        Prediction step: propagates particles through the dynamic model.
        """
        Q = params.Q if params is not None and hasattr(params, "Q") else self.Q
        assert Q is not None, "Process noise covariance Q must be provided."

        key, subkey = jax.random.split(state.key)

        if u is not None and u.size > 0:
            f_vmap = jax.vmap(lambda x: self.f(x, u))
        else:
            f_vmap = jax.vmap(self.f)

        particles_next = f_vmap(state.particles)

        state_dim = state.particles.shape[1]
        noise = jax.random.multivariate_normal(
            subkey, mean=jnp.zeros(state_dim), cov=Q, shape=(self.num_particles,)
        )

        particles_pred = particles_next + noise

        # In PFF, weights are kept uniform
        weights = jnp.ones(self.num_particles) / self.num_particles

        return ParticleState(particles=particles_pred, weights=weights, key=key)

    def _compute_covariances(self, particles: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the global empirical covariance of the particles.
        Returns a single covariance matrix of shape (state_dim, state_dim).
        """
        mean = jnp.mean(particles, axis=0)
        diff = particles - mean
        # Unbiased empirical covariance
        P = (diff.T @ diff) / (self.num_particles - 1.0)
        return P

    @eqx.filter_jit
    def update(
        self,
        state: ParticleState,
        z: jax.Array,
        params: Any = None,
        return_likelihood: bool = False,
    ):
        """
        Update step: migrates particles using the EDH flow equation.
        """
        R = params.R if params is not None and hasattr(params, "R") else self.R
        assert R is not None, "Measurement noise covariance R must be provided."

        P = self._compute_covariances(state.particles)

        I = jnp.eye(state.particles.shape[-1])
        eta_bar_0 = state.mean

        def flow_step(carry, step):
            """
            Linearize measurement function around mean particle state, eta_bar_lam

            Flow equation: deta/dlam = A(lam)eta + b(lam)

            A = -1/2 * P * H^T * (R + lam * H * P * H^T)^(-1) * H
            b = (I + 2lam * A) * ((I + lam * A) * P * H^T * R^(-1) * (z - e(lam)) + A * eta_bar_0)

            """
            particles = carry
            eta_bar = jnp.mean(particles, axis=0)

            lam = (1 - 1.2**step) / (1 - 1.2**self.flow_steps)
            dlam = lam - (1 - 1.2 ** (step - 1)) / (1 - 1.2**self.flow_steps)

            h_lam = self.h(eta_bar)
            H_lam = jax.jacfwd(self.h)(eta_bar)

            S_lam = R + lam * (H_lam @ P @ H_lam.T)
            A_lam = -1 / 2 * P @ H_lam.T @ jnp.linalg.solve(S_lam, H_lam)
            e_lam = h_lam - H_lam @ eta_bar
            b_lam = (I + 2 * lam * A_lam) @ (
                (I + lam * A_lam) @ P @ H_lam.T @ jnp.linalg.inv(R) @ (z - e_lam)
                + A_lam @ eta_bar_0
            )

            def single_particle_flow(particle):
                dx_dlam = A_lam @ particle + b_lam
                return dx_dlam

            dx_dlam = jax.vmap(single_particle_flow)(particles)

            particles_next = particles + dlam * dx_dlam
            return particles_next, None


        steps = jnp.arange(1, self.flow_steps)
        final_particles, _ = jax.lax.scan(flow_step, state.particles, steps)

        # PFF doesn't reweight particles (or weights are kept uniform)
        updated_state = ParticleState(
            particles=final_particles, weights=state.weights, key=state.key
        )

        if return_likelihood:
            # Likelihood for PFF is complex to compute exactly without full density estimation.
            # As a proxy, we can return the likelihood of the prior mean, or just 1.0.
            # For simplicity, returning a dummy likelihood if requested.
            mean_pred = jnp.mean(state.particles, axis=0)
            z_pred = self.h(mean_pred)
            S = jax.jacfwd(self.h)(mean_pred) @ P @ jax.jacfwd(self.h)(mean_pred).T + R
            ll = jax.scipy.stats.multivariate_normal.logpdf(z, mean=z_pred, cov=S)
            likelihood = jnp.exp(ll)
            return updated_state, likelihood

        return updated_state

    def smooth_step(self, *args, **kwargs):
        raise NotImplementedError("Particle smoothing is not supported.")

    def smooth_sequence(self, *args, **kwargs):
        raise NotImplementedError("Particle smoothing is not supported.")


class LEDHKNNFilter(EDHFilter):
    """
    Localized Exact Daum-Huang (LEDH) Particle Flow Filter using K-Nearest Neighbors.

    Computes a localized covariance matrix P_i for each particle using its k nearest
    neighbors, allowing the flow to handle multi-modal or highly non-Gaussian priors.
    """

    knn_fraction: float = eqx.field(static=True)

    def __init__(
        self,
        f: Callable,
        h: Callable,
        num_particles: int,
        Q: Optional[jax.Array] = None,
        R: Optional[jax.Array] = None,
        flow_steps: int = 20,
        knn_fraction: float = 0.1,
    ):
        super().__init__(f, h, num_particles, Q, R, flow_steps)
        self.knn_fraction = knn_fraction

    def _compute_covariances(self, particles: jnp.ndarray) -> jnp.ndarray:
        """
        Computes localized covariances using K-Nearest Neighbors.
        Returns a batched covariance array of shape (num_particles, state_dim, state_dim).
        """
        N = self.num_particles
        state_dim = particles.shape[1]

        # Number of neighbors (at least 2 to compute covariance)
        k = max(2, int(N * self.knn_fraction))
        k = min(k, N)

        # Pairwise squared distances (N x N)
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i^T x_j
        sq_norms = jnp.sum(particles**2, axis=1)
        dist_matrix = (
            sq_norms[:, None] + sq_norms[None, :] - 2 * (particles @ particles.T)
        )

        # We want the k SMALLEST distances. top_k returns the LARGEST, so we negate.
        # values, indices = jax.lax.top_k(-dist_matrix, k)
        # indices shape: (N, k)
        _, indices = jax.lax.top_k(-dist_matrix, k)

        # Gather neighbors for each particle
        # neighbors shape: (N, k, state_dim)
        neighbors = particles[indices]

        # Compute local means
        # local_means shape: (N, state_dim)
        local_means = jnp.mean(neighbors, axis=1)

        # Compute local covariances
        # diffs shape: (N, k, state_dim)
        diffs = neighbors - local_means[:, None, :]

        # batched outer product: (N, k, state_dim, 1) * (N, k, 1, state_dim) -> (N, k, state_dim, state_dim)
        # Then sum over k to get (N, state_dim, state_dim)
        covs = jnp.sum(diffs[:, :, :, None] * diffs[:, :, None, :], axis=1) / (k - 1.0)

        # Add a tiny epsilon to the diagonal for numerical stability
        covs = covs + jnp.eye(state_dim) * 1e-6

        return covs


class LEDHKDEFilter(EDHFilter):
    """
    Localized Exact Daum-Huang (LEDH) Particle Flow Filter using Kernel Density Estimation.

    Computes a localized covariance matrix P_i for each particle by weighting all other
    particles using a Gaussian kernel based on Euclidean distance.
    """

    bandwidth: float = eqx.field(static=True)

    def __init__(
        self,
        f: Callable,
        h: Callable,
        num_particles: int,
        Q: Optional[jax.Array] = None,
        R: Optional[jax.Array] = None,
        flow_steps: int = 20,
        bandwidth: float = 1.0,
    ):
        super().__init__(f, h, num_particles, Q, R, flow_steps)
        self.bandwidth = bandwidth

    def _compute_covariances(self, particles: jnp.ndarray) -> jnp.ndarray:
        """
        Computes localized covariances using KDE weights.
        Returns a batched covariance array of shape (num_particles, state_dim, state_dim).
        """
        state_dim = particles.shape[1]

        # Pairwise squared distances (N x N)
        sq_norms = jnp.sum(particles**2, axis=1)
        dist_matrix = (
            sq_norms[:, None] + sq_norms[None, :] - 2 * (particles @ particles.T)
        )

        # Kernel weights: W_ij = exp(-D_ij / (2 * h^2))
        log_weights = -dist_matrix / (2.0 * self.bandwidth**2)

        # Normalize weights safely over axis 1
        max_log_w = jnp.max(log_weights, axis=1, keepdims=True)
        exp_w = jnp.exp(log_weights - max_log_w)
        W = exp_w / jnp.sum(exp_w, axis=1, keepdims=True)

        # Local means: m_i = sum_j W_ij * x_j (Shape: N x state_dim)
        local_means = W @ particles

        # Differences: diff_ij = x_j - m_i (Shape: N x N x state_dim)
        diffs = particles[None, :, :] - local_means[:, None, :]

        # Local covariances: P_i = sum_j W_ij * diff_ij * diff_ij^T
        # W[:, :, None, None] * (N, N, state_dim, 1) @ (N, N, 1, state_dim) -> (N, N, state_dim, state_dim)
        outer_prods = diffs[:, :, :, None] * diffs[:, :, None, :]
        covs = jnp.sum(W[:, :, None, None] * outer_prods, axis=1)

        # Regularization for numerical stability
        covs = covs + jnp.eye(state_dim) * 1e-6

        return covs


class LEDHGMMFilter(EDHFilter):
    """
    Localized Exact Daum-Huang (LEDH) Particle Flow Filter using a Gaussian Mixture Model.

    Fits a GMM to the particles at each step and assigns the local covariance
    for particle i to be the covariance of the component it belongs to.
    """

    num_components: int = eqx.field(static=True)
    em_iterations: int = eqx.field(static=True)

    def __init__(
        self,
        f: Callable,
        h: Callable,
        num_particles: int,
        Q: Optional[jax.Array] = None,
        R: Optional[jax.Array] = None,
        flow_steps: int = 20,
        num_components: int = 5,
        em_iterations: int = 5,
    ):
        super().__init__(f, h, num_particles, Q, R, flow_steps)
        self.num_components = num_components
        self.em_iterations = em_iterations

    def _compute_covariances(self, particles: jnp.ndarray) -> jnp.ndarray:
        """
        Fits a simple K-Means GMM and returns the batched local covariance array of shape
        (num_particles, state_dim, state_dim). K-Means is much more numerically stable
        in compiled JAX loops than soft EM, preventing covariance collapse.
        """
        N = self.num_particles
        M = self.num_components
        state_dim = particles.shape[1]

        # Initialize cluster means using first M particles
        init_mu = particles[:M]

        # Initial global covariance
        global_mean = jnp.mean(particles, axis=0)
        global_diff = particles - global_mean
        global_cov = (global_diff.T @ global_diff) / (N - 1.0) + jnp.eye(
            state_dim
        ) * 1e-6

        def kmeans_step(mu, _):
            # Compute pairwise distances
            sq_norms_p = jnp.sum(particles**2, axis=1)
            sq_norms_m = jnp.sum(mu**2, axis=1)
            dist_matrix = (
                sq_norms_p[:, None] + sq_norms_m[None, :] - 2 * (particles @ mu.T)
            )

            # Hard assignments
            assignments = jnp.argmin(dist_matrix, axis=1)  # (N,)

            # Update means
            # To do this safely, we use jax.ops.segment_sum
            sum_mu = jax.ops.segment_sum(particles, assignments, num_segments=M)
            counts = jax.ops.segment_sum(jnp.ones((N, 1)), assignments, num_segments=M)

            # Avoid division by zero
            safe_counts = jnp.maximum(counts, 1.0)
            new_mu = jnp.where(counts > 0, sum_mu / safe_counts, mu)

            return new_mu, assignments

        # Run K-Means iterations
        final_mu, assignments_history = jax.lax.scan(
            kmeans_step, init_mu, None, length=self.em_iterations
        )

        final_assignments = assignments_history[-1]  # (N,)

        # Now compute the covariances for each cluster
        def compute_cluster_cov(c):
            # Mask of particles in cluster c
            mask = (final_assignments == c)[:, None]  # (N, 1)
            count = jnp.sum(mask)

            # Mean is already final_mu[c]
            diff = particles - final_mu[c][None, :]

            # Weighted outer product
            # (N, state_dim, 1) * (N, 1, state_dim) -> (N, state_dim, state_dim)
            outer = diff[:, :, None] * diff[:, None, :]

            # Sum over particles in cluster
            cov_sum = jnp.sum(mask[:, :, None] * outer, axis=0)

            # Return covariance, fallback to global if empty or 1 particle
            return jnp.where(
                count > 1,
                cov_sum / (count - 1.0) + jnp.eye(state_dim) * 1e-2,
                global_cov,
            )

        cluster_covs = jax.vmap(compute_cluster_cov)(
            jnp.arange(M)
        )  # (M, state_dim, state_dim)

        # Gather covariances for each particle
        local_covs = cluster_covs[final_assignments]  # (N, state_dim, state_dim)

        return local_covs
