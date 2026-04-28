import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any, Tuple

from .base import AbstractFilter, GaussianState
from .pf import ParticleState
from .ekf import ExtendedKalmanFilter
from dataclasses import dataclass


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

        return updated_state

    @eqx.filter_jit
    def run_step(
        self,
        state: ParticleState,
        z: jax.Array,
        u: Optional[jax.Array] = None,
        params: Any = None,
    ):
        p_pred = self.predict(state, u, params)
        p_update = self.update(p_pred, z, params)
        return p_update

    def smooth_step(self, *args, **kwargs):
        raise NotImplementedError("Particle smoothing is not supported.")

    def smooth_sequence(self, *args, **kwargs):
        raise NotImplementedError("Particle smoothing is not supported.")


@dataclass
class PFPFParam(eqx.Module):
    """
    Holds the internal state of PFPF filters
    EKF/UKF predict and update are apply before and after homotopy flow
    means are the particles (n_state) or (n_particles, n_state),
    covs are per particle (n_state, n_state) or (n_particles, n_state, n_state)
    """

    internal_state: GaussianState


class IEDHFilter(AbstractFilter):
    """
    Particle flow filter with invertible flow, built on EDH.
    Use invertible mapping property to perform efficient weight updates
    see Li & Coates, Particle Filtering with Invertible Particle Flow

    Propagates particles continuously from the prior to the posterior
    using an ODE over pseudo-time lambda, assuming a global Gaussian prior covariance.
    """

    f: Callable
    h: Callable
    internal_filter: AbstractFilter
    Q: Optional[jax.Array] = None
    R: Optional[jax.Array] = None
    num_particles: int = eqx.field(static=True)
    flow_steps: int = eqx.field(static=True)

    def __init__(
        self,
        f: Callable,
        h: Callable,
        internal_filter: AbstractFilter,
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
        self.internal_filter = internal_filter

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
    def run_step(
        self,
        state: ParticleState,
        z: jax.Array,
        u: Optional[jax.Array] = None,
        params: Tuple = None,
    ):
        """
        state: particles from previous step
        z: measurement at current step
        u: optional control signal, should be None for tracking
        params: optional time varying parameters, should be none here
        """

        """
        ================================================================
        Prediction step: propagates particles through the dynamic model.
        ================================================================
        """
        Q = params.Q if params is not None and hasattr(params, "Q") else self.Q
        assert Q is not None, "Process noise covariance Q must be provided."

        if u is not None and u.size > 0:
            f_vmap = jax.vmap(lambda x: self.f(x, u))
        else:
            f_vmap = jax.vmap(self.f)

        particles_pred_clean = f_vmap(state.particles)

        state_dim = state.particles.shape[1]

        key, subkey = jax.random.split(state.key)
        noise = jax.random.multivariate_normal(
            subkey, mean=jnp.zeros(state_dim), cov=Q, shape=(self.num_particles,)
        )

        particles_pred = particles_pred_clean + noise

        internal_state = params.internal_state
        internal_state_pred = self.internal_filter.predict(internal_state)

        """
        ================================================================
        Update step: homotopy flow and weigth update
        ================================================================
        """
        R = params.R if params is not None and hasattr(params, "R") else self.R
        assert R is not None, "Measurement noise covariance R must be provided."

        P = internal_state_pred.cov

        I = jnp.eye(state.particles.shape[-1])  # match identity mat shape to state dim
        eta_bar_0 = jnp.mean(particles_pred_clean, axis=0)  # shape (state_dim,)

        """
        =========================
        Exact homotopy flow
        =========================
        """

        def flow_step(carry, step):
            """
            Linearize measurement function around mean particle state, eta_bar_lam
            Here we use lax.scan to carry out the loop

            Flow equation: deta/dlam = A(lam)eta + b(lam)

            A = -1/2 * P * H^T * (R + lam * H * P * H^T)^(-1) * H
            b = (I + 2lam * A) * ((I + lam * A) * P * H^T * R^(-1) * (z - e(lam)) + A * eta_bar_0)
            """
            particles, eta_bar = carry
            # eta_bar = jnp.mean(particles, axis=0)

            # exponential steps over lambda, with q=1.2
            q = 1.2
            lam = (1 - q**step) / (1 - q**self.flow_steps)
            dlam = lam - (1 - q ** (step - 1)) / (1 - q**self.flow_steps)

            h_lam = self.h(eta_bar)
            H_lam = jax.jacfwd(self.h)(eta_bar)  # linearize h

            S_lam = R + lam * (H_lam @ P @ H_lam.T)
            A_lam = -1 / 2 * P @ H_lam.T @ jnp.linalg.solve(S_lam, H_lam)
            e_lam = h_lam - H_lam @ eta_bar
            b_lam = (I + 2 * lam * A_lam) @ (
                (I + lam * A_lam) @ P @ H_lam.T @ jnp.linalg.inv(R) @ (z - e_lam)
                + A_lam @ eta_bar_0
            )

            eta_bar_next = eta_bar + dlam * (A_lam @ eta_bar + b_lam)

            def single_particle_flow(particle):
                dx_dlam = A_lam @ particle + b_lam
                return dx_dlam

            dx_dlam = jax.vmap(single_particle_flow)(particles)

            particles_next = particles + dlam * dx_dlam
            return (particles_next, eta_bar_next), None

        steps = jnp.arange(1, self.flow_steps)
        (final_particles, _), _ = jax.lax.scan(
            flow_step, (particles_pred, eta_bar_0), steps
        )

        # Evaluate measurement function for all particles
        h_vmap = jax.vmap(self.h)
        z_pred = h_vmap(final_particles)

        ## Weight update
        # w_k -> prior * likelihood / proposal * w_(k-1)
        # log_w_k -> log_prior(process) + meas_log_likelihood - log_proposal + log_w_(k-1)
        # Calculate log likelihoods
        log_likelihoods = jax.vmap(
            lambda y: jax.scipy.stats.multivariate_normal.logpdf(
                y, mean=jnp.zeros(R.shape[0]), cov=R
            )
        )(z_pred - z)

        # xp_prop -> predicted particles
        # xp_prop_deterministic -> predicted particles without process noise
        log_proposal = jax.scipy.stats.multivariate_normal.logpdf(
            particles_pred, mean=particles_pred_clean, cov=Q
        )

        log_prior = jax.scipy.stats.multivariate_normal.logpdf(
            final_particles, mean=particles_pred_clean, cov=Q
        )

        # Algo 2 line 21 - 24
        log_weight = (
            log_prior  # process distribution p(x_k | x_(k-1))
            + log_likelihoods  # p(z_k | x_k)
            - log_proposal  # p(eta_0 | x_(k-1))
            + jnp.log(jnp.maximum(state.weights, 1e-20))  # w_(k-1)
        )  # avoid log(0)

        weights = jnp.exp(log_weight - jnp.max(log_weight))
        # weights = jnp.exp(log_weight)
        norm_weights = weights / jnp.sum(weights)

        updated_state = ParticleState(
            particles=final_particles, weights=norm_weights, key=key
        )

        # Algo 2 line 26
        internal_state_update = self.internal_filter.update(internal_state_pred, z)
        internal_state = GaussianState(
            mean=updated_state.mean, cov=internal_state_update.cov
        )
        return updated_state, PFPFParam(internal_state)


class ILEDHFilter(AbstractFilter):
    """
    Particle flow filter with invertible flow, built on Localized EDH.
    Use invertible mapping property to perform efficient weight updates
    see Li & Coates, Particle Filtering with Invertible Particle Flow
    """

    f: Callable
    h: Callable
    Q: Optional[jax.Array] = None
    R: Optional[jax.Array] = None
    internal_filter: AbstractFilter
    num_particles: int = eqx.field(static=True)
    flow_steps: int = eqx.field(static=True)

    def __init__(
        self,
        f: Callable,
        h: Callable,
        internal_filter: AbstractFilter,
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
        self.internal_filter = internal_filter

    @eqx.filter_jit
    def run_step(
        self,
        state: ParticleState,
        z: jax.Array,
        u: Optional[jax.Array] = None,
        params: PFPFParam = None,
    ):
        """
        state: particles from previous step
        z: measurement at current step
        u: optional control signal, should be None for tracking
        params: optional time varying parameters, should be none here
        """

        """
        ================================================================
        Prediction step: propagates particles through the dynamic model.
        ================================================================
        """
        Q = params.Q if params is not None and hasattr(params, "Q") else self.Q
        assert Q is not None, "Process noise covariance Q must be provided."

        if u is not None and u.size > 0:
            f_vmap = jax.vmap(lambda x: self.f(x, u))
        else:
            f_vmap = jax.vmap(self.f)

        particles_pred_clean = f_vmap(state.particles)

        state_dim = state.particles.shape[1]

        key, subkey = jax.random.split(state.key)
        noise = jax.random.multivariate_normal(
            subkey, mean=jnp.zeros(state_dim), cov=Q, shape=(self.num_particles,)
        )

        particles_pred = particles_pred_clean + noise

        # predict per particle covariance from internal filter
        internal_state_pred = jax.vmap(self.internal_filter.predict)(params.internal_state)

        eta_0 = particles_pred
        eta_bar_0 = particles_pred_clean
        """
        ================================================================
        Update step: homotopy flow and weigth update
        ================================================================
        """
        R = params.R if params is not None and hasattr(params, "R") else self.R
        assert R is not None, "Measurement noise covariance R must be provided."

        # P = self._compute_covariances(particles_pred)
        Ps = internal_state_pred.cov

        I = jnp.eye(state.particles.shape[-1])  # match identity mat shape to state dim

        """
        =========================
        invertibel flow
        =========================
        """

        def flow_step(carry, step):
            # [prop_with_noise, prop_no_noise, log_jac_det_sum]
            eta, eta_bar, log_theta = carry

            # exponential steps over lambda, with q=1.2
            lam = (1 - 1.2**step) / (1 - 1.2**self.flow_steps)
            dlam = lam - (1 - 1.2 ** (step - 1)) / (1 - 1.2**self.flow_steps)

            def single_particle_flow(eta_i, eta_bar_i, eta_bar_0_i, Pi, log_theta_i):
                h_lam = self.h(eta_bar_i)
                H_lam = jax.jacfwd(self.h)(eta_bar_i)  # linearize h

                S_lam = R + lam * (H_lam @ Pi @ H_lam.T)
                A_lam = -1 / 2 * Pi @ H_lam.T @ jnp.linalg.solve(S_lam, H_lam)
                e_lam = h_lam - H_lam @ eta_bar_i
                b_lam = (I + 2 * lam * A_lam) @ (
                    (I + lam * A_lam) @ Pi @ H_lam.T @ jnp.linalg.inv(R) @ (z - e_lam)
                    + A_lam @ eta_bar_0_i
                )

                eta_bar_i_next = eta_bar_i + dlam * (A_lam @ eta_bar_i + b_lam)

                eta_i_next = eta_i + dlam * (A_lam @ eta_i + b_lam)
                log_theta_i_next = log_theta_i + jnp.log(
                    jnp.abs(jnp.linalg.det(I + dlam * A_lam))
                )
                return eta_i_next, eta_bar_i_next, log_theta_i_next

            eta_next, eta_bar_next, log_theta_next = jax.vmap(single_particle_flow)(
                eta, eta_bar, eta_bar_0, Ps, log_theta
            )

            return (eta_next, eta_bar_next, log_theta_next), None

        steps = jnp.arange(1, self.flow_steps)
        log_thetas = jnp.zeros(self.num_particles)

        (final_particles, _, final_log_thetas), _ = jax.lax.scan(
            flow_step, (eta_0, eta_bar_0, log_thetas), steps
        )

        # Evaluate measurement function for all particles
        h_vmap = jax.vmap(self.h)
        z_pred = h_vmap(final_particles)

        ## Weight update
        # w_k -> prior * likelihood * theta / proposal * w_(k-1)
        # log_w_k -> log_prior(process) + meas_log_likelihood - log_proposal + log_theta + log_w_(k-1)
        log_likelihoods = jax.vmap(
            lambda y: jax.scipy.stats.multivariate_normal.logpdf(
                y, mean=jnp.zeros(R.shape[0]), cov=R
            )
        )(z_pred - z)

        log_proposal = jax.scipy.stats.multivariate_normal.logpdf(
            particles_pred, mean=state.particles, cov=Q
        )

        log_prior = jax.scipy.stats.multivariate_normal.logpdf(
            final_particles, mean=state.particles, cov=Q
        )

        log_weight = (
            log_prior
            + log_likelihoods
            - log_proposal
            + final_log_thetas
            + jnp.log(jnp.maximum(state.weights, 1e-20))
        )  # avoid log(0)

        weights = jnp.exp(log_weight - jnp.max(log_weight))
        # weights = jnp.exp(log_weight)
        norm_weights = weights / jnp.sum(weights)

        updated_state = ParticleState(
            particles=final_particles, weights=norm_weights, key=key
        )

        # Algo 1 line 28
        internal_state_update = jax.vmap(
            lambda s: self.internal_filter.update(state=s, y=z)
        )(internal_state_pred)
        internal_state = GaussianState(
            mean=updated_state.particles, cov=internal_state_update.cov
        )

        return updated_state, PFPFParam(internal_state)
