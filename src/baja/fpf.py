import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any, Tuple

from .base import AbstractFilter, GaussianState
from .pf import ParticleState


class FPF(AbstractFilter):
    """
    Feedback particle filter
    Constant gain approximation
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
    def run_step(
        self,
        state: ParticleState,
        z: jax.Array,
        u: Optional[jax.Array] = None,
        params: Any = None,
    ):
        """
        state: particles from previous step
        z: measurement at current step
        u: optional control signal, should be None for tracking
        params: optional time varying parameters, should be none here
        """

        """
        ================================================================
        Prediction step: propagates particles through the process model.
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
        # internal_state_pred = jax.vmap(self.internal_filter.predict)(params.internal_state)

        eta_0 = particles_pred

        """
        ================================================================
        Update step: feedback controled homotopy flow
        ================================================================
        """
        R = params.R if params is not None and hasattr(params, "R") else self.R
        assert R is not None, "Measurement noise covariance R must be provided."

        # P = self._compute_covariances(particles_pred)

        # I = jnp.eye(state.particles.shape[-1])  # match identity mat shape to state dim
        R_inv = jnp.linalg.inv(R)
        def flow_step(carry, step):
            """
            Constant gain approximation
            """
            # [prop_with_noise]
            eta = carry

            # exponential steps over lambda, with q=1.2
            q = 1.2
            lam = (1 - q**step) / (1 - q**self.flow_steps)
            dlam = lam - (1 - q ** (step - 1)) / (1 - q**self.flow_steps)
            # dlam = 1 / self.flow_steps

            # estimate h_hat
            h_vmap = jax.vmap(self.h)
            h_hat = jnp.mean(h_vmap(eta), axis=0)

            def calc_K(eta_i):
                return eta_i[:, None] @ (self.h(eta_i) - h_hat)[None]
            K = jnp.mean(jax.vmap(calc_K)(eta), axis=0) @ R_inv

            # def calc_dh_square(z_pred_i):
            #     return z_pred_i @ R_inv @ z_pred_i
            # g_hat = h_hat @ R_inv @ h_hat - jnp.mean(jax.vmap(calc_dh_square)(h_vmap(eta)))
            
            # def calc_Omega(eta_i):
            #     H = jax.jacfwd(self.h)(eta_i)
            #     g = jnp.sum(K.T * H)
            #     return eta_i * (g - g_hat)
            
            # Omega = jnp.mean(jax.vmap(calc_Omega)(eta), axis=0)
            
            def calc_g(eta_i):
                H = jax.jacfwd(self.h)(eta_i)
                g_i = jnp.sum(K.T * H)
                return g_i
            g = jax.vmap(calc_g)(eta)
            g_hat = jnp.mean(g)
            Omega = jnp.mean(eta * (g - g_hat)[:, None], axis=0)

            def single_particle_flow(eta_i):
                h_lam = self.h(eta_i)
                inno_lam = z - 0.5 * (h_lam + h_hat)

                eta_i_next = eta_i + dlam * (K @ inno_lam + 0.5 * Omega)
                # eta_i_next = eta_i + dlam * K @ inno_lam

                return eta_i_next

            eta_next = jax.vmap(single_particle_flow)(eta)

            return eta_next, (eta_next, K, Omega)

        steps = jnp.arange(1, self.flow_steps)

        final_particles, stuff = jax.lax.scan(flow_step, eta_0, steps)

        # No weighting
        updated_state = ParticleState(
            particles=final_particles, weights=state.weights, key=key
        )

        return updated_state, stuff
