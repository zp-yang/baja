import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any, Tuple

from .base import AbstractFilter, State

class ParticleState(eqx.Module):
    """
    State container for a Particle Filter.
    
    Attributes:
        particles: Array of shape (num_particles, state_dim).
        weights: Array of shape (num_particles,) representing the normalized
            probabilities of each particle.
        key: A JAX PRNGKey used for sampling process noise and resampling.
    """
    particles: jnp.ndarray
    weights: jnp.ndarray
    key: jax.Array

    @property
    def mean(self) -> jnp.ndarray:
        """Combined mean estimate (weighted average)."""
        return jnp.sum(self.particles * self.weights[:, None], axis=0)

    @property
    def cov(self) -> jnp.ndarray:
        """Combined covariance estimate (weighted covariance)."""
        mean = self.mean
        diff = self.particles - mean[None, :]
        # shape (num_particles, state_dim, 1) * shape (num_particles, 1, state_dim) -> (num_particles, state_dim, state_dim)
        cov_components = self.weights[:, None, None] * (diff[:, :, None] * diff[:, None, :])
        return jnp.sum(cov_components, axis=0)

class ParticleFilter(AbstractFilter):
    """
    Bootstrap Particle Filter (Standard SIR Particle Filter).
    
    This filter propagates a set of particles through a nonlinear dynamic model
    and updates their weights based on a nonlinear measurement model. It uses
    conditional resampling to prevent particle degeneracy.
    """
    f: Callable
    h: Callable
    Q: Optional[jax.Array] = None
    R: Optional[jax.Array] = None
    num_particles: int = eqx.field(static=True)
    resample_threshold: float = eqx.field(static=True)

    def __init__(
        self, 
        f: Callable, 
        h: Callable, 
        num_particles: int, 
        Q: Optional[jax.Array] = None, 
        R: Optional[jax.Array] = None,
        resample_threshold: Optional[float] = None
    ):
        self.f = f
        self.h = h
        self.Q = jnp.asarray(Q) if Q is not None else None
        self.R = jnp.asarray(R) if R is not None else None
        self.num_particles = num_particles
        
        # Default resampling threshold is N/2
        if resample_threshold is None:
            self.resample_threshold = num_particles / 2.0
        else:
            self.resample_threshold = float(resample_threshold)

    @eqx.filter_jit
    def predict(
        self, state: ParticleState, u: Optional[jax.Array] = None, params: Any = None
    ) -> ParticleState:
        """
        Particle Filter prediction step.
        """
        Q = params.Q if params is not None and hasattr(params, 'Q') else self.Q
        assert Q is not None, "Process noise covariance Q must be provided."
        
        # Split the random key
        key, subkey = jax.random.split(state.key)
        
        # We pass u as an argument to f if u is provided, else f(x)
        if u is not None and u.size > 0:
            f_vmap = jax.vmap(lambda x: self.f(x, u))
        else:
            f_vmap = jax.vmap(self.f)
            
        # Propagate particles deterministically
        particles_next = f_vmap(state.particles)
        
        # Add process noise
        state_dim = state.particles.shape[1]
        noise = jax.random.multivariate_normal(
            subkey, 
            mean=jnp.zeros(state_dim), 
            cov=Q, 
            shape=(self.num_particles,)
        )
        
        particles_pred = particles_next + noise
        
        return ParticleState(
            particles=particles_pred, 
            weights=state.weights, 
            key=key
        )

    def _resample(self, particles: jnp.ndarray, weights: jnp.ndarray, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Systematic resampling.
        """
        N = self.num_particles
        
        # Generate N uniformly distributed numbers
        positions = (jax.random.uniform(key) + jnp.arange(N)) / N
        
        # Compute cumulative sum of weights
        cum_weights = jnp.cumsum(weights)
        cum_weights = cum_weights / cum_weights[-1] # Ensure it ends exactly at 1.0
        
        # Find indices using searchsorted
        indices = jnp.searchsorted(cum_weights, positions)
        
        # Resample particles and reset weights
        resampled_particles = particles[indices]
        reset_weights = jnp.ones(N) / N
        
        return resampled_particles, reset_weights

    @eqx.filter_jit
    def update(
        self, state: ParticleState, y: jax.Array, params: Any = None, return_likelihood: bool = False
    ):
        """
        Particle Filter update step with conditional resampling.
        """
        R = params.R if params is not None and hasattr(params, 'R') else self.R
        assert R is not None, "Measurement noise covariance R must be provided."
        
        # Split the random key for potential resampling
        key, subkey = jax.random.split(state.key)
        
        # Evaluate measurement function for all particles
        h_vmap = jax.vmap(self.h)
        z_pred = h_vmap(state.particles)
        
        # Calculate log likelihoods
        # z_pred shape: (num_particles, meas_dim)
        # y shape: (meas_dim,)
        log_likelihoods = jax.vmap(
            lambda z: jax.scipy.stats.multivariate_normal.logpdf(y, mean=z, cov=R)
        )(z_pred)
        
        # Update log weights
        # Avoid log(0) by using a small epsilon, though usually weights are initialized to 1/N
        log_prior_weights = jnp.log(jnp.maximum(state.weights, 1e-300))
        log_posterior_weights = log_prior_weights + log_likelihoods
        
        # Normalize weights safely using log-sum-exp trick
        max_log_w = jnp.max(log_posterior_weights)
        exp_w = jnp.exp(log_posterior_weights - max_log_w)
        sum_exp_w = jnp.sum(exp_w)
        
        normalized_weights = exp_w / sum_exp_w
        # normalized_weights = jax.nn.softmax(log_posterior_weights)
        
        # Likelihood of the measurement p(y_k | Y_{k-1})
        # This is the sum of the unnormalized weights (in linear space)
        # p(y_k | Y_{k-1}) = sum( w_{k-1} * p(y_k | x_k) )
        # Which is mathematically exactly sum_exp_w * exp(max_log_w)
        if return_likelihood:
            likelihood = sum_exp_w * jnp.exp(max_log_w)
            
        # Calculate Effective Sample Size (ESS)
        ess = 1.0 / jnp.sum(normalized_weights ** 2)
        
        # Conditional resampling
        def do_resample(operands):
            p, w, k = operands
            return self._resample(p, w, k)
            
        def skip_resample(operands):
            p, w, _ = operands
            return p, w
            
        final_particles, final_weights = jax.lax.cond(
            ess < self.resample_threshold,
            do_resample,
            skip_resample,
            (state.particles, normalized_weights, subkey)
        )
        
        updated_state = ParticleState(
            particles=final_particles,
            weights=final_weights,
            key=key
        )
        
        if return_likelihood:
            return updated_state, likelihood
        return updated_state

    @eqx.filter_jit
    def smooth_step(
        self, state: ParticleState, next_smoothed_state: ParticleState, u: Optional[jax.Array] = None, params: Any = None
    ) -> ParticleState:
        """
        Particle smoothing is highly complex and not supported by default.
        """
        raise NotImplementedError("Particle smoothing is not currently supported.")

    def smooth_sequence(
        self,
        history_update: ParticleState,
        history_pred: ParticleState,
        U: Optional[jax.Array] = None,
        Params: Any = None
    ) -> Tuple[ParticleState, ParticleState]:
        """
        Particle smoothing sequence.
        """
        raise NotImplementedError("Particle smoothing is not currently supported.")
