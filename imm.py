import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Optional, Any

from .base import AbstractFilter, GaussianState

class IMMState(eqx.Module):
    """
    State container for an Interacting Multiple Model (IMM) filter.
    
    Attributes:
        states: A tuple/list of `GaussianState` objects, one for each model.
        mu: The probability array of shape (num_models,) representing the
            current belief over the models.
    """
    states: Tuple[GaussianState, ...]
    mu: jnp.ndarray

    @property
    def mean(self) -> jnp.ndarray:
        """Combined mean estimate."""
        mean = 0.0
        for i, s in enumerate(self.states):
            mean = mean + self.mu[i] * s.mean
        return mean

    @property
    def cov(self) -> jnp.ndarray:
        """Combined covariance estimate."""
        cov = 0.0
        x = self.mean
        for i, s in enumerate(self.states):
            diff = s.mean - x
            cov = cov + self.mu[i] * (s.cov + jnp.outer(diff, diff))
        return cov

class IMMFilter(AbstractFilter):
    """
    Interacting Multiple Model (IMM) Filter.
    
    This filter runs a bank of parallel filters, mixing their initial conditions
    at each prediction step based on a transition probability matrix.
    """
    filters: Tuple[AbstractFilter, ...]
    transition_matrix: jnp.ndarray

    def __init__(self, filters: Tuple[AbstractFilter, ...], transition_matrix: jnp.ndarray):
        self.filters = tuple(filters)
        self.transition_matrix = jnp.asarray(transition_matrix)
        
        num_models = len(filters)
        assert self.transition_matrix.shape == (num_models, num_models), "Transition matrix must be num_models x num_models"

    @eqx.filter_jit
    def predict(self, state: IMMState, u: Optional[jnp.ndarray] = None, params: Any = None) -> IMMState:
        """
        IMM prediction step.
        """
        m = len(self.filters)
        
        # 1. Normalizing factors for mixing probabilities (c_j)
        # c_j[j] = sum_i(p_ij[i, j] * MU_ip[i])
        c_j = jnp.sum(self.transition_matrix * state.mu[:, None], axis=0)
        
        # 2. Mixing probabilities (MU_ij)
        # MU_ij[i, j] = p_ij[i, j] * MU_ip[i] / c_j[j]
        # Shape: (m, m)
        mu_ij = (self.transition_matrix * state.mu[:, None]) / c_j[None, :]
        
        # 3. Mixing initial states and covariances
        mixed_states = []
        for j in range(m):
            # Mix mean
            x_0j = 0.0
            for i in range(m):
                x_0j = x_0j + state.states[i].mean * mu_ij[i, j]
                
            # Mix covariance
            P_0j = 0.0
            for i in range(m):
                diff = state.states[i].mean - x_0j
                P_0j = P_0j + mu_ij[i, j] * (state.states[i].cov + jnp.outer(diff, diff))
                
            mixed_states.append(GaussianState(mean=x_0j, cov=P_0j))
            
        # 4. Predict each filter
        predicted_states = []
        for j, filter_j in enumerate(self.filters):
            # Extract arguments specific to this filter if provided as sequences
            if params is not None and isinstance(params, (list, tuple)):
                p_j = params[j]
            else:
                p_j = params
                
            pred_s = filter_j.predict(mixed_states[j], u=u, params=p_j)
            predicted_states.append(pred_s)
            
        return IMMState(states=tuple(predicted_states), mu=c_j)

    @eqx.filter_jit
    def update(self, state: IMMState, y: jnp.ndarray, params: Any = None, return_likelihood: bool = False):
        """
        IMM update step.
        """
        m = len(self.filters)
        
        updated_states = []
        likelihoods = []
        
        for j, filter_j in enumerate(self.filters):
            if params is not None and isinstance(params, (list, tuple)):
                p_j = params[j]
            else:
                p_j = params
                
            # Perform update step, expecting (state, likelihood) since return_likelihood=True
            upd_s, ll = filter_j.update(state.states[j], y, params=p_j, return_likelihood=True)
            updated_states.append(upd_s)
            likelihoods.append(ll)
            
        likelihoods = jnp.stack(likelihoods)
        
        # Calculate new mode probabilities
        # state.mu here holds c_j from the predict step
        # MU = c_j * lambda / sum(c_j * lambda)
        unnormalized_mu = state.mu * likelihoods
        normalization = jnp.sum(unnormalized_mu)
        mu = unnormalized_mu / normalization
        
        updated_imm_state = IMMState(states=tuple(updated_states), mu=mu)
        
        if return_likelihood:
            return updated_imm_state, normalization
        return updated_imm_state

    @eqx.filter_jit
    def smooth_step(self, state: IMMState, next_smoothed_state: IMMState, u: Optional[jnp.ndarray] = None, params: Any = None) -> IMMState:
        """
        IMM backward smoothing step.
        Not yet implemented - requires full sequence smoothing overriding or 
        a complex single-step approximation.
        """
        raise NotImplementedError("IMM smoothing step requires special backward-pass logic. Use imm_smooth instead.")


    def filter_sequence(
        self, 
        initial_state: IMMState, 
        Y: jax.Array, 
        U: Optional[jax.Array] = None,
        Params: Any = None
    ) -> Tuple[IMMState, IMMState, IMMState]:
        """
        Vectorized offline filtering. 
        We just inherit this from AbstractFilter, as it uses predict and update.
        """
        return super().filter_sequence(initial_state, Y, U, Params)

    def smooth_sequence(
        self,
        history_update: IMMState,
        history_pred: IMMState,
        U: Optional[jnp.ndarray] = None,
        Params: Any = None
    ) -> Tuple[IMMState, IMMState]:
        """
        IMM-RTS Smoother.
        
        This implements the Interacting Multiple Model Rauch-Tung-Striebel (RTS) smoother,
        which works backward in time using the forward-filter estimates and the
        standard RTS smoothing steps of the sub-filters, without requiring inverse dynamics.
        """
        num_steps = history_update.mu.shape[0]
        m = len(self.filters)
        
        def scan_step(next_smoothed_state: IMMState, inputs):
            # unpack inputs for time k
            (state_upd_k, state_pred_kp1, u_k, params_k) = inputs
            
            # 1. Model probability smoothing
            # state_pred_kp1.mu is the predicted prob at k+1: c_j = \sum_i p_ij \mu_k^i
            # state_upd_k.mu is the updated prob at k: \mu_k^i
            
            # Backward transition probability: p(M_k=i | M_{k+1}=j) = p_{ij} \mu_k^i / c_j
            # Note: transition_matrix is p_{ij} = p(M_j | M_i)
            # So backward_p[i, j] = p_{ij} * \mu_k[i] / \mu_{kp1|k}[j]
            mu_pred_kp1 = state_pred_kp1.mu  # shape (m,)
            
            # Avoid division by zero
            mu_pred_kp1_safe = jnp.where(mu_pred_kp1 == 0, 1e-16, mu_pred_kp1)
            
            backward_p = (self.transition_matrix * state_upd_k.mu[:, None]) / mu_pred_kp1_safe[None, :]
            
            # Smoothed model probability: \mu_{k|N}^i = \mu_{k|k}^i \sum_j p_{ij} \mu_{k+1|N}^j / c_j
            # Equivalently: \mu_{k|N}^i = \sum_j backward_p[i, j] * \mu_{k+1|N}^j
            mu_smooth_k = jnp.sum(backward_p * next_smoothed_state.mu[None, :], axis=1)
            
            # Normalize to handle numerical issues
            mu_smooth_k = mu_smooth_k / jnp.sum(mu_smooth_k)
            
            # Smoothed mixing probabilities: \mu_{k|N}^{i|j} = p(M_k=i | M_{k+1}=j, Y_N)
            # \mu_{k|N}^{i|j} = backward_p[i, j] * \mu_{k+1|N}^j / \mu_{k|N}^i  --- wait, no.
            # To smooth state j, we mix the filtered states at k:
            # Actually, standard IMM-RTS smooths each model *conditioned* on the model at k+1, then mixes.
            # A simpler variant smooths each model j using its own RTS step from mixed filtered states.
            # Let's use the standard approximate IMM-RTS (Helmick 1995):
            
            # Mix the updated states at k to create the initial condition for the j-th smoother
            # Mixing weights for smoothing: W_s[i, j] = p(M_k=i | M_{k+1}=j) = backward_p[i, j]
            
            mixed_states_k = []
            for j in range(m):
                # Mix mean
                x_0j = 0.0
                for i in range(m):
                    x_0j = x_0j + state_upd_k.states[i].mean * backward_p[i, j]
                    
                # Mix covariance
                P_0j = 0.0
                for i in range(m):
                    diff = state_upd_k.states[i].mean - x_0j
                    P_0j = P_0j + backward_p[i, j] * (state_upd_k.states[i].cov + jnp.outer(diff, diff))
                    
                mixed_states_k.append(GaussianState(mean=x_0j, cov=P_0j))
                
            # Now run the RTS smoother step for each filter j
            smoothed_states_k = []
            for j, filter_j in enumerate(self.filters):
                if params_k is not None and isinstance(params_k, (list, tuple)):
                    p_j = params_k[j]
                else:
                    p_j = params_k
                    
                # The RTS smooth_step takes (state_k, next_smoothed_state, u, params)
                # But state_k is the *mixed* state we just computed
                # next_smoothed_state is the smoothed state of model j at k+1
                s_k = filter_j.smooth_step(
                    state=mixed_states_k[j], 
                    next_smoothed_state=next_smoothed_state.states[j],
                    u=u_k,
                    params=p_j
                )
                smoothed_states_k.append(s_k)
                
            smoothed_imm_state_k = IMMState(states=tuple(smoothed_states_k), mu=mu_smooth_k)
            return smoothed_imm_state_k, smoothed_imm_state_k

        # Prepare inputs for backward scan
        
        # Shift U and Params for backward scanning: smoother step at k uses u_k and params_k.
        # As established in the base AbstractFilter, we use U[1:] and Params[1:].
        if U is None:
            actual_U = jnp.zeros((num_steps, 0))
        else:
            actual_U = U
            
        U_shifted = jax.tree_util.tree_map(lambda x: x[1:], actual_U)
        if Params is not None:
            Params_shifted = jax.tree_util.tree_map(lambda x: x[1:] if isinstance(x, jnp.ndarray) else x, Params)
        else:
            Params_shifted = jnp.zeros((num_steps - 1, 0))
            
        # The states we scan over:
        # k ranges from 0 to N-2
        history_upd_k = jax.tree_util.tree_map(lambda x: x[:-1], history_update)
        history_pred_kp1 = jax.tree_util.tree_map(lambda x: x[1:], history_pred)
        
        scan_inputs = (history_upd_k, history_pred_kp1, U_shifted, Params_shifted)
        
        final_smoothed_state = jax.tree_util.tree_map(lambda x: x[-1], history_update)
        
        _, smoothed_history = jax.lax.scan(
            scan_step,
            init=final_smoothed_state,
            xs=scan_inputs,
            reverse=True
        )
        
        # Prepend the final state to the history
        def prepend_final(history_array, final_array):
            return jnp.concatenate([history_array, final_array[None, ...]], axis=0)
            
        full_smoothed_history = jax.tree_util.tree_map(
            prepend_final, smoothed_history, final_smoothed_state
        )
        
        # The initial smoothed state is the first element of the history
        initial_smoothed_state = jax.tree_util.tree_map(lambda x: x[0], full_smoothed_history)
        
        return initial_smoothed_state, full_smoothed_history

