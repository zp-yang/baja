import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Any, Tuple, TypeVar

State = TypeVar("State", bound=eqx.Module)


class GaussianState(eqx.Module):
    """
    Represents the state of a Gaussian filter at a given time step.

    Attributes:
        mean: A 1D array of shape (N,) representing the state mean vector.
        cov: A 2D array of shape (N, N) representing the state covariance matrix.
    """

    mean: jax.Array
    cov: jax.Array


class AbstractFilter(eqx.Module):
    """
    Base class for all recursive Bayesian filters (KF, EKF, UKF, etc.).
    """

    def predict(
        self,
        state: State,
        u: Optional[jax.Array] = None,
        params: Any = None,
    ) -> State:
        """
        Single-step prediction for real-time loops.
        """
        raise NotImplementedError

    def update(
        self,
        state: State,
        y: jax.Array,
        params: Any = None,
        return_likelihood: bool = False,
    ) -> State:
        """
        Single-step update for real-time callbacks.
        """
        raise NotImplementedError

    def run_step(
        self,
        state: State,
        y: jax.Array,
        u: Optional[jax.Array] = None,
        params: Any = None,
        return_likelihood: bool = False,
    ) -> State:
        "Update then predict"
        raise NotImplementedError

    def smooth_step(
        self,
        state: State,
        next_smoothed_state: State,
        u: Optional[jax.Array] = None,
        params: Any = None,
    ) -> State:
        """
        Single-step backward smoothing operation.
        """
        raise NotImplementedError

    @eqx.filter_jit
    def filter_sequence(
        self,
        initial_state: State,
        Y: jax.Array,
        U: Optional[jax.Array] = None,
        Params: Any = None,
    ) -> Tuple[State, State, State]:
        """
        Vectorized offline filtering over a full time series.

        Args:
            initial_state: The initial state of the filter.
            Y: Array of measurements of shape (Time, measurement_dim).
            U: Optional array of control inputs of shape (Time, input_dim).
            Params: Optional sequence of time-varying parameters (PyTree with leading Time dim).

        Returns:
            final_state: The state at the end of the sequence.
            history_pred: A GaussianState PyTree containing the history of all predictions.
            history_update: A GaussianState PyTree containing the history of all updates.
        """

        def scan_step(state: GaussianState, inputs: Tuple[jax.Array, jax.Array, Any]):
            y_t, u_t, params_t = inputs

            # Predict
            # If u_t is a dummy array (e.g. zeros of shape (0,)), we treat it as None
            actual_u_t = u_t if u_t.size > 0 else None
            actual_params_t = (
                params_t
                if (not isinstance(params_t, jax.Array) or params_t.size > 0)
                else None
            )

            predicted_state = self.predict(state, u=actual_u_t, params=actual_params_t)

            # Update (with NaN handling if measurement is missing)
            # In a real scenario, you can add conditional logic here:
            # updated_state = jax.lax.cond(
            #     jnp.isnan(y_t).any(),
            #     lambda _: predicted_state,
            #     lambda _: self.update(predicted_state, y_t, params=actual_params_t),
            #     None
            # )
            updated_state = self.update(predicted_state, y_t, params=actual_params_t)

            return updated_state, (predicted_state, updated_state)

        # Handle optional inputs to ensure they can be scanned over
        num_steps = Y.shape[0]
        U_seq = U if U is not None else jnp.zeros((num_steps, 0))

        # We need Params to be scannable. If it's None, we create a dummy sequence of Nones
        # We assume if Params is a PyTree, all its leaves have a leading dimension of `num_steps`.
        Params_seq = Params if Params is not None else jnp.zeros((num_steps, 0))

        # JAX executes the entire loop in compiled XLA
        final_state, (history_pred, history_update) = jax.lax.scan(
            scan_step, initial_state, (Y, U_seq, Params_seq)
        )

        return final_state, history_pred, history_update

    @eqx.filter_jit
    def smooth_sequence(
        self, history_update: State, U: Optional[jax.Array] = None, Params: Any = None
    ) -> State:
        """
        Vectorized backward RTS smoother over a full time series.
        """

        def scan_step(next_smoothed_state: State, inputs: Tuple[State, jax.Array, Any]):
            state_k, u_k, params_k = inputs

            actual_u_k = u_k if u_k.size > 0 else None
            actual_params_k = (
                params_k
                if (not isinstance(params_k, jax.Array) or params_k.size > 0)
                else None
            )

            smoothed_state_k = self.smooth_step(
                state=state_k,
                next_smoothed_state=next_smoothed_state,
                u=actual_u_k,
                params=actual_params_k,
            )

            return smoothed_state_k, smoothed_state_k

        # Extract Time dimension and slice arrays
        # For a generic State, we can extract num_steps from the first leaf
        num_steps = jax.tree_util.tree_leaves(history_update)[0].shape[0]

        # The last state is unchanged by smoothing
        final_smoothed_state = jax.tree_util.tree_map(lambda x: x[-1], history_update)

        # Slice off the last element for scanning backwards over 0 to T-2
        state_history_k = jax.tree_util.tree_map(lambda x: x[:-1], history_update)

        # The transition from k to k+1 uses the input/params at k+1
        U_seq = U[1:] if U is not None else jnp.zeros((num_steps - 1, 0))
        Params_seq = (
            jax.tree_util.tree_map(lambda x: x[1:], Params)
            if Params is not None
            else jnp.zeros((num_steps - 1, 0))
        )

        # JAX executes the backward loop in compiled XLA
        _, history_smooth_k = jax.lax.scan(
            scan_step,
            final_smoothed_state,
            (state_history_k, U_seq, Params_seq),
            reverse=True,
        )

        # Concatenate the history with the final step to return full shape
        full_smooth = jax.tree_util.tree_map(
            lambda hist, final: jnp.vstack([hist, final[None, ...]]),
            history_smooth_k,
            final_smoothed_state,
        )

        # Concatenate the history with the final step to return full shape
        full_smooth = jax.tree_map(
            lambda hist, final: jnp.vstack([hist, final[None, ...]]),
            history_smooth_k,
            final_smoothed_state,
        )

        return full_smooth
