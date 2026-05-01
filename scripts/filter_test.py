from baja.toy_problem import load_cubs_traj_data
import numpy as np
import matplotlib.pyplot as plt

from baja import pf, ekf, ukf, pff, GaussianState, ParticleState
import jax.numpy as jnp
import jax.random as jr
from functools import partial

df = load_cubs_traj_data("0.1s")[:200]
seq_len = len(df)

ts = df.index.to_numpy().astype(np.datetime64)
target_pos = df[["x", "y", "z"]].to_numpy()  # true position

"""
Define process function and measurement functions
"""
def dynamics_fn(states):
    dt = 0.1  # because we resampled every 0.1s -> 10hz,
    A = np.block(
        [[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]]
    )  # simple LTI process model
    return (A @ states.T).T


def dist_meas_fn(states, sensor_pos):
    """
    returns measurements (n_sensor, )
    [dist_0, dist_1, ...]
    """
    pos = states[:3]
    dist_meas = jnp.linalg.norm(pos - sensor_pos, axis=-1)
    return dist_meas


def bearing_meas_fn(states, sensor_pos):
    """
    returns measurements (n_sensor*2, )
    [az_0, el_0, az_1, el_1, ..., az_n, el_n]
    """
    pos = states[:3]
    b_vec = pos - sensor_pos

    dist = jnp.linalg.norm(b_vec, axis=-1)
    b_vec = b_vec / dist[:,None] # division is columns wise with shape (n,), so turn it to (1,n)

    elevs = jnp.asin(b_vec[..., 2])
    azims = jnp.atan2(b_vec[..., 1], b_vec[..., 0])
    return jnp.vstack([azims, elevs]).T.flatten()


def bearing_range_fn(states, sensor_pos):
    """
    returns measurements (n_sensor*3, )
    [dist_0, az_0, el_0, ...]
    """
    pos = states[:3]
    b_vec = pos - sensor_pos

    dist = jnp.linalg.norm(b_vec, axis=-1)
    b_vec = b_vec / dist[:, None]

    elevs = jnp.asin(b_vec[..., 2])
    azims = jnp.atan2(b_vec[..., 1], b_vec[..., 0])
    return jnp.vstack([dist, azims, elevs]).T.flatten()


rng = np.random.default_rng(12345) # cause im lazy, use for initial state randomization
key = jr.key(111) # use for randomization for each loop that involves jax

sensor_type = "br"
if sensor_type == "r":
    sensor_pos = np.array(
        [
            [20, 5, 10],
            [20, -5, 10],
            [-5, 5, 10],
            [-5, -5, 10],
        ]
    )
    meas_dim = sensor_pos.shape[0]
    R = np.eye(meas_dim) * 0.1**2
    meas_fn = partial(dist_meas_fn, sensor_pos=sensor_pos)
elif sensor_type == "b":  # azimuth and elevation
    sensor_pos = np.array(
        [
            [20, 5, 10],
            [20, -5, 10],
            [-5, 5, 10],
            # [-5, -5, 10],
        ]
    )
    meas_dim = sensor_pos.shape[0] * 2
    angular_res = np.deg2rad(2)
    R = np.eye(meas_dim) * angular_res**2
    meas_fn = partial(bearing_meas_fn, sensor_pos=sensor_pos)
elif sensor_type == "br":
    sensor_pos = np.array(
        [
            [20, 5, 10],
            # [20, -5, 10],
            # [-5, 5, 10],
            # [-5, -5, 10],
        ]
    )
    meas_dim = sensor_pos.shape[0] * 3
    angular_res = np.deg2rad(2)
    R = np.diag([0.1**2, angular_res**2, angular_res**2])
    meas_fn = partial(bearing_range_fn, sensor_pos=sensor_pos)

Q = np.diag([1e-1, 1e-1, 1e-1, 0.5, 0.5, 0.5])

init_state = np.hstack([rng.normal(target_pos[0], size=(3,)), rng.uniform(-1, 1, (3,))])
state_ekf = GaussianState(mean=init_state, cov=Q)
filter_ekf = ekf.ExtendedKalmanFilter(
    f=dynamics_fn,
    h=meas_fn,
    Q=Q,
    R=R,
)

filter_ukf = ukf.UnscentedKalmanFilter(
    f=dynamics_fn,
    h=meas_fn,
    Q=Q,
    R=R,
)
state_ukf = state_ekf

n_particles = 10000
init_state = np.hstack(
    [
        rng.normal(target_pos[0], size=(n_particles, 3)),
        rng.uniform(-1, 1, (n_particles, 3)),
    ]
)
state_bpf = ParticleState(init_state, np.ones(n_particles) / n_particles, key)
filter_bpf = pf.ParticleFilter(
    f=dynamics_fn,
    h=meas_fn,
    num_particles=n_particles,
    Q=Q,
    R=R,
)

n_particles = 100
init_state = np.hstack(
    [
        rng.normal(target_pos[0], size=(n_particles, 3)),
        rng.uniform(-1, 1, (n_particles, 3)),
    ]
)
state_edh = ParticleState(init_state, np.ones(n_particles) / n_particles, key)
filter_edh = pff.EDHFilter(
    f=dynamics_fn,
    h=meas_fn,
    num_particles=n_particles,
    Q=Q,
    R=R,
)

params_iedh = pff.PFPFParam(
    internal_state=GaussianState(mean=init_state[0, ...], cov=Q)
)
state_iedh = pff.ParticleState(init_state, np.ones(n_particles) / n_particles, key)
filter_iedh = pff.IEDHFilter(
    f=dynamics_fn,
    h=meas_fn,
    internal_filter=filter_ukf,
    num_particles=n_particles,
    Q=Q,
    R=R,
)

params_iledh = pff.PFPFParam(
    internal_state=GaussianState(mean=init_state, cov=np.tile(Q, (n_particles, 1, 1)))
)
state_iledh = state_iedh
filter_iledh = pff.ILEDHFilter(
    f=dynamics_fn,
    h=meas_fn,
    internal_filter=filter_ukf,
    num_particles=n_particles,
    Q=Q,
    R=R,
)


bpf_hist = []
bpf_hist.append(state_bpf)

# edh_hist = []
# edh_hist.append(state_edh)

iedh_hist = []
iedh_hist.append(state_iedh)

iledh_hist = []
iledh_hist.append(state_iledh)

ekf_hist = []
ekf_hist.append(state_ekf)

ukf_hist = []
ukf_hist.append(state_ukf)

for i in range(seq_len):
    key, sub_key = jr.split(key)
    true_pos = target_pos[i, :]

    meas_noisy = meas_fn(true_pos) + jr.uniform(
        key=sub_key, shape=(meas_dim,), minval=-1, maxval=1
    ) * jnp.sqrt(jnp.diag(R))

    state_bpf = filter_bpf.run_step(state_bpf, meas_noisy)
    bpf_hist.append(state_bpf)

    # state_edh = filter_edh.run_step(state_edh, meas_noisy)
    # edh_hist.append(state_edh)

    state_iedh, params_iedh = filter_iedh.run_step(
        state_iedh, meas_noisy, params=params_iedh
    )
    iedh_hist.append(state_iedh)

    state_iledh, params_iledh = filter_iledh.run_step(
        state_iledh, meas_noisy, params=params_iledh
    )
    iledh_hist.append(state_iledh)

    state_pred_ekf = filter_ekf.predict(state_ekf)
    state_ekf = filter_ekf.update(state_pred_ekf, meas_noisy)
    ekf_hist.append(state_ekf)

    state_pred_ukf = filter_ukf.predict(state_ukf)
    state_ukf = filter_ukf.update(state_pred_ukf, meas_noisy)
    ukf_hist.append(state_ukf)

bpf_est = np.array([p.mean for p in bpf_hist])
# edh_est = np.array([p.mean for p in edh_hist])
iedh_est = np.array([p.mean for p in iedh_hist])
ekf_est = np.array([s.mean for s in ekf_hist])
ukf_est = np.array([s.mean for s in ukf_hist])
iledh_est = np.array([p.mean for p in iledh_hist])

bpf_error = jnp.linalg.norm(bpf_est[:-1, :3] - target_pos, axis=-1)
ekf_error = jnp.linalg.norm(ekf_est[:-1, :3] - target_pos, axis=-1)
ukf_error = jnp.linalg.norm(ukf_est[:-1, :3] - target_pos, axis=-1)

iedh_error = jnp.linalg.norm(iedh_est[:-1, :3] - target_pos, axis=-1)
iledh_error = jnp.linalg.norm(iledh_est[:-1, :3] - target_pos, axis=-1)
