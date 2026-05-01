import pandas as pd
from .path import data_dir
import numpy as np
import jax.numpy as jnp
from functools import partial
import rerun as rr


def load_cubs_traj_data(resample_rule):
    traj_raw = pd.read_parquet(data_dir / "cub_sample_traj.parquet")
    if resample_rule == "raw":
        return traj_raw.set_index("timestamp")
    traj_data = (
        traj_raw.set_index("timestamp")
        .resample(resample_rule, origin="start")
        .interpolate("time")
    )
    return traj_data


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
    b_vec = (
        b_vec / dist[:, None]
    )  # division is columns wise with shape (n,), so turn it to (1,n)

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


def get_toy_setup_params(sensor_type="br"):
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
                # [-5, 5, 10],
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
    return Q, R, meas_fn, sensor_pos


def log_measurement(sensor_type, meas_noisy, sensor_pos):
    if sensor_type == "br":
        m = meas_noisy.reshape(sensor_pos.shape[0], 3)
        dis = meas_noisy[..., 0]
        azi = meas_noisy[..., 1]
        ele = meas_noisy[..., 2]
        vec = (
            np.array(
                [np.cos(ele) * np.cos(azi), np.cos(ele) * np.sin(azi), np.sin(ele)]
            )
            * dis
        )
        rr.log(
            "sensors/meas_noisy",
            rr.Arrows3D(
                vectors=vec.T,
                origins=sensor_pos,
            ),
        )
    elif sensor_type == "b":
        dis = 30  # a long enough line
        m = meas_noisy.reshape(sensor_pos.shape[0], 2)
        azi = m[..., 0]
        ele = m[..., 1]
        vec = (
            np.array(
                [np.cos(ele) * np.cos(azi), np.cos(ele) * np.sin(azi), np.sin(ele)]
            )
            * dis
        )
        rr.log(
            "sensors/meas_noisy",
            rr.Arrows3D(
                vectors=vec.T,
                origins=sensor_pos,
            ),
        )

    elif sensor_type == "r":
        rr.log(
            "sensors/meas_noisy",
            rr.Ellipsoids3D(
                radii=meas_noisy,
                centers=sensor_pos,
            ),
        )
