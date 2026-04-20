import pandas as pd
from .path import data_dir


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
