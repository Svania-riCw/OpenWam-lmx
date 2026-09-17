"""Compute EEF20 normalization statistics for a dual-Piper LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.piper_lerobot import (
    ACTION_STATS_KEY,
    GRIPPER_CONVENTION,
    NORMALIZATION_STATS_FILENAME,
    REPRESENTATION,
    STATE_STATS_KEY,
)
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, pin_rot6d_identity
from openwam.dataloader.utils.piper_kinematics import JOINT14_DIM, joint14_to_eef20

ACTION_COLUMN = "action"
STATE_COLUMN = "observation.state"


def _read_column(path: Path, column: str) -> np.ndarray:
    try:
        table = pq.read_table(path, memory_map=True, columns=[column])
    except pa.ArrowInvalid as exc:
        if "Dot path" not in str(exc):
            raise
        table = pq.read_table(path, memory_map=True).select([column])
    values = np.asarray(table.column(column).to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != JOINT14_DIM:
        raise ValueError(f"{path}: {column!r} must be (N, {JOINT14_DIM}), got {values.shape}")
    return values


def _feature_stats(values: np.ndarray, *, semantics: str) -> dict:
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    stats = {
        "min": values.min(axis=0).astype(np.float64).tolist(),
        "max": values.max(axis=0).astype(np.float64).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "q01": quantiles[0].astype(np.float64).tolist(),
        "q10": quantiles[1].astype(np.float64).tolist(),
        "q50": quantiles[2].astype(np.float64).tolist(),
        "q90": quantiles[3].astype(np.float64).tolist(),
        "q99": quantiles[4].astype(np.float64).tolist(),
        "count": [int(values.shape[0])],
        "num_timesteps": int(values.shape[0]),
        "representation": REPRESENTATION,
        "gripper_convention": GRIPPER_CONVENTION,
        "semantics": semantics,
    }
    pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
    return stats


def compute_piper_stats(dataset_dir: str | Path) -> dict:
    """Scan all data shards and return separate action/state EEF20 stats."""
    root = Path(dataset_dir)
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    actions, states = [], []
    for path in paths:
        actions.append(joint14_to_eef20(_read_column(path, ACTION_COLUMN)))
        states.append(joint14_to_eef20(_read_column(path, STATE_COLUMN)))
    action_all = np.concatenate(actions, axis=0)
    state_all = np.concatenate(states, axis=0)
    return {
        ACTION_STATS_KEY: _feature_stats(action_all, semantics="absolute commanded EEF pose"),
        STATE_STATS_KEY: _feature_stats(state_all, semantics="absolute measured EEF pose"),
    }


def build_and_save_piper_stats(dataset_dir: str | Path, output: str | Path | None = None) -> Path:
    """Compute and atomically save the stats payload."""
    root = Path(dataset_dir)
    out = Path(output) if output else root / "meta" / NORMALIZATION_STATS_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = compute_piper_stats(root)
    fd, tmp_name = tempfile.mkstemp(dir=str(out.parent), suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, payload, allow_pickle=True)
        os.replace(tmp_name, out)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = build_and_save_piper_stats(args.dataset_dir, args.output)
    payload = np.load(output, allow_pickle=True).item()
    print(f"wrote {output}")
    for key in (ACTION_STATS_KEY, STATE_STATS_KEY):
        block = payload[key]
        print(f"  {key}: {block['num_timesteps']} rows, xyz mean={np.round(block['mean'][:3], 5).tolist()}")


if __name__ == "__main__":
    main()
