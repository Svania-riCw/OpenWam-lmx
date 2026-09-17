from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from openwam.dataloader.piper_lerobot import PiperLeRobotDataset
from openwam.dataloader.registry import DATASET_REGISTRY
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20
from openwam.dataloader.utils.piper_kinematics import joint14_to_eef20, piper_forward_kinematics


def test_config_and_registry() -> None:
    config = yaml.safe_load(Path("configs/dataloader/piper_lerobot.yaml").read_text(encoding="utf-8"))
    assert config["type"] == "piper_lerobot"
    assert config["unify_action_map"] == ["0-9", "34-43"]
    assert DATASET_REGISTRY["piper_lerobot"] is PiperLeRobotDataset


def test_joint14_to_eef20_matches_piper_reference() -> None:
    joint14 = np.asarray(
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.3, 0.2, -0.1, 0.4, -0.5, 0.6, 1.2],
        dtype=np.float64,
    )
    expected = np.asarray(
        [
            0.047375617,
            -0.012321322,
            0.082297794,
            -0.319265670,
            -0.838186744,
            -0.442167857,
            -0.335811742,
            0.536374117,
            -0.774295345,
            0.7,
            0.054461662,
            -0.034630685,
            0.274869827,
            0.609587500,
            0.651059079,
            -0.452244575,
            -0.081505976,
            0.618950897,
            0.781189198,
            1.2,
        ],
        dtype=np.float32,
    )
    actual = joint14_to_eef20(joint14)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_forward_kinematics_is_vectorized_and_rigid() -> None:
    transforms = piper_forward_kinematics(np.zeros((3, 6), dtype=np.float32))
    assert transforms.shape == (3, 4, 4)
    for transform in transforms:
        rotation = transform[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-7)


def test_reader_normalization_preserves_rot6d() -> None:
    reader = object.__new__(PiperLeRobotDataset)
    reader._normalize_mode = "min-max"
    stats = {"min": np.full(20, -2.0, dtype=np.float32), "max": np.full(20, 2.0, dtype=np.float32)}
    joints = np.zeros((2, 14), dtype=np.float32)
    raw = joint14_to_eef20(joints)
    normalized = reader._normalize_eef(joints, stats)
    np.testing.assert_array_equal(normalized[:, ROT6D_DIMS_EEF20], raw[:, ROT6D_DIMS_EEF20])
