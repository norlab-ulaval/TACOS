import tarfile

import cv2
import numpy as np
import pytest
import zarr

from tacos.download import _write_images, convert_imu, convert_joints
from tacos.robot import JOINTS, LEGS


def make_tar(path, files):
    with tarfile.open(path, "w") as tar:
        for f in files:
            tar.add(f, f.name)
    return path


def test_undistortion_keeps_the_principal_point(tmp_path):
    camera = {"K": [500.0, 0, 64, 0, 500.0, 48, 0, 0, 1], "D": [-0.06, 0.003, -0.001, 0]}
    image = np.zeros((96, 128, 3), np.uint8)
    image[48, 64] = 255
    cv2.imwrite(str(tmp_path / "0.png"), image)
    (tmp_path / "out").mkdir()
    _write_images(make_tar(tmp_path / "images.tar", [tmp_path / "0.png"]), camera, np.array([7]), tmp_path / "out", 1,
                  "MISSION")
    out = cv2.imread(str(tmp_path / "out" / "7.png"))
    assert out.shape == image.shape and np.unravel_index(out[..., 0].argmax(), out.shape[:2]) == (48, 64)


def test_image_and_timestamp_counts_must_match(tmp_path):
    files = [tmp_path / "0.jpg", tmp_path / "1.jpg"]
    for f in files:
        f.write_bytes(b"")
    with pytest.raises(ValueError, match="MISSION: 2 hdr_front images vs 3 timestamps"):
        _write_images(make_tar(tmp_path / "images.tar", files), {"K": np.eye(3).ravel(), "D": [0] * 4}, np.arange(3),
                      tmp_path, 1, "MISSION")


def test_imu_is_rotated_into_the_base_frame(tmp_path):
    g = zarr.open_group(tmp_path / "anymal_imu", mode="w", zarr_format=2)
    g["timestamp"] = np.array([1.0, 1.0025])
    g["ang_vel"] = np.array([[1.0, 2, 3]] * 2)
    g["lin_acc"] = np.array([[4.0, 5, 6]] * 2)
    g.attrs["transform"] = {"rotation": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0},   # 180° about y
                            "translation": {"x": 0, "y": 0, "z": 0}}
    df = convert_imu(g)
    assert df.index.name == "timestamp_ns" and list(df.index) == [1_000_000_000, 1_002_500_000]
    np.testing.assert_allclose(df[["gyro_x", "gyro_y", "gyro_z"]].values[0], [-1, 2, -3], atol=1e-12)
    np.testing.assert_allclose(df[["acc_x", "acc_y", "acc_z"]].values[0], [-4, 5, -6], atol=1e-12)


def test_joint_columns_follow_the_actuator_index(tmp_path):
    g = zarr.open_group(tmp_path / "anymal_state_actuator", mode="w", zarr_format=2)
    g["timestamp"] = np.array([2.0, 2.0025])
    for i in range(12):
        for name, offset in (("position", 0), ("velocity", 1), ("torque", 2)):
            g[f"{i:02d}_state_joint_{name}"] = np.full(2, 10.0 * i + offset)
    df = convert_joints(g)
    assert list(df.columns) == [f"{leg}_{j}_{f}" for leg in LEGS for j in JOINTS for f in ("pos", "vel", "torque")]
    assert (df["LF_HAA_pos"].iloc[0], df["RF_HAA_vel"].iloc[0], df["RH_KFE_torque"].iloc[0]) == (0, 31, 112)
