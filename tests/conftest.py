import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from tacos import model
from tacos.robot import JOINTS, LEGS

HZ = 400.0
Q0 = {"HAA": 0.0, "HFE": 0.6, "KFE": -1.1}
K = [[200.0, 0, 64], [0, 200.0, 48], [0, 0, 1]]
T_BASE_CAM = [[0, 0, 1, 0.3], [-1, 0, 0, 0], [0, -1, 0, 0.1], [0, 0, 0, 1]]   # looking forward


def write_run(path, seconds=4.0, slip_leg=None, slip_window=(1.5, 2.5), slip_m=0.1, contact=1):
    """A robot standing still on loaded feet, one black 96x128 image per second. With `slip_leg`, that foot is
    dragged by about `slip_m` through its HFE joint during `slip_window` while the base does not move."""
    n = int(seconds * HZ)
    t = np.arange(n) / HZ
    ts = (np.arange(n) * 1e9 / HZ).astype(np.int64)
    joints = {"timestamp_ns": ts}
    for leg in LEGS:
        for j in JOINTS:
            q = np.full(n, Q0[j])
            if leg == slip_leg and j == "HFE":
                start, end = slip_window
                q += np.clip((t - start) / (end - start), 0, 1) * slip_m / 0.6
            joints[f"{leg}_{j}_pos"] = q
            joints[f"{leg}_{j}_vel"] = np.gradient(q) * HZ
            joints[f"{leg}_{j}_torque"] = np.full(n, 20.0 if j == "KFE" else 0.0)
    imu = {"gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0, "acc_x": 0.0, "acc_y": 0.0, "acc_z": 9.81}
    pose = {"x": 0.0, "y": 0.0, "z": 0.5, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    camera = {"width": 128, "height": 96, "K": K, "T_base_camera": T_BASE_CAM}

    path.mkdir(parents=True)
    pd.DataFrame(joints).to_csv(path / "joints.csv", index=False)
    pd.DataFrame({"timestamp_ns": ts, **imu}).to_csv(path / "imu.csv", index=False)
    pd.DataFrame({"timestamp_ns": ts, **dict.fromkeys(LEGS, contact)}).to_csv(path / "contacts.csv", index=False)
    pd.DataFrame({"timestamp_ns": ts, **pose}).to_csv(path / "poses.csv", index=False)
    (path / "camera.yaml").write_text(yaml.safe_dump(camera))
    (path / "images").mkdir()
    for s in range(int(seconds)):
        cv2.imwrite(str(path / "images" / f"{s * 10**9}.png"), np.zeros((96, 128, 3), np.uint8))
    return path


class StubEncoder(torch.nn.Module):
    """Stands in for DINOv3 so that no weights are downloaded: one patchify conv, repeated for the 4 levels."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(3, model.EMBED_DIM, 16, 16)

    def forward(self, x):
        return torch.stack([self.proj(x)] * 4, dim=1)


@pytest.fixture
def stub_encoder(monkeypatch):
    monkeypatch.setattr(model, "DINOv3Encoder", StubEncoder)


@pytest.fixture
def standing(tmp_path):
    return write_run(tmp_path / "standing")


@pytest.fixture
def slipping(tmp_path):
    return write_run(tmp_path / "slipping", slip_leg="LF", slip_m=0.3)   # at 0.1 m the Huber loss hides most of it


@pytest.fixture
def labels_root(tmp_path):
    """Three labelled runs of 30 random 96x128 frames, each with one LF footstep (m_hard 1.5, m_soft 0.5)."""
    rng = np.random.default_rng(0)
    ts = np.arange(30) * 10**9
    for name in ("RUN-1", "RUN-2", "RUN-3"):
        labels = tmp_path / name / "labels"
        (labels / "images").mkdir(parents=True)
        for t in ts:
            cv2.imwrite(str(labels / "images" / f"{t}.png"), rng.integers(0, 255, (96, 128, 3), dtype=np.uint8))
        pd.DataFrame({"timestamp_ns": ts, "roll": 0.1 + 0.01 * np.arange(30), "pitch": 0.05,
                      "history_front": 0.3, "history_hind": 0.4}).to_csv(labels / "frames.csv", index=False)
        pd.DataFrame({"timestamp_ns": ts, "foot": "LF", "t_start": ts - 5, "pixel_x": 20 + np.arange(30),
                      "pixel_y": 60, "distance": 1.0, "pitch": 0.8, "m_hard": 1.5, "m_soft": 0.5}
                     ).to_csv(labels / "footsteps.csv", index=False)
        (labels / "camera.yaml").write_text(yaml.safe_dump({"width": 128, "height": 96, "K": K}))
    return tmp_path
