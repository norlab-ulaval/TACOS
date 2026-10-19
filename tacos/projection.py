"""Project stance positions into camera images and splat them as label masks."""
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

COLUMNS = ["foot", "t_start", "pixel_x", "pixel_y", "distance", "pitch", "m_hard", "m_soft"]


class Projector:
    """Projects world-frame stance positions into the camera at an image time.

    `pose_at(ts_ns)` returns the 4x4 world <- base transform at `ts_ns` (see `for_run`);
    `T_base_camera` is the static base <- camera extrinsic, so the camera pose is `pose_at(ts) @ T_base_camera`.
    """

    def __init__(self, K, T_base_camera, pose_at: Callable[[int], np.ndarray] | None, foot_diameter: float = 0.06):
        self.K, self.T_base_camera, self.pose_at = np.asarray(K, float), np.asarray(T_base_camera, float), pose_at
        self.foot_diameter_px = foot_diameter * self.K[0, 0]

    @classmethod
    def for_run(cls, run) -> "Projector":
        """The pose is the nearest row of `Run.base_pose`, i.e. the smoothed odometry on the joints time base."""
        xyz, rpy = run.base_pose
        ts = run.joints.index.values.astype(np.int64)

        def pose_at(ts_ns: int) -> np.ndarray:
            i = int(np.abs(ts - int(ts_ns)).argmin())
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Rotation.from_euler("xyz", rpy[i]).as_matrix(), xyz[i]
            return T

        return cls(run.camera["K"], run.camera["T_base_camera"], pose_at)

    @classmethod
    def for_labels(cls, labels_dir) -> "Projector":
        """Label-mask-only projector from `<labels>/camera.yaml`: training needs only K, so it has no pose."""
        cam = yaml.safe_load((Path(labels_dir) / "camera.yaml").read_text())
        return cls(cam["K"], np.eye(4), pose_at=None)

    def _T_world_camera(self, ts_ns: int) -> np.ndarray:
        return self.pose_at(ts_ns) @ self.T_base_camera

    def orientation_at(self, ts_ns: int) -> "tuple[float, float]":
        """(roll, pitch) of the base at `ts_ns` [rad], 'xyz' Euler."""
        roll, pitch, _ = Rotation.from_matrix(self.pose_at(ts_ns)[:3, :3]).as_euler("xyz")
        return float(roll), float(pitch)

    def project(self, table: pd.DataFrame, ts_ns: int, image_shape, max_future_sec: float = 10.0,
                min_dist: float = 0.5, max_dist: float = 3.0) -> pd.DataFrame:
        """Stances that start within `max_future_sec` after `ts_ns`, lie in front of the camera at a distance in
        [min_dist, max_dist] and fall inside the image. Columns: `COLUMNS`."""
        t = table[(table["t_start"] > ts_ns) & (table["t_start"] <= ts_ns + int(max_future_sec * 1e9))]
        if t.empty:
            return pd.DataFrame(columns=COLUMNS)
        T = self._T_world_camera(ts_ns)
        pts = (np.linalg.inv(T) @ np.column_stack([t[["x", "y", "z"]].values, np.ones(len(t))]).T).T[:, :3]
        dist = np.linalg.norm(pts, axis=1)
        keep = (pts[:, 2] > 0) & (dist >= min_dist) & (dist <= max_dist)
        t, pts, dist = t[keep], pts[keep], dist[keep]
        if t.empty:
            return pd.DataFrame(columns=COLUMNS)
        px = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.K, None)[0].reshape(-1, 2)
        h, w = image_shape[:2]
        inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        t, px, dist = t[inside], px[inside], dist[inside]
        # pitch of the ray through each pixel: optical-axis elevation + vertical angle off the principal point
        axis_pitch = np.arcsin(abs((T[:3, :3].T @ [0.0, 0.0, 1.0])[2]))
        pitch = axis_pitch + np.arctan2(px[:, 1] - self.K[1, 2], self.K[1, 1])
        return pd.DataFrame({"foot": t["foot"].values, "t_start": t["t_start"].values, "pixel_x": px[:, 0],
                             "pixel_y": px[:, 1], "distance": dist, "pitch": pitch,
                             "m_hard": t["m_hard"].values.astype(float), "m_soft": t["m_soft"].values.astype(float)})

    def _ellipse(self, shape, row) -> np.ndarray:
        """Depth-scaled filled ellipse of the foot at `row`, bool (H, W)."""
        px_diam = self.foot_diameter_px / max(row.distance, 1e-3)
        axes = (max(1, int(px_diam / 2)), max(1, int(px_diam * abs(np.sin(row.pitch)) / 2)))
        canvas = np.zeros(shape, np.uint8)
        cv2.ellipse(canvas, (int(row.pixel_x), int(row.pixel_y)), axes, 0, 0, 360, 1, -1)
        return canvas.astype(bool)

    def label_mask(self, footsteps: pd.DataFrame, shape, column: str, log: bool) -> np.ndarray:
        """float32 (H, W): mean of `column` over the ellipses covering each pixel, NaN elsewhere (log of it if
        `log`). NaN labels (still stances) are skipped; with `log`, a value <= 0 raises."""
        total, count = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        for row in footsteps.itertuples():
            value = float(getattr(row, column))
            if not np.isfinite(value):
                continue
            if log and value <= 0:
                raise ValueError(f"log mode needs strictly positive labels; {column} = {value}")
            foot = self._ellipse(shape, row)
            total[foot] += value
            count[foot] += 1.0
        mask = np.full(shape, np.nan, np.float32)
        mask[count > 0] = total[count > 0] / count[count > 0]
        return np.log(mask) if log else mask
