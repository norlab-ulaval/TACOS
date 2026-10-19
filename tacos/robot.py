"""ANYmal-D leg kinematics and the run-format loader (`Run`).

A run folder holds joints.csv, imu.csv, contacts.csv, poses.csv, camera.yaml and images/ (see README). `Run.load`
resamples the IMU and the contact states onto the joint timestamps, which are the time base of every later stage.
"""
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

LEGS   = ("LF", "RF", "LH", "RH")
JOINTS = ("HAA", "HFE", "KFE")
FRONT  = ("LF", "RF")
HIND   = ("LH", "RH")

# Fixed-joint translations [m], signs for LF; mirrored per leg in LegKinematics.
HAA_X  = 0.304      # base → HAA,  fore-aft
HAA_Y  = 0.109      # base → HAA,  lateral
HFE_X  = 0.069      # HAA  → HFE,  fore-aft
HFE_Y  = 0.006      # HAA  → HFE,  lateral
KFE_Y  = 0.1805     # HFE  → KFE,  lateral
KFE_Z  = -0.285     # HFE  → KFE,  vertical
FOOT_X = 0.10       # KFE  → FOOT, fore-aft
FOOT_Y = 0.01744    # KFE  → FOOT, lateral
FOOT_Z = -0.39204   # KFE  → FOOT, vertical

LPF_CUTOFF_HZ, LPF_ORDER = 8.0, 2


def infer_hz(index) -> float:
    """Sample rate [Hz] from a nanosecond timestamp index (median spacing)."""
    if len(index) < 2:
        return 0.0
    med_dt = np.median(np.diff(np.asarray(index, dtype=np.int64)))
    return float(1e9 / med_dt) if med_dt > 0 else 0.0


def lowpass(values: np.ndarray, hz: float, cutoff: float = LPF_CUTOFF_HZ, order: int = LPF_ORDER) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter along time (axis 0), for (N,) or (N, D) input."""
    if len(values) < 8 or hz <= 0:
        return values
    nyq = 0.5 * hz
    if cutoff >= nyq:
        return values
    b, a = butter(order, cutoff / nyq, btype="low")
    try:
        return filtfilt(b, a, values, axis=0)
    except ValueError:
        return values


class LegKinematics:
    """Vectorised forward kinematics of one ANYmal-D leg, in the base frame.

    Joint chain::

        base ─ p_base_haa ─[Rx(q_HAA)]→ HAA ─ p_haa_hfe ─[Ry(q_HFE)]→ HFE ─ p_hfe_kfe ─[Ry(q_KFE)]→ KFE ─ p_kfe_foot → FOOT

    Every offset is the LF offset with y negated on right legs and x negated on hind legs. All joint axes keep the
    same direction on every leg, so a right leg mirrors its left twin (y → -y) when its HAA angle is negated.
    `q`, `dq` and `tau` are (N, 3) arrays in HAA/HFE/KFE order.
    """

    def __init__(self, leg: str):
        if leg not in LEGS:
            raise ValueError(f"leg must be one of {LEGS}, got '{leg}'")
        self.leg = leg
        side = 1.0 if "L" in leg else -1.0
        fore = 1.0 if "F" in leg else -1.0
        self.p_base_haa = np.array([HAA_X  * fore,  HAA_Y  * side,  0.0   ])
        self.p_haa_hfe  = np.array([HFE_X  * fore,  HFE_Y  * side,  0.0   ])
        self.p_hfe_kfe  = np.array([0.0,            KFE_Y  * side,  KFE_Z ])
        self.p_kfe_foot = np.array([FOOT_X * fore,  FOOT_Y * side,  FOOT_Z])

    def _forward_kinematics(self, q):
        """Joint rotations and joint origins in the base frame."""
        R_haa = Rotation.from_euler("x", q[:, 0:1])
        R_hfe = Rotation.from_euler("y", q[:, 1:2])
        R_kfe = Rotation.from_euler("y", q[:, 2:3])

        p_hfe  = self.p_base_haa + R_haa.apply(self.p_haa_hfe)
        p_kfe  = p_hfe           + R_haa.apply(R_hfe.apply(self.p_hfe_kfe))
        p_foot = p_kfe           + R_haa.apply(R_hfe.apply(R_kfe.apply(self.p_kfe_foot)))
        return R_haa, R_hfe, R_kfe, p_hfe, p_kfe, p_foot

    def foot_position(self, q) -> np.ndarray:
        """Foot position in the base frame, (N, 3) [m]."""
        return self._forward_kinematics(q)[5]

    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian of the foot position, (N, 3, 3), column j = joint j."""
        R_haa, R_hfe, _, p_hfe, p_kfe, p_foot = self._forward_kinematics(q)
        z_haa = np.broadcast_to([1., 0., 0.], (len(q), 3))
        z_hfe = R_haa.apply([0., 1., 0.])
        z_kfe = (R_haa * R_hfe).apply([0., 1., 0.])
        return np.stack([np.cross(z_haa, p_foot - self.p_base_haa), np.cross(z_hfe, p_foot - p_hfe),
                         np.cross(z_kfe, p_foot - p_kfe)], axis=2)

    def foot_velocity(self, q, dq) -> np.ndarray:
        """Foot linear velocity in the base frame, J·q̇, (N, 3) [m/s]."""
        return np.einsum("nij,nj->ni", self.jacobian(q), dq)

    def foot_force(self, q, tau) -> np.ndarray:
        """Force the leg exerts at the foot, base frame, from Jᵀ F = τ (Fahmi et al. without the h term), (N, 3).
        The ground reaction force on the foot is its negative (see `Run.foot_force_z`)."""
        return np.linalg.solve(self.jacobian(q).transpose(0, 2, 1), tau[:, :, None])[:, :, 0]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="timestamp_ns").sort_index()


@dataclass
class Run:
    """One run in the run format. `joints`, `imu` and `contacts` share the joints index (`timestamp_ns`); `poses`
    keeps its own timestamps (world ← base, quaternion xyzw)."""
    name:     str
    hz:       float
    joints:   pd.DataFrame
    imu:      pd.DataFrame
    contacts: pd.DataFrame
    poses:    "pd.DataFrame | None" = None
    camera:   "dict | None"         = None
    images:   list[Path]            = field(default_factory=list)

    @classmethod
    def load(cls, path) -> "Run":
        path   = Path(path)
        joints = _read(path / "joints.csv")
        ts     = joints.index.values.astype(np.int64)

        imu   = _read(path / "imu.csv")
        t_s, imu_ts = ts * 1e-9, imu.index.values.astype(np.int64) * 1e-9
        imu   = pd.DataFrame({c: np.interp(t_s, imu_ts, imu[c].values) for c in imu.columns}, index=joints.index)

        src   = _read(path / "contacts.csv")
        src_ts = src.index.values.astype(float)
        contacts = pd.DataFrame({leg: np.interp(ts.astype(float), src_ts, src[leg].values.astype(float)) > 0.
                                 for leg in LEGS}, index=joints.index)

        poses  = _read(path / "poses.csv") if (path / "poses.csv").exists() else None
        camera = yaml.safe_load((path / "camera.yaml").read_text()) if (path / "camera.yaml").exists() else None
        images = sorted((path / "images").glob("*.png"), key=lambda p: int(p.stem))
        return cls(path.name, infer_hz(joints.index), joints, imu, contacts, poses, camera, images)

    # ── Joint signals ─────────────────────────────────────────────────────────

    def _joint(self, leg: str, field: str) -> np.ndarray:
        return np.column_stack([self.joints[f"{leg}_{j}_{field}"].values for j in JOINTS])

    def q(self, leg: str) -> np.ndarray:   return self._joint(leg, "pos")
    def dq(self, leg: str) -> np.ndarray:  return self._joint(leg, "vel")
    def tau(self, leg: str) -> np.ndarray: return self._joint(leg, "torque")

    def contact_state(self, leg: str) -> np.ndarray:
        """Onboard contact state on the joints index, bool (N,)."""
        return self.contacts[leg].to_numpy(dtype=bool)

    # ── Feet, base frame ──────────────────────────────────────────────────────

    def foot_pos(self, leg: str) -> np.ndarray:
        return LegKinematics(leg).foot_position(self.q(leg))

    def foot_vel(self, leg: str) -> np.ndarray:
        return LegKinematics(leg).foot_velocity(self.q(leg), self.dq(leg))

    def max_foot_speed(self) -> np.ndarray:
        """Per-sample maximum base-frame foot speed across the four legs [m/s], (N,); drives the `still` test."""
        return np.column_stack([np.linalg.norm(self.foot_vel(leg), axis=1) for leg in LEGS]).max(axis=1)

    # ── World frame ───────────────────────────────────────────────────────────

    @cached_property
    def base_pose(self) -> "tuple[np.ndarray, np.ndarray]":
        """World ← base pose on the joints index: position (N, 3) by linear interpolation, roll/pitch/yaw (N, 3)
        ('xyz' Euler) by Slerp with timestamps clamped to the pose range."""
        if self.poses is None:
            raise FileNotFoundError(f"run '{self.name}' has no poses.csv")
        odom_ts   = self.poses.index.values.astype(np.int64)
        joints_ts = self.joints.index.values.astype(np.int64)
        ts_clamp  = np.clip(joints_ts, odom_ts[0], odom_ts[-1])
        xyz = np.column_stack([np.interp(joints_ts, odom_ts, self.poses[c].values) for c in "xyz"])
        R_src = Rotation.from_quat(self.poses[["qx", "qy", "qz", "qw"]].values)
        return xyz, Slerp(odom_ts, R_src)(ts_clamp).as_euler("xyz")

    def foot_pos_world(self, leg: str) -> np.ndarray:
        """Foot position in the world frame, (N, 3)."""
        xyz, rpy = self.base_pose
        return xyz + Rotation.from_euler("xyz", rpy).apply(self.foot_pos(leg))

    def foot_force_z(self, leg: str) -> np.ndarray:
        """Vertical ground reaction force on the foot [N], (N,): -Jᵀ⁻¹τ rotated by roll and pitch only (yaw = 0)."""
        rp = self.base_pose[1][:, :2]
        R_tilt = Rotation.from_euler("xyz", np.column_stack([rp, np.zeros(len(rp))]))
        return R_tilt.apply(-LegKinematics(leg).foot_force(self.q(leg), self.tau(leg)))[:, 2]
