"""Fixed-lag IMU + leg-contact factor graph and its contact residuals d_ik.

The graph, built at 100 Hz keyframes k over a 1 s window (GTSAM ISAM2 fixed-lag smoother):
- states X_k = (R_k, p_k) world ← base and v_k (world velocity), IMU bias b_k (accelerometer, gyroscope);
- a gravity-levelled prior on X_0, v_0, b_0 and a preintegrated `CombinedImuFactor` between consecutive keyframes;
- footholds f_i (world points). Hard: one f_i per stance. Soft: one per stance sample, chained by a random walk;
  either way a new stance's first foothold gets an isotropic σ_f prior at its seeded position;
- per stance sample, a Huber-robust contact factor tying the FK foot position s_ik (base frame) to its foothold,
  in body form: e_ik = R_kᵀ(f_i − p_k) − s_ik, with the base-frame covariance Σ_ik (hard: the per-sample FK covariance
  R_S·I + R_α·J·Jᵀ; soft: (0.08 m)²·I).

d_ik = sqrt(e_ikᵀ Σ_ik⁻¹ e_ik) is read at the solution in one of two ways:
- hard (non-causal): each sample is read at the last update before its keyframe X_k is marginalized out of the window,
  so it has seen ~1 s of later measurements. A stance's score is final once its liftoff sample has been read; the
  samples still in the window at the end of the data are read at the final estimate.
- soft (causal): each foothold node's residual is read right after the update that added it.
"""
from dataclasses import dataclass

import gtsam
import numpy as np
import pandas as pd
from gtsam import FixedLagSmootherKeyTimestampMap, IncrementalFixedLagSmoother
from gtsam.symbol_shorthand import B, L, V, X
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from tacos.robot import LEGS, LegKinematics, Run

GRAVITY = 9.81
R_S     = 1.0e-4         # [m²]   FK model noise (Bloesch eq. 22)
R_ALPHA = 2.5e-5         # [rad²] joint encoder noise
ACC_NOISE,  GYRO_NOISE = 2e-2 ** 2, 8e-4 ** 2      # IMU white noise PSD [(m/s²)²/Hz], [(rad/s)²/Hz]
ACC_WALK,   GYRO_WALK  = 1.0e-6, 1.0e-8            # bias random walk PSD
INTEGRATION_NOISE      = 1.0e-8
PRIOR_POS, PRIOR_VEL, PRIOR_ROT = 1e-2, 0.5, 0.1   # initial σ [m], [m/s], [rad]
PRIOR_ACC_BIAS, PRIOR_GYRO_BIAS = 0.1, 0.01        # initial σ [m/s²], [rad/s]
RELINEARIZE_THRESHOLD  = 0.01


@dataclass(frozen=True)
class SmootherConfig:
    name: str
    random_walk: bool              # soft: per-sample foothold random-walk chain; hard: one foothold per stance
    contact_sigma: float | None    # None: per-sample FK covariance (hard); 0.08 m isotropic (soft)
    update_every: int              # keyframes between ISAM2 updates (hard 10, soft 1)
    causal: bool                   # soft: read d_ik right after its update; hard: just before X_k leaves the window
    keyframe_hz: float = 100.0
    lag_sec: float = 1.0
    foothold_prior_sigma: float = 0.5   # sigma_f [m]: prior on a new stance's first foothold
    random_walk_sigma: float = 0.2      # sqrt(0.04 m^2/s): foothold random walk
    huber_k: float = 1.345


HARD = SmootherConfig("hard", random_walk=False, contact_sigma=None, update_every=10, causal=False)
SOFT = SmootherConfig("soft", random_walk=True, contact_sigma=0.08, update_every=1, causal=True)


def _skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def contact_error(pose, f, s, H=None) -> np.ndarray:
    """Contact error, body form: e = Rᵀ(f − p) − s for pose X = (R, p) world ← base, foothold f (world) and FK foot
    s (base). If H is given, fills H[0] = ∂e/∂X (3×6, gtsam Pose3 tangent: rotation then body-frame translation,
    X·Exp(ξ)) and H[1] = ∂e/∂f (3×3). With R' = R(I + [ω]ₓ), p' = p + R·v: e' ≈ e + [Rᵀ(f − p)]ₓ·ω − v."""
    R = pose.rotation().matrix()
    a = R.T @ (np.asarray(f, float) - pose.translation())
    if H is not None:
        H[0] = np.hstack([_skew(a), -np.eye(3)])
        H[1] = R.T
    return a - np.asarray(s, float)


class ResidualSmoother:
    """Fixed-lag IMU + leg-contact factor graph. Returns the Mahalanobis contact residual d_ik = sqrt(eᵀ Σ⁻¹ e)
    of every stance sample, read according to the config."""

    def __init__(self, run: Run, masks: dict[str, np.ndarray], config: SmootherConfig):
        self.cfg     = config
        self.name    = run.name
        self.time_s  = run.joints.index.values.astype(np.int64) * 1e-9
        self.dt      = np.clip(np.diff(self.time_s), 1e-4, 0.1)
        self.acc     = run.imu[["acc_x", "acc_y", "acc_z"]].to_numpy(float)
        self.gyro    = run.imu[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(float)
        self.contact = np.column_stack([np.asarray(masks[leg], bool) for leg in LEGS])
        self.foot    = {leg: run.foot_pos(leg) for leg in LEGS}
        self.fk_cov  = {}
        for leg in LEGS:
            J = LegKinematics(leg).jacobian(run.q(leg))
            self.fk_cov[leg] = R_S * np.eye(3) + R_ALPHA * np.einsum("nij,nkj->nik", J, J)

    def run(self) -> pd.DataFrame:
        """Index: time [s]; columns residual_LF..residual_RH. Values only on stance keyframes, NaN elsewhere."""
        self.kf          = self._keyframes()
        self.residual    = np.full((len(self.time_s), len(LEGS)), np.nan)
        self.smoother    = IncrementalFixedLagSmoother(self.cfg.lag_sec, self._isam_params())
        self.imu_params  = self._imu_params()
        self.foothold    = [-1] * len(LEGS)     # latest foothold id per leg, -1 = swing
        self.t_foothold  = [0.0] * len(LEGS)    # time of the latest foothold node per leg (random walk)
        self.next_id     = 0
        self.pending     = []                   # (k, sample, leg index, foothold id) not read yet, in keyframe order
        self._new_batch()

        K = len(self.kf)
        for k in tqdm(range(K), desc=f"{self.name}: {self.cfg.name} smoother", unit="keyframe"):
            pose = self._initial_factors() if k == 0 else self._imu_factor(k)
            self._contact_factors(k, pose)
            if (k > 0 and k % self.cfg.update_every == 0) or k == K - 1:
                self._update(k)
        if not self.cfg.causal:   # samples still in the final window
            self._read_entries(self.pending, self.smoother.calculateEstimate())
        return pd.DataFrame({f"residual_{leg}": self.residual[:, j] for j, leg in enumerate(LEGS)},
                            index=pd.Index(self.time_s, name="time_s"))

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _keyframes(self) -> np.ndarray:
        """Joint-sample indices of the keyframes: every round(rate / keyframe_hz)-th sample, plus the last one."""
        step = max(1, int(round(1.0 / float(np.median(self.dt)) / self.cfg.keyframe_hz)))
        kf = np.arange(0, len(self.time_s), step)
        if kf[-1] != len(self.time_s) - 1:
            kf = np.append(kf, len(self.time_s) - 1)
        return kf.astype(int)

    def _isam_params(self) -> gtsam.ISAM2Params:
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(RELINEARIZE_THRESHOLD)   # offline: relinearize aggressively
        params.relinearizeSkip = 1
        # gtsam 4.3.0 rejects Cholesky pivots 2^12 below their diagonal (cholesky.cpp), which the strongly correlated
        # p/v chain of this graph trips (IndeterminateSystemException). QR has no such check; scores match Cholesky to
        # ~1e-7. Set through the enum: setFactorization() silently maps an unknown string to CHOLESKY.
        params.factorization = gtsam.ISAM2Params.Factorization.QR
        return params

    def _imu_params(self):
        """Preintegration noise (gravity along −z) for the `CombinedImuFactor`."""
        p = gtsam.PreintegrationCombinedParams.MakeSharedU(GRAVITY)
        p.setAccelerometerCovariance(np.eye(3) * ACC_NOISE)
        p.setGyroscopeCovariance(np.eye(3) * GYRO_NOISE)
        p.setIntegrationCovariance(np.eye(3) * INTEGRATION_NOISE)
        p.setBiasAccCovariance(np.eye(3) * ACC_WALK)
        p.setBiasOmegaCovariance(np.eye(3) * GYRO_WALK)
        # Legacy: the pre-4.3.0 IMU error, which the reference labels use; the 4.3.0 default, the SE_2(3) Logmap chart,
        # moves m_hard by up to 6e-3.
        p.setImuFactorErrorMode(gtsam.ImuFactorErrorMode.Legacy)
        return p

    def _level_attitude(self) -> np.ndarray:
        """World ← base rotation with yaw 0 and roll/pitch from the mean accelerometer over the first all-feet-down
        phase (fallback: the first 0.2 s)."""
        full = self.contact.all(axis=1)
        if full.any():
            start = int(np.argmax(full))
            end = start + (int(np.argmin(full[start:])) if not full[start:].all() else len(full) - start)
        else:
            start, end = 0, max(1, int(np.searchsorted(self.time_s - self.time_s[0], 0.2)))
        a = self.acc[start:end].mean(axis=0)
        roll, pitch = np.arctan2(a[1], a[2]), np.arctan2(-a[0], np.hypot(a[1], a[2]))
        return Rotation.from_euler("xyz", [roll, pitch, 0.0]).as_matrix()

    # ── Factors ───────────────────────────────────────────────────────────────

    def _new_batch(self):
        self.factors, self.values, self.stamps = gtsam.NonlinearFactorGraph(), gtsam.Values(), []

    def _add_state(self, k: int, pose, vel, bias):
        t = float(self.time_s[self.kf[k]])
        self.values.insert(X(k), pose)
        self.values.insert(V(k), vel)
        self.values.insert(B(k), bias)
        self.stamps += [(X(k), t), (V(k), t), (B(k), t)]

    def _initial_factors(self):
        """Gravity-levelled priors on X_0 (origin, yaw 0), v_0 = 0 and b_0 = 0."""
        pose = gtsam.Pose3(gtsam.Rot3(self._level_attitude()), gtsam.Point3(0.0, 0.0, 0.0))
        vel, bias = np.zeros(3), gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        self.factors.add(gtsam.PriorFactorPose3(X(0), pose, gtsam.noiseModel.Diagonal.Sigmas(
            np.array([PRIOR_ROT] * 3 + [PRIOR_POS] * 3))))
        self.factors.add(gtsam.PriorFactorVector(V(0), vel, gtsam.noiseModel.Isotropic.Sigma(3, PRIOR_VEL)))
        self.factors.add(gtsam.PriorFactorConstantBias(B(0), bias, gtsam.noiseModel.Diagonal.Sigmas(
            np.array([PRIOR_ACC_BIAS] * 3 + [PRIOR_GYRO_BIAS] * 3))))
        self._add_state(0, pose, vel, bias)
        self.bias, self.nav = bias, gtsam.NavState(pose, vel)
        return pose

    def _imu_factor(self, k: int):
        """Preintegrate the IMU between keyframes k−1 and k with the latest bias estimate, add the `CombinedImuFactor`
        and seed X_k, v_k by dead reckoning from the previous state. Returns the seeded pose."""
        pim = gtsam.PreintegratedCombinedMeasurements(self.imu_params, self.bias)
        for i in range(self.kf[k - 1], self.kf[k]):
            pim.integrateMeasurement(self.acc[i], self.gyro[i], float(self.dt[i]))
        self.factors.add(gtsam.CombinedImuFactor(X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pim))
        self.nav = pim.predict(self.nav, self.bias)
        self._add_state(k, self.nav.pose(), self.nav.velocity(), self.bias)
        return self.nav.pose()

    def _contact_factors(self, k: int, pose):
        """For each leg in stance at keyframe k: open or extend its foothold, then add the contact factor."""
        i, t = int(self.kf[k]), float(self.time_s[self.kf[k]])
        R, p = pose.rotation().matrix(), pose.translation()
        for j, leg in enumerate(LEGS):
            if not self.contact[i, j]:
                self.foothold[j] = -1
                continue
            seed = R @ self.foot[leg][i] + p
            if self.foothold[j] < 0:
                self._open_foothold(j, seed)
            elif self.cfg.random_walk:
                self._extend_foothold(j, seed, t)
            self.t_foothold[j] = t
            fid = self.foothold[j]
            self.stamps.append((L(fid), t))   # keep the foothold alive while it is used
            self.pending.append((k, i, j, fid))
            self.factors.add(self._contact_factor(X(k), L(fid), self.foot[leg][i], self._contact_cov(i, leg)))

    def _new_foothold(self, j: int, seed) -> int:
        fid, self.next_id = self.next_id, self.next_id + 1
        self.foothold[j] = fid
        self.values.insert(L(fid), gtsam.Point3(*map(float, seed)))
        return fid

    def _open_foothold(self, j: int, seed):
        """Touchdown: a new foothold f_i at the seeded foot position, with an isotropic σ_f prior."""
        fid = self._new_foothold(j, seed)
        prior = gtsam.noiseModel.Isotropic.Sigma(3, float(self.cfg.foothold_prior_sigma))
        self.factors.add(gtsam.PriorFactorPoint3(L(fid), gtsam.Point3(*map(float, seed)), prior))

    def _extend_foothold(self, j: int, seed, t: float):
        """Soft: a new foothold node tied to the previous one by a zero-mean random walk,
        σ = σ_rw·sqrt(Δt) with Δt floored at 1 ms."""
        prev = self.foothold[j]
        fid = self._new_foothold(j, seed)
        sigma = self.cfg.random_walk_sigma * np.sqrt(max(t - self.t_foothold[j], 1e-3))
        self.factors.add(gtsam.BetweenFactorPoint3(L(prev), L(fid), gtsam.Point3(0.0, 0.0, 0.0),
                                                   gtsam.noiseModel.Isotropic.Sigma(3, float(sigma))))

    def _contact_cov(self, i: int, leg: str) -> np.ndarray:
        """Σ of the contact factor: (contact_sigma)²·I, or the FK covariance R_S·I + R_α·J·Jᵀ of sample i."""
        if self.cfg.contact_sigma is not None:
            return np.diag(np.square([self.cfg.contact_sigma] * 3))
        return self.fk_cov[leg][i]

    def _contact_factor(self, x_key, l_key, s, cov):
        """The contact error (`contact_error`), Huber-robust, as a `CustomFactor` over (X_k, f_i)."""
        noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(self.cfg.huber_k),
                                               gtsam.noiseModel.Gaussian.Covariance(cov))

        def error(_factor, v, H):
            return contact_error(v.atPose3(x_key), v.atPoint3(l_key), s, H)

        return gtsam.CustomFactor(noise, gtsam.KeyVector([x_key, l_key]), error)

    # ── Solve and read ────────────────────────────────────────────────────────

    def _update(self, k: int):
        """Flush the batch into ISAM2, re-anchor the IMU seed and bias on X_k, then read the residuals that are due."""
        latest = {}
        for key, t in self.stamps:                  # a foothold recurs: keep its latest time
            latest[int(key)] = max(float(t), latest.get(int(key), -np.inf))
        stamps = FixedLagSmootherKeyTimestampMap()
        for key, t in latest.items():
            stamps.insert((key, t))
        self.smoother.update(self.factors, self.values, stamps)
        est = self.smoother.calculateEstimate()
        self.bias = est.atConstantBias(B(k))
        self.nav  = gtsam.NavState(est.atPose3(X(k)), est.atVector(V(k)))
        self._read_residuals(k, est)
        self._new_batch()

    def _read_residuals(self, k: int, est):
        """Soft: read every pending sample now. Hard: read the pending samples whose keyframe X_k the next update
        marginalizes (gtsam drops keys stamped before t_next − lag; 1 µs margin so a tie reads one update early).
        The foothold is stamped with its latest use, so it outlives every keyframe that used it."""
        n = len(self.pending)
        if not self.cfg.causal:
            t_out = float(self.time_s[self.kf[min(k + self.cfg.update_every, len(self.kf) - 1)]]) - self.cfg.lag_sec
            n = next((m for m, (kk, *_) in enumerate(self.pending) if self.time_s[self.kf[kk]] >= t_out + 1e-6), n)
        self._read_entries(self.pending[:n], est)
        del self.pending[:n]

    def _read_entries(self, entries, est):
        """d_ik = sqrt(eᵀ Σ⁻¹ e) at the current estimate, for entries still in the window."""
        for k, i, j, fid in entries:
            if not (est.exists(X(k)) and est.exists(L(fid))):
                continue
            e = contact_error(est.atPose3(X(k)), est.atPoint3(L(fid)), self.foot[LEGS[j]][i])
            self.residual[i, j] = float(np.sqrt(e @ np.linalg.solve(self._contact_cov(i, LEGS[j]), e)))
