import gtsam
import numpy as np
from conftest import write_run

from tacos.robot import LEGS, Run
from tacos.smoother import HARD, SOFT, ResidualSmoother, contact_error


def peak_residuals(path, config):
    """Max residual per leg with every leg in 0.5 s stances split by 20 ms swings (the slip spans several stances)."""
    run = Run.load(path)
    stance = np.arange(len(run.joints)) % int(0.5 * run.hz) >= int(0.02 * run.hz)
    return ResidualSmoother(run, dict.fromkeys(LEGS, stance), config).run().max()


def test_soft_residual_singles_out_the_slipping_foot(slipping):
    peaks = peak_residuals(slipping, SOFT)
    assert peaks["residual_LF"] > 3 * peaks.drop("residual_LF").max()


def test_standing_robot_stays_within_one_sigma(standing):
    assert (peak_residuals(standing, HARD) < 1.0).all()


def test_hard_reads_every_sample_of_a_long_stance(tmp_path):
    """One 4 s stance per leg, slipping at 2.0-2.8 s, long after its first keyframe left the 1 s window: each sample
    must be read before its keyframe is marginalized out."""
    run = Run.load(write_run(tmp_path / "long", slip_leg="LF", slip_window=(2.0, 2.8), slip_m=0.3))
    smoother = ResidualSmoother(run, dict.fromkeys(LEGS, np.ones(len(run.joints), bool)), HARD)
    residuals = smoother.run()
    assert np.isfinite(residuals.iloc[smoother.kf].to_numpy()).all()
    peaks = residuals.max()
    assert peaks["residual_LF"] > 5 * peaks.drop("residual_LF").max()


def test_contact_error_jacobians_match_central_differences():
    rng = np.random.default_rng(0)
    pose = gtsam.Pose3(gtsam.Rot3.Expmap(rng.normal(size=3)), rng.normal(size=3))
    f, s, H, eps = rng.normal(size=3), rng.normal(size=3), [None, None], 1e-6
    e = contact_error(pose, f, s, H)
    assert np.allclose(e, pose.rotation().matrix().T @ (f - pose.translation()) - s)
    dx = [(contact_error(pose.retract(d), f, s) - contact_error(pose.retract(-d), f, s)) / (2 * eps)
          for d in np.eye(6) * eps]
    df = [(contact_error(pose, f + d, s) - contact_error(pose, f - d, s)) / (2 * eps) for d in np.eye(3) * eps]
    assert np.allclose(H[0], np.column_stack(dx), atol=1e-6) and np.allclose(H[1], np.column_stack(df), atol=1e-6)
