import numpy as np

from tacos.robot import LegKinematics, Run


def test_jacobian_matches_finite_differences():
    leg, q, eps = LegKinematics("LF"), np.array([[0.1, 0.6, -1.1]]), 1e-6
    J = leg.jacobian(q)[0]
    for j in range(3):
        dq = q.copy()
        dq[0, j] += eps
        assert np.allclose(J[:, j], (leg.foot_position(dq)[0] - leg.foot_position(q)[0]) / eps, atol=1e-4)


def test_right_leg_mirrors_left_leg():
    q = np.array([[0.0, 0.6, -1.1]])
    lf = LegKinematics("LF").foot_position(q)[0]
    rf = LegKinematics("RF").foot_position(q * [-1, 1, 1])[0]
    assert np.allclose(lf * [1, -1, 1], rf, atol=1e-9)


def test_run_is_resampled_on_the_joint_timestamps(standing):
    run = Run.load(standing)
    assert run.hz == 400.0
    assert run.imu.index.equals(run.joints.index) and run.contacts.index.equals(run.joints.index)
    assert run.foot_pos("LF").shape == (1600, 3)
