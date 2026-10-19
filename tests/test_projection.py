import numpy as np
import pandas as pd
import pytest

from tacos.projection import Projector

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
T_BASE_CAM = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)   # optical axis = base +x
SHAPE = (480, 640)
PROJECTOR = Projector(K, T_BASE_CAM, lambda ts: np.eye(4))


def stance(x, t_start=1.0, value=2.0):
    """A stance `x` m ahead of and 0.5 m below the camera, starting `t_start` s after the image at t = 0."""
    return {"foot": "LF", "t_start": int(t_start * 1e9), "t_end": int((t_start + 0.3) * 1e9), "idx_start": 0,
            "idx_end": 1, "x": x, "y": 0.0, "z": -0.5, "still": False, "m_hard": value, "m_soft": value}


def project(*stances):
    return PROJECTOR.project(pd.DataFrame(stances), 0, SHAPE)


def test_point_ahead_projects_below_the_principal_point():
    (row,) = project(stance(2.0)).itertuples()
    assert row.pixel_x == pytest.approx(320, abs=1) and row.pixel_y > 240


@pytest.mark.parametrize("s", [stance(0.3), stance(4.0), stance(2.0, t_start=11.0), stance(2.0, t_start=-1.0)],
                         ids=["below the image", "too far", "too late", "already started"])
def test_stance_is_dropped(s):
    assert project(s).empty


def test_label_mask_averages_overlapping_footsteps():
    mask = PROJECTOR.label_mask(project(stance(2.0, value=1.0), stance(2.0, value=3.0)), SHAPE, "m_hard", log=False)
    assert np.nanmax(mask) == pytest.approx(2.0) and np.isnan(mask[0, 0])


def test_label_mask_skips_nan_scores():
    assert np.isnan(PROJECTOR.label_mask(project(stance(2.0, value=np.nan)), SHAPE, "m_hard", log=True)).all()


def test_log_label_mask_rejects_nonpositive_scores():
    with pytest.raises(ValueError, match="m_hard"):
        PROJECTOR.label_mask(project(stance(2.0, value=0.0)), SHAPE, "m_hard", log=True)
