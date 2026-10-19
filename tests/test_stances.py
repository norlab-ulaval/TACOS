import numpy as np

from tacos.robot import Run
from tacos.stances import debounce, drop_short, refine_liftoffs, stance_masks, stance_table


def test_debounce_drops_short_contacts_and_bridges_short_gaps():
    mask = np.zeros(200, bool)
    mask[10:20] = mask[50:120] = mask[125:190] = True
    out = debounce(mask, n_open=40, n_close=10)
    assert not out[10:20].any() and out[50:190].all()


def test_liftoff_is_pulled_back_to_the_last_loaded_sample():
    fz = np.where(np.arange(100) < 80, 100.0, 10.0)
    out = refine_liftoffs(np.ones(100, bool), fz, n_window=60, threshold=50.0)
    assert out[:80].all() and not out[80:].any()
    assert refine_liftoffs(np.ones(100, bool), np.full(100, 100.0), n_window=60, threshold=50.0).all()


def test_drop_short():
    mask = np.zeros(100, bool)
    mask[0:5] = mask[20:80] = True
    out = drop_short(mask, n_min=40)
    assert not out[0:5].any() and out[20:80].all()


def test_standing_robot_has_one_still_stance_per_leg(standing):
    run = Run.load(standing)
    table = stance_table(run, stance_masks(run))
    assert sorted(table["foot"]) == ["LF", "LH", "RF", "RH"] and table["still"].all()
