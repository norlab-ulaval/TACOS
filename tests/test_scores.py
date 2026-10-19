import numpy as np
import pandas as pd

from tacos.scores import aggregate, history_at, history_tables


def test_aggregate_max_and_p95_and_still_stances_are_nan():
    table = pd.DataFrame({"foot": ["LF", "LF", "RH"], "t_start": [0, 2 * 10**9, 0],
                          "t_end": [10**9, 3 * 10**9, 10**9], "still": [False, True, False]})
    t = np.arange(0, 3.0, 0.01)
    residuals = pd.DataFrame({"residual_LF": t, "residual_RF": np.nan, "residual_LH": np.nan, "residual_RH": 1.0},
                             index=t)
    m_hard = aggregate(table, residuals, "max", "m_hard")["m_hard"]
    assert np.isclose(m_hard[0], 1.0) and np.isnan(m_hard[1]) and np.isclose(m_hard[2], 1.0)
    m_soft = aggregate(table, residuals, "p95", "m_soft")["m_soft"]
    assert np.isclose(m_soft[0], np.percentile(t[t <= 1.0], 95))


def test_history_is_a_causal_rolling_mean_per_leg_group():
    table = pd.DataFrame({"foot": ["LF", "RF", "LF", "LH", "RH"], "t_start": [0, 10, 20, 5, 15],
                          "m_hard": [1.0, 3.0, 5.0, 7.0, 9.0]})
    tables = history_tables(table, "m_hard", window=10)
    # the value at a stance is the mean of the earlier stances of its group, never its own
    assert history_at(tables, 15) == {"history_front": 1.0, "history_hind": 7.0}
    assert history_at(tables, 25)["history_front"] == 2.0
    # before a group's first value, the first value is used
    assert history_at(tables, 2)["history_front"] == 1.0
