"""Per-stance TACOS scores from the smoother residuals, and the history context of each frame.

- m_hard: the maximum of the HARD residuals d_ik over the stance;
- m_soft: their 95th percentile under the SOFT smoother;
- still stances (robot standing, no locomotion) get NaN for both.
"""
from typing import Literal

import numpy as np
import pandas as pd

from tacos.robot import FRONT, HIND, LEGS, Run
from tacos.smoother import HARD, SOFT, ResidualSmoother

AGGREGATIONS = {"max": np.nanmax, "p95": lambda x: np.nanpercentile(x, 95)}
HISTORY_GROUPS = {"history_front": FRONT, "history_hind": HIND}


def aggregate(table: pd.DataFrame, residuals: pd.DataFrame, how: Literal["max", "p95"], column: str) -> pd.DataFrame:
    """Add `column` to a copy of the stance table: `how` over the stance foot's residual samples with time in
    [t_start, t_end] (inclusive). Stances with no finite sample, and still stances, get NaN."""
    fn, out = AGGREGATIONS[how], table.copy()
    out[column] = np.nan
    t = residuals.index.to_numpy(np.float64)
    for leg in LEGS:
        rows = out.index[out["foot"] == leg]
        signal = residuals[f"residual_{leg}"].to_numpy(np.float64)
        vals = []
        for t0, t1 in zip(out.loc[rows, "t_start"], out.loc[rows, "t_end"]):
            seg = signal[(t >= int(t0) * 1e-9) & (t <= int(t1) * 1e-9)]
            vals.append(float(fn(seg)) if np.isfinite(seg).any() else np.nan)
        out.loc[rows, column] = vals
    out.loc[out["still"].astype(bool), column] = np.nan
    return out


def compute_scores(run: Run, masks: dict[str, np.ndarray], table: pd.DataFrame) -> pd.DataFrame:
    """Stance table + m_hard = max(HARD residuals) and m_soft = P95(SOFT residuals)."""
    table = aggregate(table, ResidualSmoother(run, masks, HARD).run(), "max", "m_hard")
    return aggregate(table, ResidualSmoother(run, masks, SOFT).run(), "p95", "m_soft")


def history_tables(table: pd.DataFrame, column: str = "m_hard", window: int = 10) -> dict[str, pd.DataFrame]:
    """Per leg group (front / hind): (t_start, causal rolling mean of `column` over the `window` previous stances of
    the group). The stance's own value is excluded."""
    out = {}
    for name, legs in HISTORY_GROUPS.items():
        group = table[table["foot"].isin(legs)].sort_values("t_start")
        mean = group[column].shift(1).rolling(window, min_periods=1).mean()
        out[name] = pd.DataFrame({"t_start": group["t_start"].to_numpy(), "average": mean.to_numpy()})
    return out


def history_at(tables: dict[str, pd.DataFrame], ts_ns: int) -> dict[str, float]:
    """{history_front, history_hind} at `ts_ns`: the latest rolling mean at or before it, backfilled before a group's
    first value; NaN for a group with no stance."""
    out = {}
    for name, t in tables.items():
        if len(t) == 0:
            out[name] = float("nan")
            continue
        pos = max(int(np.searchsorted(t["t_start"].to_numpy(), ts_ns, side="right")) - 1, 0)
        out[name] = float(t["average"].bfill().to_numpy()[pos])
    return out
