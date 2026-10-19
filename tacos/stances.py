"""Stance detection. The onboard contact state is debounced, its liftoffs are pulled back to the last
loaded sample, and the surviving contacts become a table of stances with their touchdown position in the world."""
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_opening

from tacos.robot import LEGS, Run, lowpass

OPEN_SEC          = 0.1     # [s]   opening: drop contacts shorter than this (also the minimum stance length)
CLOSE_SEC         = 0.025   # [s]   closing: fill only sub-sample detector chatter
REFINE_WINDOW_SEC = 0.15    # [s]   how far back a liftoff may be pulled
REFINE_FZ_N       = 50.0    # [N]   the last Fz above this ends the stance
FZ_CUTOFF_HZ      = 10.0    # [Hz]  low-pass applied to Fz before the refinement
STILL_SPEED       = 0.1     # [m/s] mean base-frame foot speed below which a stance is "still"


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Boolean mask -> (start, end) index pairs, end exclusive."""
    edges  = np.diff(mask.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    ends   = np.where(edges == -1)[0] + 1
    if mask[0]:   starts = np.r_[0, starts]
    if mask[-1]:  ends   = np.r_[ends, len(mask)]
    return list(zip(starts.tolist(), ends.tolist()))


def debounce(mask: np.ndarray, n_open: int, n_close: int) -> np.ndarray:
    """Morphological opening (drop contacts shorter than `n_open` samples), then closing (fill gaps under `n_close`)."""
    return binary_closing(binary_opening(mask, np.ones(n_open)), np.ones(n_close))


def refine_liftoffs(mask: np.ndarray, fz: np.ndarray, n_window: int, threshold: float) -> np.ndarray:
    """Pull each liftoff back to the last sample with Fz > `threshold`, looking at most `n_window` samples back.
    Only stance ends move, so the number of stances is unchanged."""
    out = mask.copy()
    for s, e in segments(mask):
        start = max(s + 1, e - n_window)
        if start >= e:
            continue
        loaded = np.flatnonzero(fz[start:e] > threshold)
        out[(start + int(loaded[-1]) + 1 if len(loaded) else start):e] = False
    return out


def drop_short(mask: np.ndarray, n_min: int) -> np.ndarray:
    """Remove stances shorter than `n_min` samples (refinement can whittle a contact down to a stub). In 1-D, an
    opening drops exactly the runs shorter than the structuring element and leaves the others whole."""
    return binary_opening(mask, np.ones(n_min))


def stance_masks(run: Run) -> dict[str, np.ndarray]:
    """Per-leg stance masks: contact state -> debounce -> liftoff refinement on the low-passed Fz -> drop short."""
    hz = run.hz
    n_open, n_close, n_window = (max(1, int(sec * hz)) for sec in (OPEN_SEC, CLOSE_SEC, REFINE_WINDOW_SEC))
    out = {}
    for leg in LEGS:
        mask = debounce(run.contact_state(leg), n_open, n_close)
        fz   = lowpass(run.foot_force_z(leg).astype(float), hz, FZ_CUTOFF_HZ)
        out[leg] = drop_short(refine_liftoffs(mask, fz, n_window, REFINE_FZ_N), n_open)
    return out


def stance_table(run: Run, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per stance: foot, t_start, t_end (inclusive, ns), idx_start, idx_end (exclusive), x, y, z (world foot
    position at touchdown) and `still` (mean base-frame foot speed over the stance below STILL_SPEED)."""
    idx, speed, rows = run.joints.index, run.max_foot_speed(), []
    for leg, mask in masks.items():
        world = run.foot_pos_world(leg)
        for s, e in segments(mask):
            rows.append([leg, idx[s], idx[e - 1], s, e, *world[s], bool(np.nanmean(speed[s:e]) < STILL_SPEED)])
    return pd.DataFrame(rows, columns=["foot", "t_start", "t_end", "idx_start", "idx_end", "x", "y", "z", "still"])
