"""tacos-labels: run format -> `<run>/labels/` (stances.csv, frames.csv, footsteps.csv, camera.yaml, images/).

Stances are detected (`stances`), scored by the smoother (`scores`) and projected into every image of the run
(`projection`). A frame is kept only if at least one future footstep falls inside it; by default the image is cropped
to its bottom half (the ground in front of the robot) and K is shifted to match.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from tacos.projection import COLUMNS, Projector
from tacos.robot import Run
from tacos.scores import compute_scores, history_at, history_tables
from tacos.stances import stance_masks, stance_table

MIN_DIST, MAX_DIST, MAX_FUTURE_SEC = 0.5, 3.0, 10.0   # [m], [m], [s]: footsteps kept in a frame
FRAME_COLUMNS = ["timestamp_ns", "roll", "pitch", "history_front", "history_hind"]
FOOTSTEP_COLUMNS = ["timestamp_ns", *COLUMNS]


def make_labels(run_dir: Path, crop_bottom_half: bool = True) -> Path:
    """Write the labels format into `<run_dir>/labels/` and return that folder."""
    print(f"{Path(run_dir).name}: loading and detecting stances", flush=True)
    run = Run.load(run_dir)
    if run.camera is None:
        raise FileNotFoundError(f"{Path(run_dir) / 'camera.yaml'} not found: it is needed to project the footsteps")
    out = Path(run_dir) / "labels"
    (out / "images").mkdir(parents=True, exist_ok=True)

    masks = stance_masks(run)
    table = stance_table(run, masks)
    print(f"{run.name}: {len(table)} stances over {len(run.joints) / run.hz / 60:.1f} min", flush=True)
    if table.empty:   # no stance: skip the smoother, keep the columns
        table = table.assign(m_hard=pd.Series(dtype=float), m_soft=pd.Series(dtype=float))
    else:
        table = compute_scores(run, masks, table)
    table.to_csv(out / "stances.csv", index=False)

    projector = Projector.for_run(run)
    top = run.camera["height"] // 2 if crop_bottom_half else 0   # rows cut off the top of every image
    projector.K[1, 2] -= top
    shape = (run.camera["height"] - top, run.camera["width"])
    history = history_tables(table, "m_hard", 10)
    frames, footsteps = [], []
    for path in tqdm(run.images, desc=f"{run.name}: images", unit="image"):
        ts, image = int(path.stem), cv2.imread(str(path))[top:]
        rows = projector.project(table, ts, shape, MAX_FUTURE_SEC, MIN_DIST, MAX_DIST)
        if rows.empty:
            continue
        cv2.imwrite(str(out / "images" / f"{ts}.png"), image)
        footsteps.append(rows.assign(timestamp_ns=ts))
        frames.append([ts, *projector.orientation_at(ts), *history_at(history, ts).values()])

    pd.DataFrame(frames, columns=FRAME_COLUMNS).to_csv(out / "frames.csv", index=False)
    fs = pd.concat(footsteps, ignore_index=True) if footsteps else pd.DataFrame(columns=FOOTSTEP_COLUMNS)
    fs[FOOTSTEP_COLUMNS].to_csv(out / "footsteps.csv", index=False)
    h, w = shape
    (out / "camera.yaml").write_text(yaml.safe_dump({"width": int(w), "height": int(h),
                                                     "K": np.asarray(projector.K).tolist()}))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate TACOS footstep labels for one or more runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--no-crop", action="store_true", help="keep the full image instead of its bottom half")
    args = parser.parse_args(argv)
    for run_dir in args.run_dirs:
        print(f"wrote {make_labels(run_dir, crop_bottom_half=not args.no_crop)}")


if __name__ == "__main__":
    main()
