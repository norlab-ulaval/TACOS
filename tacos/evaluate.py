"""tacos-evaluate: footstep-level Spearman and MAE of trained folds on their held-out test runs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader

from tacos.dataset import FootstepDataset, footstep_collate
from tacos.model import CONTEXT_VARIANTS
from tacos.predict import load_model

COLUMNS = ["n", "spearman", "mae", "mae_log"]


def _row(df):
    label, prediction = df.label.to_numpy(float), df.prediction.to_numpy(float)
    rho = stats.spearmanr(label, prediction)[0] if len(df) > 1 else np.nan
    return pd.Series({"n": len(df), "spearman": rho, "mae": np.abs(label - prediction).mean(),
                      "mae_log": np.abs(np.log(label) - np.log(prediction)).mean()})


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per `run` plus `pooled` (all footsteps together, not the mean of the runs); labels in label units."""
    rows = {run: _row(g) for run, g in df.groupby("run")}
    table = pd.DataFrame({**rows, "pooled": _row(df)}).T[COLUMNS]
    return table.astype({"n": int})


@torch.no_grad()
def footstep_predictions(model, label_dirs, score, fields, input_size, device) -> pd.DataFrame:
    """One prediction per footstep: exp of the predicted log map, averaged over the footstep's mask, then over the
    frames it appears in (label units). Masks that vanish at the prediction grid are skipped."""
    dataset = FootstepDataset(label_dirs, score, fields, train=False, input_size=input_size, return_footsteps=True)
    records = {}
    for image, context, _, footsteps in DataLoader(dataset, 1, collate_fn=footstep_collate):
        if not footsteps[0]:
            continue
        cost = model(image.to(device), context.to(device) if fields else None)[0, 0].float().exp().cpu()   # label units
        for run, foot, t_start, _, mask, value in footsteps[0]:
            region = mask > 0
            if region.shape != cost.shape:                   # nearest resize to the prediction grid
                region = torch.nn.functional.interpolate(region[None, None].float(), size=cost.shape)[0, 0] > 0
            if region.any():
                records.setdefault((run, foot, t_start), {"label": value, "cost": []})["cost"].append(cost[region].mean().item())
    return pd.DataFrame([{"run": r, "foot": f, "t_start": t, "label": v["label"], "prediction": np.mean(v["cost"])}
                         for (r, f, t), v in records.items()],
                        columns=["run", "foot", "t_start", "label", "prediction"])


def evaluate_fold(fold_dir, data_root=None, device="cpu") -> pd.DataFrame:
    config = yaml.safe_load((fold_dir / "config.yaml").read_text())
    data_root = Path(data_root or config["data_root"])
    dirs = [data_root / run / "labels" for run in config["test_runs"]]
    if missing := [d.parent.name for d in dirs if not d.is_dir()]:
        raise FileNotFoundError(f"{fold_dir}: test runs {missing} have no labels folder under {data_root} "
                                "(pass --data-root if the data moved)")
    model = load_model(fold_dir / "model.ckpt", device)
    df = footstep_predictions(model, dirs, "m_" + config["score"], CONTEXT_VARIANTS[config["context"]],
                              tuple(config["input_size"]), device)
    df.to_csv(fold_dir / "footsteps.csv", index=False)
    table = metrics(df)
    table.to_csv(fold_dir / "metrics.csv")
    print(f"{fold_dir.name}\n{table.to_string()}")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="A tacos-train output folder holding fold_* directories.")
    parser.add_argument("--data-root", help="Overrides the data root recorded in each fold's config.yaml.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    folds = sorted(args.run_dir.glob("fold_*"))
    if not folds:
        parser.error(f"no fold_* directory under {args.run_dir}")
    tables = {fold.name: evaluate_fold(fold, args.data_root, args.device) for fold in folds}
    if len(folds) > 1:
        stacked = pd.concat(tables, names=["fold", "run"])
        by_run = stacked.groupby("run", sort=False)
        extra = {name: by_run.agg(name) for name in ("median", "mean", "std")}
        pd.concat([stacked, pd.concat(extra, names=["fold", "run"])]).to_csv(args.run_dir / "summary.csv")


if __name__ == "__main__":
    main()
