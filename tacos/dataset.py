"""Frames + footsteps -> tensors: `FootstepDataset`, per-run contiguous validation folds, `TacosDataModule`."""
import json
import warnings
from pathlib import Path
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from tacos.model import CONTEXT_VARIANTS, IMAGE_MEAN, IMAGE_STD
from tacos.projection import Projector

DEFAULT_TEST_RUNS = json.loads((Path(__file__).parent / "config" / "grandtour_runs.json").read_text())["test"]


def contiguous_fold(n: int, fold: int, n_splits: int = 10):
    """Validation = chronological decile [(2k-1)*10%, 2k*10%) of `n` frames (k = fold in 1..5); train = the rest."""
    if n_splits != 10:
        raise ValueError(f"n_splits must be 10, got {n_splits}")
    if n < 10:
        raise ValueError(f"contiguous folds need at least 10 frames, got {n}")
    start, end = n * (2 * fold - 1) // 10, n * (2 * fold) // 10
    return np.r_[0:start, end:n], np.arange(start, end)


class FootstepDataset(Dataset):
    """One sample per frame of `<run>/labels/frames.csv`: `(image (3,H,W), context (C,), target (1,H,W)[, footsteps])`.

    The target is the log of `score` splatted over the footsteps seen in the frame, NaN elsewhere. Each footstep is
    `(run, foot, t_start, log_value, mask, raw_value)`, `log_value` being the mean of that footstep's own mask."""

    def __init__(self, label_dirs, score: Literal["m_hard", "m_soft"], context_fields, train: bool,
                 input_size=(512, 704), return_footsteps=False):
        h, w = input_size
        self.score, self.fields, self.train, self.return_footsteps = score, list(context_fields), train, return_footsteps
        crop = [A.CropNonEmptyMaskIfExists(height=h, width=w, p=1)] if train else []   # eval keeps the full frame
        self.transform = A.Compose([A.SmallestMaxSize(max_size=h, p=1), *crop, A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD, p=1),
                                    ToTensorV2()], additional_targets={"label_map": "mask"})
        self.samples, self.projectors, self.footsteps = [], {}, {}
        for d in map(Path, sorted(label_dirs)):
            self.projectors[d] = Projector.for_labels(d)
            frames = pd.read_csv(d / "frames.csv").sort_values("timestamp_ns")
            self.footsteps[d] = dict(tuple(pd.read_csv(d / "footsteps.csv").groupby("timestamp_ns")))
            self.samples += [(d, row) for row in frames.itertuples(index=False)]

    def __len__(self):
        return len(self.samples)

    def context_matrix(self) -> np.ndarray:
        """(N, C) unflipped context of every frame, for the normalization statistics."""
        return np.array([[getattr(row, f) for f in self.fields] for _, row in self.samples], np.float32).reshape(
            len(self), len(self.fields))

    def __getitem__(self, idx):
        d, frame = self.samples[idx]
        image = cv2.cvtColor(cv2.imread(str(d / "images" / f"{frame.timestamp_ns}.png")), cv2.COLOR_BGR2RGB)
        rows = self.footsteps[d].get(frame.timestamp_ns, pd.DataFrame(columns=["foot"]))
        projector, shape = self.projectors[d], image.shape[:2]
        label = projector.label_mask(rows, shape, self.score, log=True) if len(rows) else np.full(shape, np.nan, np.float32)
        footsteps = self._footsteps(d, rows, projector, shape) if self.return_footsteps else []
        inputs = {"image": image, "mask": np.isfinite(label).astype(np.uint8), "label_map": label}
        if footsteps:
            inputs["masks"] = np.stack([f[4] for f in footsteps])
        out = self.transform(**inputs)
        image, target = out["image"].float(), torch.as_tensor(out["label_map"]).unsqueeze(0)
        masks = [torch.as_tensor(m) for m in out["masks"]] if footsteps else []
        context = torch.tensor([getattr(frame, f) for f in self.fields], dtype=torch.float32)
        if self.train and torch.rand(()) < 0.5:
            image, target, masks = image.flip(-1), target.flip(-1), [m.flip(-1) for m in masks]
            if "roll" in self.fields:
                context[self.fields.index("roll")] *= -1
        if not self.return_footsteps:
            return image, context, target
        return image, context, target, [(*f[:4], m, f[5]) for f, m in zip(footsteps, masks)]

    def _footsteps(self, d, rows, projector, shape):
        """Per-footstep records on the full frame; footsteps with no labelled pixel are skipped."""
        records = []
        for row in rows.itertuples():
            mask = projector.label_mask(rows.loc[[row.Index]], shape, self.score, log=True)
            valid = np.isfinite(mask)
            if valid.any():
                records.append((d.parent.name, row.foot, int(row.t_start), float(mask[valid].mean()),
                                valid.astype(np.uint8), float(getattr(row, self.score))))
        return records



def _n_frames(labels_dir) -> int:
    try:
        return len(pd.read_csv(labels_dir / "frames.csv"))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return 0


def footstep_collate(batch):
    """Default-collate the tensors and keep each image's variable-length footstep list (`return_footsteps=True`)."""
    return (*default_collate([s[:-1] for s in batch]), [s[-1] for s in batch])


class TacosDataModule(LightningDataModule):
    """Every run under `data_root` with a `labels/` folder, except `test_runs`, split per run by `contiguous_fold`.
    After `setup("fit")`, `context_mean` / `context_std` hold the train-frame statistics to pass to `TacosNet`."""

    def __init__(self, data_root, test_runs, score, context, fold, batch_size, input_size=(512, 704), num_workers=4):
        super().__init__()
        self.data_root, self.test_runs, self.score, self.fields = Path(data_root), list(test_runs), score, CONTEXT_VARIANTS[context]
        self.fold, self.batch_size, self.input_size, self.num_workers = fold, batch_size, input_size, num_workers
        self.context_mean = self.context_std = None

    def setup(self, stage="fit"):
        runs = [d for d in sorted(self.data_root.iterdir()) if (d / "labels").is_dir()]
        if not runs:
            raise FileNotFoundError(f"no <run>/labels folders under {self.data_root} (--data-root is the folder of "
                                    "runs)")
        if missing := sorted(set(self.test_runs) - {d.name for d in runs}):
            raise FileNotFoundError(f"test runs {missing} have no labels folder under {self.data_root}")
        dirs, short = [], []
        for d in runs:
            if d.name not in self.test_runs:
                (dirs if _n_frames(d / "labels") >= 10 else short).append(d / "labels")
        if short:
            warnings.warn(f"skipping runs with fewer than 10 labelled frames: {[d.parent.name for d in short]}")
        if not dirs:
            raise ValueError(f"no training run with at least 10 labelled frames under {self.data_root}")
        args = (dirs, self.score, self.fields)
        train_ds = FootstepDataset(*args, train=True, input_size=self.input_size)
        val_ds = FootstepDataset(*args, train=False, input_size=self.input_size)
        splits, start = [], 0
        for d in dirs:                                       # samples are grouped by run, in `dirs` order
            n = sum(1 for s, _ in train_ds.samples if s == d)
            splits.append((start + (a := contiguous_fold(n, self.fold))[0], start + a[1]))
            start += n
        train_idx, val_idx = (np.concatenate(x) for x in zip(*splits))
        self.train_data, self.val_data = Subset(train_ds, train_idx), Subset(val_ds, val_idx)
        if self.fields:
            context = train_ds.context_matrix()[train_idx].astype(np.float64)
            with warnings.catch_warnings():              # NaN frames are ignored here; TacosNet maps them to the mean
                warnings.simplefilter("ignore", RuntimeWarning)
                self.context_mean, self.context_std = (f(context, 0).astype(np.float32) for f in (np.nanmean, np.nanstd))
            self.context_mean[np.isnan(self.context_mean)] = 0.0
            self.context_std[~(self.context_std >= 1e-6)] = 1.0      # constant or all-NaN columns

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, max(1, self.batch_size // 2), num_workers=self.num_workers)
