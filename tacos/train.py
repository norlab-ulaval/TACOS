"""tacos-train: train the DPT predictor (optionally with gated cross-attention context) on one validation fold."""
import argparse
import math
import warnings
from importlib.resources import files
from datetime import datetime
from pathlib import Path

import torch
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.tuner import Tuner

from tacos.dataset import DEFAULT_TEST_RUNS, TacosDataModule
from tacos.model import CONTEXT_VARIANTS, TacosNet


def largest_fitting_batch(trainer, model, data, effective_batch_size) -> int:
    """Largest divisor of `effective_batch_size` that trains without running out of GPU memory, after Lightning's
    batch size finder (a few training steps per trial)."""
    fits = Tuner(trainer).scale_batch_size(model, datamodule=data, mode="binsearch", max_val=effective_batch_size)
    return max(d for d in range(1, fits + 1) if effective_batch_size % d == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Folder of runs, each with a labels/ folder (tacos-labels).")
    parser.add_argument("--score", choices=["hard", "soft"], default="hard")
    parser.add_argument("--context", choices=list(CONTEXT_VARIANTS), default="none")
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--test-runs", default=",".join(DEFAULT_TEST_RUNS), help="Comma-separated runs held out for evaluation.")
    parser.add_argument("--batch-size", type=int, help="Per-step batch size (default on a GPU: the largest that fits in "
                                                       "memory; without one, the recipe's). Gradient accumulation keeps "
                                                       "the recipe's effective batch.")
    parser.add_argument("--out", default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", metavar="PROJECT", help="Log to Weights & Biases instead of CSV (needs the wandb extra).")
    parser.add_argument("--config", help="Training recipe yaml (default: the recipe shipped as tacos/config/train.yaml).")
    parser.add_argument("--max-epochs", type=int, help="Debug override of the recipe's max_epochs.")
    parser.add_argument("--input-size", type=int, nargs=2, metavar=("H", "W"), help="Debug override of the 512 704 crop.")
    args = parser.parse_args()

    recipe = yaml.safe_load(Path(args.config).read_text() if args.config else files("tacos").joinpath("config", "train.yaml").read_text())
    effective = recipe["effective_batch_size"]
    batch_size = args.batch_size or recipe["batch_size"]
    input_size = args.input_size or recipe["input_size"]
    max_epochs = args.max_epochs or recipe["max_epochs"]
    test_runs = [r for r in args.test_runs.split(",") if r]
    seed_everything(args.seed, workers=True)

    fold_dir = Path(args.out) / f"{args.score}_{args.context}_{datetime.now():%Y%m%d_%H%M%S}" / f"fold_{args.fold}"
    fold_dir.mkdir(parents=True)
    data = TacosDataModule(args.data_root, test_runs, f"m_{args.score}", args.context, args.fold, batch_size, tuple(input_size))
    data.setup("fit")                                                # the context statistics go into the checkpoint
    stats = [None if x is None else x.tolist() for x in (data.context_mean, data.context_std)]   # plain lists: safe to unpickle
    model = TacosNet(args.context, recipe["optimizer"]["lr"], recipe["optimizer"]["weight_decay"],
                     recipe["scheduler"]["warmup_steps"], recipe["scheduler"]["min_lr"],
                     recipe["optimizer"]["gate_lr_scale"], recipe["loss"]["huber_delta"], *stats,
                     recipe["context"]["attention_dim"], recipe["context"]["heads"])
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(project=args.wandb, name=f"{args.score}_{args.context}_fold{args.fold}", save_dir=str(fold_dir))
    else:
        logger = CSVLogger(str(fold_dir), name="logs")
    stopping = recipe["early_stopping"]
    callbacks = [ModelCheckpoint(dirpath=fold_dir, filename="model", monitor=stopping["monitor"], mode="min", save_top_k=1),
                 EarlyStopping(monitor=stopping["monitor"], mode="min", patience=stopping["patience"],
                               min_delta=stopping["min_delta"])]
    # min_epochs vetoes any stop before start_epoch; patience still counts from epoch 0
    trainer = Trainer(max_epochs=max_epochs, min_epochs=min(stopping["start_epoch"], max_epochs),
                      gradient_clip_val=recipe["gradient_clip_val"], precision=recipe["precision"], logger=logger,
                      callbacks=callbacks, default_root_dir=str(fold_dir),
                      accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, log_every_n_steps=1)
    if args.batch_size is None and torch.cuda.is_available():
        batch_size = largest_fitting_batch(trainer, model, data, effective)
        print(f"batch size {batch_size}: the largest divisor of {effective} that fits in GPU memory")
    data.batch_size = batch_size
    trainer.accumulate_grad_batches = accumulate = math.ceil(effective / batch_size)
    if accumulate * batch_size != effective:
        warnings.warn(f"batch size {batch_size} does not divide {effective}: the effective batch is "
                      f"{accumulate * batch_size}.")

    config = {**recipe, "data_root": str(Path(args.data_root).resolve()), "score": args.score, "context": args.context,
              "fold": args.fold, "test_runs": test_runs, "seed": args.seed, "batch_size": batch_size,
              "accumulate": accumulate, "max_epochs": max_epochs, "input_size": list(input_size)}
    (fold_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    trainer.fit(model, data)
    print(f"Saved {fold_dir / 'model.ckpt'}")


if __name__ == "__main__":
    main()
