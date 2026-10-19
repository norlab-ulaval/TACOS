from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
import yaml

from tacos import evaluate, train
from tacos.dataset import TacosDataModule
from tacos.predict import load_model


@pytest.mark.parametrize("context", ["none", "all"])
def test_train_then_evaluate(labels_root, tmp_path, monkeypatch, stub_encoder, context):
    recipe = yaml.safe_load(files("tacos").joinpath("config", "train.yaml").read_text())
    recipe["precision"] = "32-true"                       # bf16 autocast is very slow on CPUs without AMX
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    monkeypatch.setattr("sys.argv", ["tacos-train", "--config", str(tmp_path / "recipe.yaml"),
                                     "--data-root", str(labels_root), "--context", context, "--test-runs", "RUN-3",
                                     "--batch-size", "2", "--out", str(tmp_path), "--max-epochs", "1",
                                     "--input-size", "96", "128"])
    train.main()
    run_dir = next(tmp_path.glob(f"hard_{context}_*"))
    fold = run_dir / "fold_1"
    config = yaml.safe_load((fold / "config.yaml").read_text())
    assert config["test_runs"] == ["RUN-3"] and config["accumulate"] == 126
    if context == "all":                                  # the checkpoint carries the train-frame statistics
        data = TacosDataModule(labels_root, ["RUN-3"], "m_hard", context, 1, 2, (96, 128))
        data.setup("fit")
        model = load_model(fold / "model.ckpt")
        assert np.allclose(model.context_mean.numpy(), data.context_mean)
        assert np.allclose(model.context_std.numpy(), data.context_std)

    monkeypatch.setattr("sys.argv", ["tacos-evaluate", str(run_dir), "--data-root", str(labels_root),
                                     "--device", "cpu"])
    evaluate.main()
    metrics = pd.read_csv(fold / "metrics.csv", index_col=0)
    assert list(metrics.index) == ["RUN-3", "pooled"] and {"spearman", "mae", "mae_log"} <= set(metrics.columns)
    footsteps = pd.read_csv(fold / "footsteps.csv")
    assert list(footsteps.columns) == ["run", "foot", "t_start", "label", "prediction"]
    assert (footsteps.prediction > 0).all()
