import numpy as np
import pandas as pd
import pytest
import torch

from tacos.dataset import FootstepDataset, TacosDataModule, contiguous_fold

FIELDS = ["roll", "pitch", "history_front", "history_hind"]
SIZE = (96, 128)


def test_validation_fold_is_one_contiguous_decile():
    for fold in range(1, 6):
        train, val = contiguous_fold(100, fold)
        assert list(val) == list(range((2 * fold - 1) * 10, 2 * fold * 10))
        assert sorted([*train, *val]) == list(range(100))


def test_target_is_the_log_score_on_footstep_pixels(labels_root):
    _, _, target = FootstepDataset([labels_root / "RUN-1" / "labels"], "m_hard", [], train=False, input_size=SIZE)[0]
    labelled = target[target.isfinite()]
    assert len(labelled) and torch.allclose(labelled, torch.tensor(np.log(1.5), dtype=torch.float32))


def test_horizontal_flip_negates_roll_only(labels_root):
    dirs = [labels_root / "RUN-1" / "labels"]
    _, context, target = FootstepDataset(dirs, "m_hard", FIELDS, train=False, input_size=SIZE)[0]
    train_set = FootstepDataset(dirs, "m_hard", FIELDS, train=True, input_size=SIZE)
    torch.manual_seed(0)
    _, flipped_context, flipped_target = next(s for s in (train_set[0] for _ in range(50)) if s[1][0] == -context[0])
    assert torch.equal(flipped_context[1:], context[1:])
    assert torch.equal(flipped_target.isnan(), target.flip(-1).isnan())


def test_footstep_records(labels_root):
    dataset = FootstepDataset([labels_root / "RUN-2" / "labels"], "m_soft", [], train=False, input_size=SIZE,
                              return_footsteps=True)
    (run, foot, t_start, log_value, mask, value), = dataset[3][-1]
    assert (run, foot, t_start, value) == ("RUN-2", "LF", 3 * 10**9 - 5, 0.5)
    assert np.isclose(log_value, np.log(0.5)) and mask.shape == SIZE and mask.any()


def test_datamodule_split_and_context_stats(labels_root):
    data = TacosDataModule(labels_root, ["RUN-3"], "m_hard", "all", fold=1, batch_size=4, input_size=SIZE)
    data.setup("fit")
    assert len(data.train_data) == 2 * 27 and len(data.val_data) == 2 * 3     # fold 1 = frames 3-5 of each run
    assert data.context_mean.shape == data.context_std.shape == (4,)
    image, context, _ = next(iter(data.train_dataloader()))
    assert image.shape == (4, 3, *SIZE) and context.shape == (4, 4)


def test_unknown_test_run_raises(labels_root):
    with pytest.raises(FileNotFoundError, match="RUN-9"):
        TacosDataModule(labels_root, ["RUN-9"], "m_hard", "none", fold=1, batch_size=4).setup("fit")


def test_runs_under_ten_frames_are_skipped(labels_root):
    frames = labels_root / "RUN-1" / "labels" / "frames.csv"
    pd.read_csv(frames).head(5).to_csv(frames, index=False)
    (labels_root / "RUN-2" / "labels" / "frames.csv").write_text("")
    data = TacosDataModule(labels_root, [], "m_hard", "none", fold=1, batch_size=4, input_size=SIZE)
    with pytest.warns(UserWarning, match=r"\['RUN-1', 'RUN-2'\]"):
        data.setup("fit")
    assert len(data.train_data) + len(data.val_data) == 30


def test_context_stats_ignore_nan(labels_root):
    frames = labels_root / "RUN-1" / "labels" / "frames.csv"
    df = pd.read_csv(frames)
    df.loc[0, "pitch"] = np.nan
    df.to_csv(frames, index=False)
    data = TacosDataModule(labels_root, ["RUN-3"], "m_hard", "all", fold=1, batch_size=4, input_size=SIZE)
    data.setup("fit")
    assert np.isclose(data.context_mean[1], 0.05) and np.isfinite(data.context_std).all()
