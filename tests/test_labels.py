import numpy as np
import pandas as pd
import yaml
from conftest import write_run

from tacos.labels import make_labels

T_BASE_CAM_DOWN = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0.1], [0, 0, 0, 1]]   # looking down, image up = backward


def stepping(run):
    """Make every leg take a stance 0.5 s after each image and point the camera down at the feet, so that every frame
    has a future footstep in view."""
    contacts = pd.read_csv(run / "contacts.csv")
    t = contacts["timestamp_ns"] * 1e-9
    for leg in ("LF", "RF", "LH", "RH"):
        contacts[leg] = ((t % 1.0 >= 0.5) & (t < 3.9)).astype(int)
    contacts.to_csv(run / "contacts.csv", index=False)
    camera = yaml.safe_load((run / "camera.yaml").read_text())
    camera.update(K=[[50.0, 0, 64], [0, 50.0, 48], [0, 0, 1]], T_base_camera=T_BASE_CAM_DOWN)
    (run / "camera.yaml").write_text(yaml.safe_dump(camera))
    return run


def test_labels_folder(slipping):
    out = make_labels(stepping(slipping))
    assert {"foot", "t_start", "m_hard", "m_soft", "still"} <= set(pd.read_csv(out / "stances.csv").columns)
    assert list(pd.read_csv(out / "frames.csv").columns) == ["timestamp_ns", "roll", "pitch", "history_front",
                                                            "history_hind"]
    assert {"timestamp_ns", "pixel_x", "pixel_y", "m_hard"} <= set(pd.read_csv(out / "footsteps.csv").columns)
    assert len(list((out / "images").glob("*.png"))) == 4
    camera = yaml.safe_load((out / "camera.yaml").read_text())
    assert (camera["height"], camera["width"]) == (48, 128) and np.isclose(camera["K"][1][2], 0.0)   # bottom half


def test_run_without_contacts_writes_empty_tables(tmp_path):
    out = make_labels(write_run(tmp_path / "empty", contact=0))
    for name in ("stances.csv", "frames.csv", "footsteps.csv"):
        table = pd.read_csv(out / name)
        assert table.empty and len(table.columns)
    assert (out / "camera.yaml").exists()
