"""tacos-download: fetch one GrandTour mission from Hugging Face and convert it to the run format (see README).

    tacos-download KAB-2 --out data          # a run name (tacos/config/grandtour_runs.json) or a mission id

Each topic is a tar of a zarr v2 group with float64 UNIX-second timestamps. Three conventions matter:

1. TF DIRECTION. The `tf` group's .zattrs stores T_child_parent for every frame, so building T_base_<frame> inverts
   every hop (`T_base_frame`). Reading it the natural way puts hdr_front 0.67 m / 119° off.
2. DLIO CHILD FRAME. `dlio_map_odometry` is the pose of `hesai_lidar` in `dlio_map` (the zarr does not say so; it
   reproduces GrandTour's dlio_map → base tf exactly). The base pose is T_map_lidar · T_lidar_base.
3. DISTORTION. The `images/hdr_front.tar` frames are raw equidistant fisheye. They are undistorted with Knew = K, so
   camera.yaml keeps the original K and needs no distortion coefficients.
"""
import argparse
import json
import shutil
import tarfile
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
import zarr
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from tacos.robot import JOINTS, LEGS

REPO       = "leggedrobotics/grand_tour_dataset"
TOPICS     = ("anymal_state_actuator", "anymal_state_state_estimator", "anymal_imu", "dlio_map_odometry", "tf", "hdr_front")
CAMERA     = "hdr_front"
LIDAR      = "hesai_lidar"                 # child frame of dlio_map_odometry
C_BASE_IMU = np.diag([-1.0, 1.0, -1.0])    # anymal_imu → base: 180° about y
RUNS       = json.loads((Path(__file__).parent / "config" / "grandtour_runs.json").read_text())


# ── tf ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _matrix(rotation: dict, translation: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([rotation[k] for k in "xyzw"]).as_matrix()
    T[:3, 3]  = [translation[k] for k in "xyz"]
    return T


def T_base_frame(tf: dict, frame: str) -> np.ndarray:
    """T_base_<frame> (maps a point in <frame> into base), walking frame → … → base and inverting each stored
    T_child_parent hop. Most sensors hang off box_base, which hangs off base; a few hang off base directly."""
    T, seen = np.eye(4), set()
    while frame != "base":
        if frame in seen or frame not in tf:
            raise KeyError(f"frame '{frame}' has no path to 'base' in the tf tree")
        seen.add(frame)
        e = tf[frame]
        T, frame = np.linalg.inv(_matrix(e["rotation"], e["translation"])) @ T, e["base_frame_id"]
    return T


# ── conversion (one zarr group → one run-format table) ───────────────────────────────────────────────────────────────
def _ns(seconds) -> np.ndarray:
    return np.round(np.asarray(seconds) * 1e9).astype(np.int64)


def _table(group, columns: dict) -> pd.DataFrame:
    return pd.DataFrame(columns, index=pd.Index(_ns(group["timestamp"][:]), name="timestamp_ns"))


def convert_joints(g) -> pd.DataFrame:
    """anymal_state_actuator: joint NN (0-11 = LF_HAA … RH_KFE) has measured arrays NN_state_joint_<field>."""
    fields = (("pos", "position"), ("vel", "velocity"), ("torque", "torque"))
    return _table(g, {f"{leg}_{j}_{f}": g[f"{i:02d}_state_joint_{name}"][:]
                      for i, (leg, j) in enumerate(product(LEGS, JOINTS)) for f, name in fields})


def convert_contacts(g) -> pd.DataFrame:
    """anymal_state_state_estimator: <LEG>_FOOT_state is 0 open, 1 closed, 2 slipping; contact = closed or slipping."""
    return _table(g, {leg: (g[f"{leg}_FOOT_state"][:] > 0).astype(np.uint8) for leg in LEGS})


def convert_imu(g) -> pd.DataFrame:
    """anymal_imu, rotated into base (rotation only, no lever arm)."""
    C = Rotation.from_quat([g.attrs["transform"]["rotation"][k] for k in "xyzw"]).as_matrix()
    if not np.allclose(C, C_BASE_IMU, atol=1e-9):
        raise ValueError(f"anymal_imu mounting {C.round(6).tolist()} is not diag(-1,1,-1)")
    gyro, acc = g["ang_vel"][:] @ C.T, g["lin_acc"][:] @ C.T
    return _table(g, {"gyro_x": gyro[:, 0], "gyro_y": gyro[:, 1], "gyro_z": gyro[:, 2],
                      "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2]})


def convert_poses(g, tf: dict) -> pd.DataFrame:
    """dlio_map_odometry (dlio_map ← hesai_lidar) → world ← base."""
    T_lidar_base = np.linalg.inv(T_base_frame(tf, LIDAR))
    R_wl = Rotation.from_quat(g["pose_orien"][:])
    p    = g["pose_pos"][:] + R_wl.apply(T_lidar_base[:3, 3])
    q    = (R_wl * Rotation.from_matrix(T_lidar_base[:3, :3])).as_quat()
    return _table(g, {"x": p[:, 0], "y": p[:, 1], "z": p[:, 2], "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3]})


# ── download ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def mission_files(mission: str) -> dict:
    """Repo path of every tar `download` needs; raises if the mission lacks one."""
    wanted = {t: f"{mission}/data/{t}.tar" for t in TOPICS} | {f"images/{CAMERA}": f"{mission}/images/{CAMERA}.tar"}
    try:
        present = {e.path for e in HfApi().list_repo_tree(REPO, path_in_repo=mission, repo_type="dataset", recursive=True)}
    except EntryNotFoundError:
        present = set()
    if missing := [k for k, v in wanted.items() if v not in present]:
        raise FileNotFoundError(f"{', '.join(missing)} not found in GrandTour mission {mission} ({REPO})")
    return wanted


def _write_images(tar_path: Path, cam: dict, stamps: np.ndarray, out: Path, every: int, mission: str) -> int:
    """Undistort (equidistant fisheye → pinhole with the same K) and write every `every`-th frame from the first one,
    streamed from the tar (frame i ↔ stamp i)."""
    K, D = np.array(cam["K"], np.float64).reshape(3, 3), np.array(cam["D"], np.float64)[:4]
    with tarfile.open(tar_path) as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile() and m.name.endswith((".jpeg", ".jpg", ".png"))),
                         key=lambda m: m.name)
        if len(members) != len(stamps):   # never pad or guess: a shifted pairing would misplace every projection
            raise ValueError(f"{mission}: {len(members)} {CAMERA} images vs {len(stamps)} timestamps, cannot pair them")
        keep = range(0, len(stamps), every)
        for i in tqdm(keep, desc="images"):
            img = cv2.imdecode(np.frombuffer(tar.extractfile(members[i]).read(), np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(str(out / f"{stamps[i]}.png"), cv2.fisheye.undistortImage(img, K, D, Knew=K))
    return len(keep)


def download(run: str, out_root: Path, image_every: int = 10, keep_raw: bool = False) -> Path:
    """Fetch one mission (a run name or a GrandTour mission id) and write it to `out_root/<run>` in the run
    format. Raises before writing anything if the mission lacks one of the needed topics."""
    mission = RUNS.get(run, run)
    files   = mission_files(mission)

    out = Path(out_root) / run
    raw = out / "raw"
    (out / "images").mkdir(parents=True, exist_ok=True)
    local = {k: Path(hf_hub_download(REPO, v, repo_type="dataset", local_dir=raw)) for k, v in files.items()}
    for t in TOPICS:
        with tarfile.open(local[t]) as tar:
            tar.extractall(raw, filter="data")
    group = {t: zarr.open_group(raw / t, mode="r") for t in TOPICS}
    tf    = group["tf"].attrs["tf"]

    convert_joints(group["anymal_state_actuator"]).to_csv(out / "joints.csv")
    convert_contacts(group["anymal_state_state_estimator"]).to_csv(out / "contacts.csv")
    convert_imu(group["anymal_imu"]).to_csv(out / "imu.csv")
    convert_poses(group["dlio_map_odometry"], tf).to_csv(out / "poses.csv")

    cam = group[CAMERA].attrs["camera_info"]
    if cam["distortion_model"] != "equidistant":
        raise ValueError(f"{mission}: expected equidistant {CAMERA} distortion, got {cam['distortion_model']!r}")
    (out / "camera.yaml").write_text(yaml.safe_dump({
        "width": cam["width"], "height": cam["height"], "K": np.reshape(cam["K"], (3, 3)).tolist(),
        "T_base_camera": T_base_frame(tf, CAMERA).tolist()}))
    n = _write_images(local[f"images/{CAMERA}"], cam, _ns(group[CAMERA]["timestamp"][:]), out / "images", image_every,
                      mission)

    if not keep_raw:
        shutil.rmtree(raw)
    print(f"wrote {out} (mission {mission}): {n} images")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="run name (e.g. KAB-2) or GrandTour mission id (e.g. 2024-11-14-15-22-43)")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--image-every", type=int, default=10, help="keep every Nth camera frame (10 Hz → 1 Hz)")
    ap.add_argument("--keep-raw", action="store_true", help="keep the downloaded tars and zarr groups in <run>/raw")
    a = ap.parse_args()
    download(a.run, a.out, a.image_every, a.keep_raw)


if __name__ == "__main__":
    main()
