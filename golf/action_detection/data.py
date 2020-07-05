"""Data taken from https://github.com/wmcnally/golfdb,"""
import hashlib
import os
import subprocess
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io
from loguru import logger
from tqdm import tqdm
from PIL import Image

import golf.io

np.random.seed(42)


GOLFDB_COLUMNS = [
    "id",
    "youtube_id",
    "player",
    "sex",
    "club",
    "view",
    "slow",
    "events",
    "bbox",
    "split",
]

GOLFDB_MAT_PATH = "../../data/golfDB/golfDB.mat"
GOLFDB_ARRAYS_PATH = "../../data/golfDB/arrays/"

YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v={youtube_id}"


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def golfDB_from_mat() -> pd.DataFrame:
    """Majority of this code taken from https://github.com/wmcnally/golfdb/blob/master/data/generate_splits.py."""
    mdata = scipy.io.loadmat(GOLFDB_MAT_PATH)["golfDB"][0]
    mdtype = mdata.dtype

    ddata = {
        col: mdata[col]
        if col in {"bbox", "events"}
        else [item.item() for item in mdata[col]]
        for col in mdtype.names
    }
    df = pd.DataFrame(ddata)
    df.index = df.index.astype(np.int)
    df["events"] = df["events"].apply(np.squeeze)
    df["bbox"] = df["bbox"].apply(np.squeeze)

    df = df.drop(["split", "id"], axis=1)
    df["swing_id"] = (
        df["youtube_id"]
        + df["events"].apply(lambda a: str(a[0]))
        + df["bbox"].apply(lambda a: str(a[0]))
    ).apply(sha1)

    df = df[df["youtube_id"] != "RibG0A13urY"]

    return df


def download_videos(
    n: Optional[int] = None, return_ids: bool = False, df: Optional[pd.DataFrame] = None
) -> Optional[list]:
    """Download videos from YouTube."""
    if df is None:
        df = golfDB_from_mat()

    youtube_ids = df["youtube_id"].unique()

    if n is not None:
        youtube_ids = youtube_ids[:n]

    for youtube_id in tqdm(youtube_ids):
        youtube_url = YOUTUBE_VIDEO_URL.format(youtube_id=youtube_id)
        youtube_dl_str = f"youtube-dl -f 'best[ext=mp4],fps=30' -o '../../data/golfDB/videos/%(id)s.%(ext)s' {youtube_url}"
        subprocess.check_call(
            [youtube_dl_str],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    if return_ids:
        return youtube_ids


def crop_array(a: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    assert bbox.shape[0] == 4 and bbox.ndim == 1
    x = int(a.shape[2] * bbox[0])
    y = int(a.shape[1] * bbox[1])
    w = int(a.shape[2] * bbox[2])
    h = int(a.shape[1] * bbox[3])
    return a[:, y: (y + h), x: (x + w), :]


def load_array(swing_id: str) -> Tuple[np.ndarray, np.ndarray]:
    df = golfDB_from_mat()

    swing = df.query(f"swing_id == '{swing_id}'").iloc[0]
    video_id = swing["youtube_id"]

    filepath = f"../../data/golfDB/videos/{video_id}.mp4"

    X = golf.io.read_video(filepath, rot=False)

    X = crop_array(X, swing["bbox"])

    logger.info(f"Shape of video: {X.shape}")

    events = np.squeeze(swing["events"])

    y = np.zeros((X.shape[0], 9))
    y[:, -1] = 1  # set all no event class to be 1
    for i, event in enumerate(events):
        if i != 0 and i != 9:  # if not start of clip or end of clip
            y[event, i - 1] = 1
            y[event, -1] = 0

    # randomly sample the array keeping all the event frames
    random_frames = np.random.choice(
        np.setdiff1d(np.arange(events[0], events[-1]), events[1:-1]),
        size=20,
        replace=False,
    )
    frames = np.concatenate((random_frames, events[1:-1]))

    X = X[frames, :, :, :]
    y = y[frames, :]

    return X, y


def preprocess_videos(view_type: str):
    """Download golfDB videos, and process the frames into a directory that can be used
    with PyTorch's ImageFolder dataset."""
    assert view_type in ("down-the-line", "face-on", "other", "all")
    data = golfDB_from_mat()
    if view_type != "all":
        data = data[data["view"] == view_type]
    logger.info(
        f"Number of unique YouTube videos: {len(data['youtube_id'].unique()):3d}"
    )
    logger.info(f"Number of annotations: {len(data['swing_id'].unique()):3d}")
    # youtube_ids = download_videos(return_ids=True, df=data)
    # data = data[data["youtube_id"].isin(youtube_ids)]
    for index, row in data.iterrows():
        logger.info(f"Loading {row['swing_id']} as array")
        X, y = load_array(row["swing_id"])
        for i, (x_, y_) in enumerate(zip(X, y)):
            im = Image.fromarray(x_)
            cls = np.argmax(y_)
            im.save(
                f"../../data/golfDB/PyTorchImageFolder/{cls}/{row['swing_id']}_{i}.png"
            )


if __name__ == "__main__":
    preprocess_videos("down-the-line")
