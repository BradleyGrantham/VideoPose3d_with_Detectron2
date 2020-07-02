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

import golf.io


GOLFDB_COLUMNS = ["id", "youtube_id", "player", "sex",
                  "club", "view", "slow", "events",
                  "bbox", "split"]

GOLFDB_MAT_PATH = "../../data/golfDB/golfDB.mat"

YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v={youtube_id}"


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def golfDB_from_mat() -> pd.DataFrame:
    """Majority of this code taken from https://github.com/wmcnally/golfdb/blob/master/data/generate_splits.py."""
    mdata = scipy.io.loadmat(GOLFDB_MAT_PATH)["golfDB"][0]
    mdtype = mdata.dtype

    ddata = {col: mdata[col] if col in {"bbox", "events"} else [item.item() for item in
                                                                mdata[col]] for col in mdtype.names}
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

    logger.info(f"Number of unique YouTube videos: {len(df['youtube_id'].unique()):3d}")
    logger.info(f"Number of annotations: {len(df['swing_id'].unique()):3d}")

    return df


def download_videos(n: Optional[int] = None):
    """Download videos from YouTube."""
    df = golfDB_from_mat()

    youtube_ids = df["youtube_id"]

    if n is not None:
        youtube_ids = youtube_ids[:n]

    for youtube_id in tqdm(youtube_ids):
        youtube_url = YOUTUBE_VIDEO_URL.format(youtube_id=youtube_id)
        youtube_dl_str = f"youtube-dl -f 'best[ext=mp4],fps=30' -o '../../data/golfDB/videos/%(id)s.%(ext)s' {youtube_url}"
        subprocess.check_call([youtube_dl_str], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def crop_array(a: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    assert bbox.shape[0] == 4 and bbox.ndim == 1
    x = int(a.shape[2] * bbox[0])
    y = int(a.shape[1] * bbox[1])
    w = int(a.shape[2] * bbox[2])
    h = int(a.shape[1] * bbox[3])
    return a[:, y:(y + h), x:(x + w), :]


def load_array(swing_id: str) -> Tuple[np.ndarray, np.ndarray]:
    df = golfDB_from_mat()

    swing = df.query(f"swing_id == '{swing_id}'").loc[0]
    video_id = swing["youtube_id"]

    filepath = f"../../data/golfDB/videos/{video_id}.mp4"

    X = golf.io.read_video(filepath, rot=False)
    X = np.stack(X, axis=0)

    X = crop_array(X, swing["bbox"])

    logger.info(f"Shape of video: {X.shape}")

    events = np.squeeze(df[df["youtube_id"] == video_id]["events"][0])

    y = np.zeros((X.shape[0], 9))
    y[:, -1] = 1  # set all no event class to be 1
    for i, event in enumerate(events):
        if i != 0 and i != 9:  # if not start of clip or end of clip
            y[event, i - 1] = 1
            y[event, -1] = 0

    # trim clip
    X = X[events[0]:events[-1], :, :, :]
    y = y[events[0]:events[-1], :]

    return X, y


if __name__ == "__main__":
    data = golfDB_from_mat()
    # download_videos(1)
    xx, yy = load_array("11da921e7d69821fadcfccadaa0cc52752e5cd4b")
    logger.info("hello")
