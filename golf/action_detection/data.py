"""Data taken from https://github.com/wmcnally/golfdb,"""
import hashlib

import numpy as np
import pandas as pd
import scipy.io
from loguru import logger


GOLFDB_COLUMNS = ["id", "youtube_id", "player", "sex", "club", "view", "slow", "events",
                  "bbox", "split"]

GOLFDB_MAT_PATH = "../../data/golfDB/golfDB.mat"


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

    df = df.drop(["id", "split"], axis=1)
    df["id"] = df["youtube_id"].apply(sha1)

    logger.info(f"Number of unique YouTube videos: {len(df['youtube_id'].unique()):3d}")
    logger.info(f"Number of annotations: {len(df['id']):3d}")

    return df


if __name__ == "__main__":
    data = golfDB_from_mat()
