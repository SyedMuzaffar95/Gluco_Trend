import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "GlucoBench_benchmark_dataset.csv"
)

# Cache the parsed CSV so repeated page loads do not re-read the file.
# load_dataset() always returns a fresh copy so callers may mutate freely.
_DATASET_CACHE = {"frame": None, "key": None}


def load_dataset():
    try:
        stat = os.stat(DATASET_PATH)
        key = "{}:{}".format(stat.st_size, int(stat.st_mtime))
    except OSError:
        key = DATASET_PATH
    if _DATASET_CACHE["frame"] is None or _DATASET_CACHE["key"] != key:
        _DATASET_CACHE["frame"] = pd.read_csv(DATASET_PATH)
        _DATASET_CACHE["key"] = key
    return _DATASET_CACHE["frame"].copy()


def get_head(df):
    return df.head()


def get_info(df):
    return df.info()


def get_description(df):
    return df.describe()


def get_shape(df):
    return df.shape


def get_missing_values(df):
    return df.isnull().sum()


def get_tail(df):
    return df.tail()


def get_duplicates(df):
    return df.duplicated().sum()


def get_unique_values(df):
    return df.nunique()


def get_columns(df):
    return df.columns.tolist()
