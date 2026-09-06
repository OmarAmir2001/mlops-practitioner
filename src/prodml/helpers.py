import os
import pickle
import tempfile
import time
from contextlib import contextmanager
from functools import wraps

import structlog
import yaml
from git import Repo

from .config import get_settings

"""Helper functions."""

settings = get_settings()
log = structlog.get_logger(__name__)


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info("timed_call", function=func.__name__, elapsed_ms=round(elapsed_ms, 2))
        return result

    return wrapper


def get_git_commit_hash() -> str:
    """Return the current git commit hash."""
    repo = Repo(search_parent_directories=True)
    return repo.head.object.hexsha


@contextmanager
def timer():
    """Measure elapsed seconds for a block. Access via the yielded dict after the block exits."""
    start = time.perf_counter()
    holder = {}
    try:
        yield holder
    finally:
        holder["elapsed"] = time.perf_counter() - start


def model_size_mb(model) -> float:
    """Serialized size of a model in megabytes."""
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
        pickle.dump(model, tmp)
        tmp.flush()
        return os.path.getsize(tmp.name) / (1024 * 1024)


def get_data_version(
    dvc_file: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc",
) -> str:
    """Read the DVC-tracked hash of the input dataset."""
    with open(dvc_file) as f:
        return yaml.safe_load(f)["outs"][0]["md5"]
