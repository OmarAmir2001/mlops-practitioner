import pickle
from .config import get_settings
import structlog
import hashlib
import json
import time
from functools import wraps

settings = get_settings()
log = structlog.get_logger(__name__)

class Metadata():


    def __init__(self, metadata_file: str = settings.METADATA_FILE) -> None:

        if metadata_file is None:
            log.error("metadata_not_found", reason="metadata_file_not_found")
            return
        with open(metadata_file, 'r') as f_in:
            metadata = json.load(f_in)

        self.version = metadata["version"]
        self.trained_at = metadata["trained_at"]
        self.framework = metadata["framework"]
        self.artifact_hash = self.compute_file_hash()
        log.info("metadata_loaded", path=metadata_file, version=self.version, artifact_hash=self.artifact_hash)

    def get_model_version(self) -> str:
        """Return the version of the model."""
        return self.version
    
    def get_trained_at(self) -> str:
        """Return the date and time when the model was trained."""
        return self.trained_at
    
    def get_framework(self) -> str:
        """Return the framework used to train the model."""
        return self.framework

    def compute_file_hash(self, file_path: str =settings.MODEL_FILE) -> str:
        """Compute the SHA256 hash of a file."""
        with open(file_path, 'rb') as f_in:
            file_bytes = f_in.read()
        return hashlib.sha256(file_bytes).hexdigest()


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info("timed_call", function=func.__name__, elapsed_ms=round(elapsed_ms, 2))
        return result
    return wrapper
    
        