from abc import ABC, abstractmethod

import mlflow.pyfunc
import pandas as pd
import structlog

from .config import apply_aws_env, get_settings
from .helpers import timed

settings = get_settings()
log = structlog.get_logger(__name__)


class ModelBase(ABC):
    """Abstract base class for models."""

    @abstractmethod
    def predict(self, X: dict) -> dict: ...
    @abstractmethod
    def predict_batch(self, X: list[dict]) -> list[dict]: ...


class ChurnPredictor(ModelBase):
    def __init__(self, threshold: float = settings.CHURN_THRESHOLD) -> None:
        self.threshold = threshold
        apply_aws_env()
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self.model_uri = f"models:/{settings.MODEL_NAME}@production"
        self.model = mlflow.pyfunc.load_model(self.model_uri)
        log.info("model_loaded", uri=self.model_uri, threshold=threshold)

    def get_metadata(self) -> dict:
        meta = self.model.metadata
        return {
            "model_name": settings.MODEL_NAME,
            "model_uri": self.model_uri,
            "run_id": meta.run_id,
            "trained_at": str(meta.utc_time_created),
            "model_size_bytes": meta.model_size_bytes,
            "framework": self.model.unwrap_python_model().framework,
        }

    @timed
    def predict(self, X: dict) -> dict:
        """Single prediction."""
        df = pd.DataFrame([X])
        probability = float(self.model.predict(df)[0])
        return {
            "churn_probability": round(probability, 4),
            "churn": bool(probability >= self.threshold),
            "threshold": self.threshold,
        }

    @timed
    def predict_batch(self, X: list[dict]) -> list[dict]:
        """Batch prediction."""
        df = pd.DataFrame(X)
        probabilities = self.model.predict(df)
        return [
            {
                "churn_probability": round(float(p), 4),
                "churn": bool(p >= self.threshold),
                "threshold": self.threshold,
            }
            for p in probabilities
        ]
