import pickle
from abc import ABC, abstractmethod
from .config import get_settings
import structlog
import xgboost as xgb

settings = get_settings()
log = structlog.get_logger(__name__)




class ModelBase(ABC):
    @abstractmethod
    def predict(self, X: dict) -> dict: ...


class ChurnPredictor(ModelBase):
    def __init__(self, threshold: float = settings.CHURN_THRESHOLD) -> None:
        self.threshold = threshold
        self.threshold = threshold

        with open(settings.MODEL_FILE, 'rb') as f_in:
            self.dv, self.model = pickle.load(f_in)

        log.info("model_loaded", path=settings.MODEL_FILE, threshold=settings.CHURN_THRESHOLD)

    def predict(self, X: dict) -> dict:
        Xt = self.dv.transform([X])
        dmatrix = xgb.DMatrix(Xt,
                               feature_names=self.dv.get_feature_names_out().tolist())
        probability = float(self.model.predict(dmatrix)[0])

        return {
            'churn_probability': round(probability, 4),
            'churn': bool(probability >= self.threshold),
            'threshold': self.threshold,
        }
    
    def predict_batch(self, X: list[dict]) -> list[dict]:
        Xt = self.dv.transform(X)
        dmatrix = xgb.DMatrix(Xt,
                               feature_names=self.dv.get_feature_names_out().tolist())
        probabilities = self.model.predict(dmatrix)
        results = [
                    {
                        'churn_probability': round(float(p), 4),
                        'churn': bool(p >= self.threshold),
                        'threshold': self.threshold,
                    }
                    for p in probabilities
                ]
        return results