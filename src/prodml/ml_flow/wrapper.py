import json, pickle
import mlflow.pyfunc
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from prodml.data import CATEGORICAL, NUMERICAL

import structlog
log = structlog.get_logger(__name__)


class SklearnAdapter:
    """Covers LogisticRegression and XGBClassifier — both expose the sklearn API."""
    def __init__(self, model):
        self.model = model
    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class TorchAdapter:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict_proba(self, X) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_proba = torch.sigmoid(self.model(X_t))
        return y_proba.numpy().ravel()


def _load_sklearn(path):
    with open(path, "rb") as f:
        return SklearnAdapter(pickle.load(f))

def _load_xgboost(path):
    model = xgb.XGBClassifier()
    model.load_model(path)
    return SklearnAdapter(model)

def _load_torch(path):
    model = torch.load(path, weights_only=False)
    return TorchAdapter(model)

LOADERS = {
    "logistic_regression": _load_sklearn,
    "xgboost": _load_xgboost,
    "pytorch": _load_torch,
}

class ChurnModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["metadata"]) as f:
            meta = json.load(f)

        self.framework = meta["framework"]       
        self.pipeline = LOADERS[self.framework](context.artifacts["model"])

        with open(context.artifacts["dv"], "rb") as f:
            self.dv = pickle.load(f)
        with open(context.artifacts["scaler"], "rb") as f:
            self.scaler = pickle.load(f)
        log.info("wrapper_loaded", framework=self.framework)

    def predict(self, context, model_input: pd.DataFrame):
        df = model_input.copy()
        df[NUMERICAL] = self.scaler.transform(df[NUMERICAL])
        X = self.dv.transform(df[CATEGORICAL + NUMERICAL].to_dict(orient="records"))
        return self.pipeline.predict_proba(X)


# SAVERS

def _save_sklearn(model, dirpath) -> str:
    path = f"{dirpath}/model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path

def _save_xgboost(model, dirpath) -> str:
    path = f"{dirpath}/model.json"
    model.save_model(path)
    return path

def _save_torch(model, dirpath) -> str:
    path=f"{dirpath}/model.pt"
    torch.save(model, path)
    return path

SAVERS = {
    "logistic_regression": _save_sklearn,
    "xgboost": _save_xgboost,
    "pytorch": _save_torch,
}

def log_wrapped_model(model, framework: str, dv, scaler):
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = SAVERS[framework](model, tmpdir)

        dv_path = f"{tmpdir}/dv.pkl"
        scaler_path = f"{tmpdir}/scaler.pkl"
        meta_path = f"{tmpdir}/metadata.json"

        # pickle dv and scaler, write metadata.json containing {"framework": framework}
        with open(dv_path, "wb") as f:
            pickle.dump(dv, f)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        with open(meta_path, "w") as f:
            json.dump({"framework": framework}, f)

        mlflow.pyfunc.log_model(
            name="wrapped_model",
            python_model=ChurnModelWrapper(),
            artifacts={
                "model": model_path,
                "dv": dv_path,
                "scaler": scaler_path,
                "metadata": meta_path,
            },
            registered_model_name="churn-predictor",
        )