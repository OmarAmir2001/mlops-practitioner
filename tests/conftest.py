import socket
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app
from prodml.model import ChurnPredictor

SAMPLE_CUSTOMER = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85,
}


@pytest.fixture
def sample_customer():
    return dict(SAMPLE_CUSTOMER)


@pytest.fixture
def predictor():
    """A ChurnPredictor whose loaded pyfunc model is a mock."""
    p = ChurnPredictor.__new__(ChurnPredictor)
    p.threshold = 0.5
    p.model_uri = "models:/churn-predictor@production"
    p.model = MagicMock()
    return p


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def real_dv():
    """The fitted DictVectorizer from the registered production model."""
    predictor = ChurnPredictor()
    return predictor.model.unwrap_python_model().dv


@pytest.fixture
def raw_csv(tmp_path):
    """A tiny raw CSV in the shape the Telco dataset arrives in."""
    path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "customerID": ["A1", "B2", "C3", "D4", "E5", "F6", "G7", "H8", "I9", "J10"],
            "gender": ["Female", "Male"] * 5,
            "SeniorCitizen": [0, 1] * 5,
            "Partner": ["Yes", "No"] * 5,
            "Dependents": ["No", "Yes"] * 5,
            "tenure": [1, 34, 2, 45, 2, 8, 22, 10, 28, 62],
            "PhoneService": ["No", "Yes"] * 5,
            "MultipleLines": ["No phone service", "No"] * 5,
            "InternetService": ["DSL", "Fiber optic"] * 5,
            "OnlineSecurity": ["No", "Yes"] * 5,
            "OnlineBackup": ["Yes", "No"] * 5,
            "DeviceProtection": ["No", "Yes"] * 5,
            "TechSupport": ["No", "Yes"] * 5,
            "StreamingTV": ["No", "Yes"] * 5,
            "StreamingMovies": ["No", "Yes"] * 5,
            "Contract": ["Month-to-month", "One year"] * 5,
            "PaperlessBilling": ["Yes", "No"] * 5,
            "PaymentMethod": ["Electronic check", "Mailed check"] * 5,
            "MonthlyCharges": [
                29.85,
                56.95,
                53.85,
                42.30,
                70.70,
                99.65,
                89.10,
                29.75,
                104.80,
                56.15,
            ],
            "TotalCharges": [
                "29.85",
                " ",
                "108.15",
                "1840.75",
                "151.65",
                "820.5",
                "1949.4",
                "301.9",
                "3046.05",
                "3487.95",
            ],
            "Churn": ["No", "Yes"] * 5,
        }
    ).to_csv(path, index=False)
    return str(path)


def _mlflow_reachable(host="localhost", port=5000, timeout=1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


requires_mlflow = pytest.mark.skipif(
    not _mlflow_reachable(),
    reason="MLflow is not reachable; skipping tests that load the registered model",
)
