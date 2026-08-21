import pytest
from fastapi.testclient import TestClient
from prodml.api.main import app
from unittest.mock import MagicMock
from prodml.model import ChurnPredictor
import pickle
from prodml.config import get_settings


@pytest.fixture
def predictor(monkeypatch):
    """A ChurnPredictor with a fake dv and model — no real pickle involved."""
    monkeypatch.setattr("prodml.model.xgb.DMatrix", MagicMock())
    p=ChurnPredictor.__new__(ChurnPredictor)
    p.threshold = 0.5
    p.dv = MagicMock()
    p.dv.get_feature_names_out.return_value.tolist.return_value = ["gender=female", "tenure", "monthlycharges"]
    p.model = MagicMock()
    return p

@pytest.fixture
def client():
    """A TestClient for the app."""
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="session")
def real_dv():
    """Load the real fitted DictVectorizer once, for the whole test session."""
    settings = get_settings()
    with open(settings.MODEL_FILE, 'rb') as f_in:
        dv, _ = pickle.load(f_in)
    return dv