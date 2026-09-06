from conftest import requires_mlflow

pytestmark = requires_mlflow

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "app_name": "prodml",
        "app_version": "0.1.0",
    }


def test_predict_success(client):
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "churn_probability" in response.json()


def test_predict_rejects_bad_input(client):
    response = client.post("/predict", json={"gender": "Matrix", "tenure": -1})
    assert response.status_code == 422


def test_predict_batch_success(client):
    payload = [
        {
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
        },
        {
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
        },
    ]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 2
    for prediction in body:
        assert "churn_probability" in prediction


def test_metadata_check(client):
    response = client.get("/metadata")
    assert response.status_code == 200

    body = response.json()
    assert body["model_name"] == "churn-predictor"
    assert body["model_uri"] == "models:/churn-predictor@production"
    assert "run_id" in body
    assert "trained_at" in body
    assert body["framework"] in {"logistic_regression", "xgboost", "pytorch"}
