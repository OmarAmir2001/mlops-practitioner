# mlops-practitioner
## Customer Churn Prediction Service

A production-style FastAPI service that wraps an XGBoost churn classifier
behind a tested, containerized API. Given a customer's account details, it
returns a churn probability and a threshold-based decision. Built as Mini
Project 1 of the MLOps Practitioner course (ITI × MLOps MENA Community),
covering packaging, structured logging, serialization, testing, and
containerization end to end.

## Quickstart (3 commands)

```bash
git clone git@github.com:OmarAmir2001/mlops-practitioner.git && cd mlops-practitioner
cp .env.example .env   # fill in APP_NAME, APP_VERSION, MODEL_FILE, METADATA_FILE, PORT
docker compose -f docker/docker-compose.yml up --build
```

The API is now live at `http://localhost:8000`.

> **Already have the image?** If you don't want to clone at all, you can run
> the published image directly — just supply the required settings inline,
> since none of them ship with defaults on purpose (fail loudly if
> misconfigured, rather than silently guessing):
> ```bash
> docker run -p 8000:8000 \
>   -e APP_NAME=prodml -e APP_VERSION=0.1.0 \
>   -e MODEL_FILE=models/churn-model_xgb.pkl \
>   -e METADATA_FILE=models/model_metadata.json \
>   -e PORT=8000 \
>   omaramir2001/prodml-api:0.1.0
> ```

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female", "seniorcitizen": 0, "partner": "yes", "dependents": "no",
    "phoneservice": "no", "multiplelines": "no_phone_service", "internetservice": "dsl",
    "onlinesecurity": "no", "onlinebackup": "yes", "deviceprotection": "no",
    "techsupport": "no", "streamingtv": "no", "streamingmovies": "no",
    "contract": "month-to-month", "paperlessbilling": "yes",
    "paymentmethod": "electronic_check", "tenure": 1,
    "monthlycharges": 29.85, "totalcharges": 29.85
  }'
```

**Response:**
```json
{
  "churn_probability": 0.73,
  "churn": true,
  "threshold": 0.5
}
```

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | 200 only if the model is actually loaded in memory |
| `/metadata` | GET | Model version, training date, framework, feature names, artifact sha256 hash |
| `/predict` | POST | Single customer → churn probability + decision |
| `/predict/batch` | POST | List of customers → list of predictions, scored in one batched call |

Interactive docs (Swagger UI) available at `http://localhost:8000/docs` once running.

## Local development (without Docker)

```bash
pip install -e ".[dev]"                                          # install
ruff check src tests && black --check src tests                  # lint
pytest -v --cov=src/prodml --cov-report=term-missing             # test
prodml-serve                                                     # serve (reads PORT from .env)
```

## Repository structure

```
mlops-practitioner/
├── README.md
├── pyproject.toml
├── docker/
│   ├── Dockerfile              # multi-stage build, non-root user
│   ├── Dockerfile.single-stage # kept for the Step 7 size comparison
│   └── docker-compose.yml
├── models/
│   ├── churn-model_xgb.pkl     # DictVectorizer + XGBoost Booster
│   └── model_metadata.json     # version, trained_at, framework
├── src/prodml/
│   ├── config.py                # pydantic-settings, reads .env
│   ├── logging_conf.py          # structlog JSON configuration
│   ├── helpers.py                # Metadata class, @timed decorator, artifact hashing
│   ├── model.py                  # ChurnPredictor: predict() / predict_batch()
│   ├── data_schema.py            # Customer Pydantic schema
│   ├── run.py                    # prodml-serve entry point
│   └── api/
│       ├── main.py               # FastAPI app + lifespan (model load on startup)
│       ├── health.py             # /health, /metadata
│       └── predict.py            # /predict, /predict/batch
└── tests/
    ├── conftest.py                # fixtures: predictor (mocked), client (TestClient), real_dv
    ├── test_model.py              # ChurnPredictor logic, mocked model
    ├── test_api.py                # full HTTP request/response, real model via lifespan
    └── test_features.py           # Customer schema ↔ dv feature-name contract
```
