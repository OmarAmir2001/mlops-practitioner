# mlops-practitioner

![CI](https://github.com/OmarAmir2001/mlops-practitioner/actions/workflows/ci.yml/badge.svg)

## Customer Churn Prediction Service

A production-style FastAPI service that serves a churn classifier pulled live from an MLflow Model Registry. Given a customer's account details, it returns a churn probability and a threshold-based decision.

Built for the MLOps Practitioner course (ITI × MLOps MENA Community):

- **Module 1** — packaging, structured logging, serialization, testing, containerization
- **Module 2** — experiment tracking, a model registry, data versioning with DVC, and CI

The defining property of the current version: **switching the served model requires zero code changes and zero rebuilds.** Move an alias in the MLflow registry, restart the container, and the API serves a different model — possibly a different framework entirely.

---

## Architecture

```
                    ┌──────────────┐
   raw CSV ────────►│     DVC      │  versioned data + a 4-stage pipeline
                    └──────┬───────┘
                           │  dvc repro
                           ▼
      ┌────────── prepare → featurize → train → evaluate ──────────┐
      │                              │                             │
      │                              ▼                             │
      │                    ┌──────────────────┐                    │
      │                    │  MLflow tracking │  runs, params,     │
      │                    │  (Postgres +     │  metrics, plots    │
      │                    │   MinIO)         │                    │
      │                    └────────┬─────────┘                    │
      │                             │  promote_if_better()         │
      │                             ▼                              │
      │                    ┌──────────────────┐                    │
      └───────────────────►│  Model Registry  │                    │
                           │  @production     │                    │
                           └────────┬─────────┘                    │
                                    │  models:/churn-predictor@production
                                    ▼
                           ┌──────────────────┐
                           │   FastAPI  API   │
                           └──────────────────┘
```

Three model families compete on every training run — logistic regression, XGBoost, and a PyTorch MLP — plus a 10-trial Optuna sweep over XGBoost. The best by `roc_auc` is wrapped, registered, and promoted only if it beats the incumbent.

---

## Quickstart

```bash
git clone git@github.com:OmarAmir2001/mlops-practitioner.git && cd mlops-practitioner
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
```

That brings up four services:

| Service | Port | Purpose |
|---|---|---|
| `api` | 8000 | The prediction service |
| `mlflow` | 5000 | Tracking server and model registry |
| `minio` | 9000 / 9001 | Artifact store (S3-compatible) + web console |
| `postgres` | — | MLflow's backend store |

The API needs a model carrying the `production` alias before it will start. On a fresh install, train one first:

```bash
uv sync --extra dev
uv run dvc repro
```

---

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

```json
{
  "churn_probability": 0.5858,
  "churn": true,
  "threshold": 0.5
}
```

---

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | 200 only if a model is loaded |
| `/metadata` | GET | Which model is being served: name, URI, run ID, training time, size, framework |
| `/predict` | POST | Single customer → probability + decision |
| `/predict/batch` | POST | List of customers → list of predictions, scored in one call |

Swagger UI at `http://localhost:8000/docs`.

`/metadata` is worth calling — it reports the *actual* model behind the alias, so you can confirm which version is live:

```json
{
  "model_name": "churn-predictor",
  "model_uri": "models:/churn-predictor@production",
  "run_id": "530d02ca80a4493f998e8d9bf98bee6f",
  "trained_at": "2026-09-04 13:55:41.992961",
  "model_size_bytes": 96851,
  "framework": "pytorch"
}
```

---

## Training

```bash
uv run dvc repro                    # the full pipeline, with caching
python -m prodml.ml_flow.train      # training only, standalone
```

Each run produces 14 MLflow runs: three baselines, an Optuna sweep parent, and ten nested trials. Every run records params, `roc_auc` / `f1` / `log_loss`, training duration, model size, a confusion matrix, feature importance (XGBoost), and tags for `framework`, `git_commit`, and `data_version`.

The champion is wrapped in a custom `mlflow.pyfunc.PythonModel` that bundles the `DictVectorizer`, the `StandardScaler`, the model, and a `metadata.json` recording which framework it is. A factory dispatches to the right loader and adapter at load time, so `predict()` contains no framework-specific branching.

### Promotion

`registry.py`'s `promote_if_better()` compares the candidate's `roc_auc` against whatever currently holds the `production` alias, and repoints the alias only on an improvement.

**Aliases, not stages.** The handbook specifies the `None → Staging → Production` stage lifecycle, but MLflow deprecated stages in 2.9. This project uses `models:/churn-predictor@production` instead — a documented deviation, on the grounds that building on an API scheduled for removal was the worse option.

---

## Data versioning

The dataset is tracked with DVC, with MinIO as the remote:

```bash
dvc pull                    # fetch the data
dvc repro                   # run the pipeline; only changed stages re-run
dvc metrics show            # current metrics
dvc metrics diff main       # what changed vs main
dvc dag                     # the stage graph
```

`dvc.yaml` declares four stages — `prepare → featurize → train → evaluate` — with source files listed as dependencies, so changing training logic re-runs training, and changing a hyperparameter re-runs only training and evaluation.

Every MLflow run carries a `data_version` tag holding the DVC hash of its input, so any run ID can be traced back to the exact bytes it learned from.

**DVC is installed as a `uv tool`, not a project dependency:**

```bash
uv tool install "dvc[s3]"
```

It is a CLI, never imported. Installing it as a dependency forces it to resolve against this project's Python 3.14, where the newest compatible release is from 2021 and crashes on a `pathlib` API removed in 3.12.

---

## Local development

```bash
uv sync --extra dev
uv run ruff check src pipelines tests
uv run black --check src pipelines tests
uv run pytest
prodml-serve
```

Tests that need a live MLflow skip automatically when `localhost:5000` is unreachable, via a one-second socket check in `conftest.py`. Locally with the stack up, all 48 tests run and coverage is ~83%; on a CI runner the reachable subset covers ~70%.

`train.py`, `sweep.py`, `wrapper.py`, and `run.py` are excluded from coverage measurement — the first two only execute during training, and `wrapper.py` only executes inside a loaded MLflow model. All are integration-tested by `dvc repro` and the API's startup model load.

---

## CI

Two workflows, deliberately separate:

**`ci.yml`** — runs on every PR and push to `main`. `lint` → `test` → `build`, on GitHub-hosted runners. Fast, disposable, no infrastructure.

**`train.yml`** — manual trigger, runs on a **self-hosted runner**. Pulls data, runs the pipeline, and applies the quality gate.

The split is not stylistic. GitHub-hosted runners cannot reach `localhost:5000`, and an ephemeral MLflow started inside CI has an empty registry — so the quality gate would find no baseline and pass unconditionally. A self-hosted runner on the development machine gives CI access to the real registry with its full history.

In production this problem does not arise, because nothing lives on localhost: MLflow runs on a server, artifacts in S3, metadata in managed Postgres. Production CI also does not train — that belongs in a separate pipeline with the time and hardware for it, gating against a shared persistent registry.

---

## Repository structure

```
mlops-practitioner/
├── dvc.yaml                       # 4-stage pipeline (root: DVC resolves paths from here)
├── dvc.lock                       # committed — records which inputs produced which outputs
├── pyproject.toml
├── .github/workflows/
│   ├── ci.yml                     # lint → test → build (GitHub-hosted)
│   └── train.yml                  # train → gate (self-hosted)
├── docker/
│   ├── Dockerfile                 # multi-stage, non-root
│   ├── mlflow.Dockerfile          # mlflow + psycopg2 + boto3
│   └── docker-compose.yml         # api, mlflow, postgres, minio, create-bucket
├── infra/                         # Terraform, Docker provider
├── pipelines/                     # thin DVC stage entry points
│   ├── run_prepare.py
│   ├── run_featurize.py
│   ├── run_train.py
│   └── run_evaluate.py
├── scripts/
│   └── check_model_quality.py     # the CI quality gate
├── metrics/                       # train.json, eval.json (committed, cache: false)
├── reports/module-2.md
├── src/prodml/
│   ├── config.py                  # pydantic-settings; apply_aws_env()
│   ├── logging_conf.py
│   ├── helpers.py                 # @timed, timer(), model_size_mb(), git + data version
│   ├── data.py                    # prepare() / split() / featurize()
│   ├── model.py                   # ChurnPredictor — loads from the registry
│   ├── registry.py                # promote_if_better(), alias-based
│   ├── data_schema.py
│   ├── run.py
│   ├── ml_flow/
│   │   ├── train.py               # three families + champion selection
│   │   ├── sweep.py               # Optuna, nested runs
│   │   ├── wrapper.py             # pyfunc wrapper, factory + adapters
│   │   ├── pipeline_schema.py     # Pydantic validation of the YAML config
│   │   └── pipeline_config.yaml   # hyperparameters and sweep ranges
│   └── api/
│       ├── main.py                # app + lifespan (model load on startup)
│       ├── health.py              # /health, /metadata
│       └── predict.py             # /predict, /predict/batch
└── tests/
    ├── conftest.py                # fixtures + the MLflow-reachability skip marker
    ├── test_data.py               # prepare / split / featurize, including a leakage check
    ├── test_model.py              # ChurnPredictor, mocked pyfunc model
    ├── test_registry.py           # both promotion branches
    ├── test_pipeline_schema.py    # config validation + the committed YAML
    ├── test_api.py                # HTTP layer (needs MLflow)
    └── test_features.py           # Customer schema ↔ dv contract (needs MLflow)
```

---

## Configuration

All settings come from `.env`, validated by `pydantic-settings`. Nothing has a default that would let a misconfiguration pass silently.

| Variable | Example |
|---|---|
| `APP_NAME` | `prodml` |
| `APP_VERSION` | `0.1.0` |
| `PORT` | `8000` |
| `MODEL_NAME` | `churn-predictor` |
| `CHURN_THRESHOLD` | `0.5` |
| `DATA_PATH` | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| `ML_CONFIG_PATH` | `src/prodml/ml_flow/pipeline_config.yaml` |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` |
| `MLFLOW_S3_ENDPOINT_URL` | `http://localhost:9000` |
| `AWS_ACCESS_KEY_ID` | `minioadmin` |
| `AWS_SECRET_ACCESS_KEY` | `minioadmin` |

**Note on hostnames:** use `localhost` from the host, and Docker service names (`mlflow`, `minio`) from inside a container. `docker-compose.yml` overrides the two URLs for the `api` service accordingly.

---

## Results

| Model | roc_auc | f1 | log_loss | train (s) | size (MB) |
|---|---|---|---|---|---|
| PyTorch MLP | **0.8582** | 0.6211 | 0.4125 | 8.22 | 0.089 |
| Logistic Regression | 0.8575 | **0.6298** | **0.4134** | **0.089** | **0.0010** |
| Swept XGBoost | 0.8546 | 0.5641 | — | 0.98 | 0.224 |
| XGBoost baseline | 0.8417 | 0.5763 | 0.4306 | 11.79 | 0.645 |

Three of the four land within 0.004 `roc_auc` of each other. That convergence suggests the ceiling here is set by the features rather than the model class — and it means promotion decisions are being made on differences smaller than run-to-run variance, a limitation documented in `reports/module-2.md`.

Logistic regression trains 92× faster than the MLP and serialises 91× smaller, for effectively the same score. On this dataset the baseline is the sensible production choice.

Full analysis: [`reports/module-2.md`](reports/module-2.md).
