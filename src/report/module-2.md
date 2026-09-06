# Module 2 — Tracking, Versioning, Automation

**Project:** `prodml` — Telco customer churn prediction (binary classification)
**Task deviation:** the handbook's worked example is a regression task (ride duration, RMSE/MAE/R²). This project is a classifier, so all metrics are adapted: **`roc_auc`, `f1`, `log_loss`** replace RMSE/MAE/R² throughout.

---

## Step 01 — Tracking infrastructure

MLflow tracking server backed by **PostgreSQL** (metadata) and **MinIO** (artifacts), all declared in `docker/docker-compose.yml`.

Two things worth recording:

- **Bucket creation is not automatic.** MinIO does not create `mlflow-artifacts` on demand, so a one-shot `create-bucket` service runs `mc mb -p` before MLflow starts. The `-p` flag (`--ignore-existing`) makes it idempotent across restarts.
- **Startup ordering matters.** Plain `depends_on` only waits for a container to *start*, not to be *ready*. MLflow was crash-looping against a Postgres that hadn't finished initialising. Fixed with `condition: service_healthy` (Postgres) and `condition: service_completed_successfully` (bucket job).

---

## Step 02 — Tracking three model families

**14 runs total:** 3 baselines + 1 sweep parent + 10 nested Optuna trials.

All families train on identical splits produced by a single `featurize()` call, so the comparison is valid — `DictVectorizer` and `StandardScaler` are fit once on train and only ever `.transform()`-ed onto val/test.

### Results

| Model | roc_auc | f1 | log_loss | train (s) | size (MB) |
|---|---|---|---|---|---|
| **PyTorch MLP** | **0.8582** | 0.6211 | 0.4125 | 8.22 | 0.089 |
| **Logistic Regression** | 0.8575 | **0.6298** | **0.4134** | **0.089** | **0.0010** |
| Swept XGBoost | 0.8546 | 0.5641 | — | 0.98 | 0.224 |
| XGBoost baseline | 0.8417 | 0.5763 | 0.4306 | 11.79 | 0.645 |

### What the numbers say

**Three of the four approaches land within 0.004 `roc_auc` of each other.** That convergence is the real finding: it suggests the ceiling here is set by the features, not the model class. Further effort would be better spent on feature engineering than on architecture search.

**The best `roc_auc` and the best `f1` belong to different models.** The MLP ranks marginally better; logistic regression classifies marginally better at the 0.5 threshold. A 0.0007 `roc_auc` gap is well inside run-to-run noise on a single validation split — treating it as a real difference would be overclaiming.

**Cost matters as much as accuracy.** Logistic regression trains in 0.089s and serialises to 1 KB. The MLP takes 92× longer to train and is 91× larger; XGBoost is 132× longer and 645× larger, for the *worst* score of the three. On this dataset the baseline is the sensible production choice, and the only reason it isn't currently serving is a 0.0007 metric difference the promotion gate treats as meaningful.

### Hyperparameter sweep

10 Optuna trials over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, each logged as a nested run under an `xgb-sweep` parent.

Best params: `max_depth=4, learning_rate=0.0232, n_estimators=134, subsample=0.752` → `roc_auc 0.8546`.

Notable: the search converged on **shallow trees with a low learning rate** — `max_depth=4` against the hand-picked baseline's 6. Deep trees consistently scored worse (`max_depth=10` produced the worst trial at 0.808). Consistent with a dataset that has limited interaction structure to exploit.

The sweep improved on the hand-tuned XGBoost baseline (0.8417 → 0.8546) but still did not beat logistic regression.

**Two things the sweep found incidentally.** It is 12× faster to train (0.98s vs 11.79s) and 3× smaller (0.224 MB vs 0.645 MB) than the hand-tuned baseline, purely from the shallower trees and fewer estimators.

**And one thing it gave up.** The swept model's `f1` is *worse* than the baseline it beat — 0.5641 against 0.5763 — despite the better `roc_auc`. Optuna was told to maximise `roc_auc`, so it traded away threshold-specific performance to get it. A clean reminder that the metric you optimise is the metric you get, and it may not be the one the business cares about.

**The sweep is not reproducible run-to-run.** Optuna's sampler is seeded from fresh randomness on each execution, so repeated runs land on different "best" parameters (an earlier run selected `max_depth=3, learning_rate=0.0194, n_estimators=149`). Fixing the sampler seed would make the search deterministic; leaving it unseeded means the reported best params are one draw from a distribution rather than a stable answer.

**Autologging leaks into the sweep.** `mlflow.xgboost.autolog()` patches the library globally rather than being scoped to a `with` block, so the sweep's trials — which run after `train_xgboost` — inherit it and log 38 parameters each. Harmless here, but worth knowing: `autolog()` is process-wide until explicitly disabled.

**Search ranges live in `pipeline_config.yaml`; best params are *not* written back.** Config files are inputs and belong in git; results belong in the tracking store. A training run that rewrites its own config breaks reproducibility, can't work in a read-only CI checkout, and corrupts DVC's dependency hashing in Step 04.

### Autologging

`mlflow.xgboost.autolog()` enabled on the XGBoost baseline only, so the difference is observable.

| Run | Parameters logged |
|---|---|
| `lr-baseline` (manual only) | 4 |
| `xgboost-baseline` (manual + autolog) | **39** |

Autolog captured 34 parameters that were never written in code, including `objective: binary:logistic`, `num_boost_round`, `early_stopping_rounds`, and every regularisation default (`reg_alpha`, `reg_lambda`, `gamma`, `min_child_weight`, `colsample_*`).

Most show `None` — but recording that a parameter was *left unset* is exactly what's needed to reproduce the run later, and it's the sort of thing nobody logs by hand.

**What autolog could not do:** the `f1` at this project's specific `CHURN_THRESHOLD`, the confusion matrix, `train_duration_sec`, `model_size_mb`, and the `git_commit` tag. Autolog knows the library; it doesn't know the business decisions layered on top. It's a floor, not a replacement for deliberate logging.

### Per-run logging

- **Params** — all hyperparameters, `split_seed`
- **Metrics** — `roc_auc`, `f1`, `log_loss`, `train_duration_sec`, `model_size_mb`
- **Artifacts** — the model, confusion matrix, feature importance (XGBoost), and dependencies exported automatically from `uv.lock`
- **Tags** — `framework`, `author`, `git_commit` (via GitPython)

`data_version` is deferred to Step 04, where it will carry a real DVC hash rather than a placeholder.

### One bug worth recording

The MLP initially scored `roc_auc 0.759 / f1 0.222`. The cause was a config/code mismatch: `pipeline_config.yaml` defined both `n_layers: 2` and `hidden_size: 128`, but the network passed `n_layers` as the layer *width* — producing a 2-neuron bottleneck while `hidden_size` went unused.

Rebuilt to honour both values by constructing layers in a loop. Result: **`roc_auc 0.759 → 0.858`, `f1 0.222 → 0.621`.** The tell was the shape of the failure — respectable ranking ability alongside poor thresholded classification points at capacity starvation rather than a broken pipeline.

---

## Step 03 — Registry and promotion lifecycle

### The wrapper problem

`mlflow.xgboost.log_model()` and friends serialise only the estimator. They know nothing about the `DictVectorizer` and `StandardScaler` that raw customer data must pass through first. A registry version holding a bare estimator is not a servable model.

Solved with a custom `mlflow.pyfunc.PythonModel` that bundles `dv`, `scaler`, `model`, and a `metadata.json` as four separate artifacts.

**Separate rather than one pickled bundle**, for two reasons: XGBoost saves to its own stable JSON format instead of version-brittle pickle, and the model artifact can later be swapped for ONNX without touching the transformers.

### Framework dispatch

The champion can be any of three families, each with a different serialisation format and a different way of producing probabilities. Rather than branching inside `predict()`, the wrapper uses a **factory with thin adapters**:

- `metadata.json` records the framework; `load_context()` dispatches through a `LOADERS` dict
- Each loader returns an adapter exposing one uniform method, `predict_proba(X) -> np.ndarray`
- `SklearnAdapter` covers both LogisticRegression and XGBClassifier; `TorchAdapter` handles tensor conversion, `sigmoid`, and flattening `(N,1)` to `(N,)` so both return identical shapes

`predict()` therefore contains no framework branching at all. Adding ONNX later means adding one adapter and one loader — nothing else changes.

A `SAVERS` dict mirrors `LOADERS` and lives beside it, since the two must stay in lockstep.

### Stages vs aliases

The handbook specifies the `None → Staging → Production` stage lifecycle. **MLflow deprecated stages in 2.9**, and `transition_model_version_stage` emits a `FutureWarning` pointing at the alias migration guide.

This project uses **aliases** instead: `models:/churn-predictor@production`. The deviation is deliberate — building on an API scheduled for removal seemed worse than a documented departure from the handbook text.

Conceptually the alias model is also better: a version can hold only one stage, but can carry several aliases, so `@champion` and `@production` can diverge during a canary rollout.

### The acceptance check

Same request, no code changes, no rebuild — only the alias moved in the MLflow UI:

| | Before | After |
|---|---|---|
| `framework` | logistic_regression | **pytorch** |
| `run_id` | 27c9c055… | **feec922e…** |
| `model_size_bytes` | 3,331 | **96,817** |
| `churn_probability` | 0.5858 | **0.5886** |

The API went from serving a linear model to serving a neural network because of a click in a web UI. `ChurnPredictor` reads the alias at load time; the factory dispatched to `TorchAdapter` on its own.

The probabilities differ by only 0.003, which makes for an undramatic screenshot — but that's a property of two models that genuinely agree on this customer. The metadata difference is the unambiguous evidence.

### Promotion gate

`registry.py`'s `promote_if_better()` compares a candidate's metric against whatever currently holds the `production` alias, and only repoints it on an improvement. Wired into the end of `train.py`, replacing unconditional registration.

Observed in a real run:

```
not_promoted  candidate=0.8574799 production=0.8581547 version=11
```

**Known limitation:** the gate compares with a bare `>`, and it declined on a difference of 0.0007 — far below the noise floor. A production gate needs a margin (`candidate > production + MARGIN`). Step 05's CI quality gate asks for exactly this, and the same fix applies here.

The comparison also hardcodes "higher is better," which would silently promote the *worse* model if called with `metric="log_loss"`.

---

## Step 04 — Data versioning and pipelines with DVC

### An installation problem worth recording

`uv add "dvc[s3]"` resolved to **DVC 2.1.0** — a 2021 release — and every command crashed with `AttributeError: type object 'PosixPathInfo' has no attribute '_from_parts'`.

The cause: this project targets Python 3.14, and `_from_parts` is a private `pathlib` method removed in Python 3.12. No recent DVC declares 3.14 support, so the resolver walked backwards until it found a release with no upper bound — one old enough to predate the incompatibility.

Fixed by installing DVC as a standalone tool rather than a project dependency:

```bash
uv tool install "dvc[s3]"     # → 3.67.1
uv remove dvc
```

The reasoning generalises: **DVC is a CLI, not a library.** Nothing in this codebase ever writes `import dvc` — it's invoked from the shell and from `dvc.yaml` commands, the same way `git` and `docker` are. Forcing it to resolve against the project's Python version was the mistake; as a `uv tool` it gets its own environment on a Python it actually supports.

### Remote

The existing MinIO instance is reused, with DVC storing under a `dvc/` prefix in the same bucket MLflow uses for artifacts:

```bash
dvc remote add -d storage s3://mlflow-artifacts/dvc
dvc remote modify storage endpointurl http://localhost:9000
dvc remote modify --local storage access_key_id minioadmin
dvc remote modify --local storage secret_access_key minioadmin
```

Credentials go in `.dvc/config.local` via `--local`, which DVC gitignores. `.dvc/config` — with the URL and endpoint but no secrets — is committed.

Same `localhost:9000` versus `minio:9000` split that MLflow already required: the host-side tooling uses `localhost`, anything inside the Docker network uses the service name.

### Proving versioning works

The demonstration that matters is **content versioning at a fixed path**, not tracking two differently-named files. A first attempt tracked `churn.csv` and `churn2.csv` side by side — DVC correctly added and removed a whole file on checkout, but that proves nothing about versioning the dataset itself.

Done properly: truncate the same file to 5,000 rows, `dvc add`, commit, then travel.

```
wc -l churn.csv                       → 5001
git checkout HEAD~1 && dvc checkout   → M data/…churn.csv
wc -l churn.csv                       → 7044
git checkout mini-project-2 && dvc checkout
wc -l churn.csv                       → 5001
```

Three different line counts from one path, driven entirely by which commit is checked out. DVC reporting `M` (modified) rather than an add/delete confirms it is versioning content, not juggling files.

The full dataset was restored afterwards by checking out the older `.dvc` pointer and re-adding it, so both versions remain in history.

### Pipeline

Four stages, mapping onto functions that already existed because `data.py` was split this way in Step 02:

```
churn.csv → prepare → clean.parquet → featurize → features.pkl → train → train.json → evaluate → eval.json
```

Each stage is a thin script in `pipelines/` that imports from `prodml` and handles file I/O, since `dvc.yaml` runs shell commands rather than Python functions.

**`dvc.yaml` lives at the repository root, not in `pipelines/` as the handbook suggests.** DVC resolves every path in the file relative to *the file's own directory* — with it in `pipelines/`, `cmd: python pipelines/run_prepare.py` was resolved as `pipelines/pipelines/run_prepare.py`. Keeping it there would mean prefixing every dep, out, and command with `../`. Root placement is also DVC's own default, which `dvc repro` assumes.

### Three design decisions

**Features cross the process boundary as one pickled tuple.** `featurize()` returns eight objects — six arrays plus the fitted `dv` and `scaler`. They are meaningless apart: `X_train` without its matching `dv` is an array whose columns can't be interpreted. One file, one hash, no way for the pieces to drift out of sync.

Worth contrasting with the *opposite* decision made for `ChurnModelWrapper`, where three separate artifacts were chosen precisely so the model could be swapped for ONNX independently. The distinction is lifecycle: the wrapper's artifacts ship to production and are replaced individually; `features.pkl` is a build-time intermediate consumed once and never deployed.

**`train` declares no `outs`.** Its real output — a registered model version — lands in MLflow and MinIO, which DVC cannot track. The only trackable artifact is the metrics JSON. This is a genuine seam between the two tools, not a modelling error.

**`evaluate` takes `metrics/train.json` as a dependency.** Without it, the two stages share no file, so DVC treats them as parallel branches off `featurize` and may run `evaluate` first. The added dependency forces `train → evaluate` ordering.

**`evaluate` scores the held-out test set, which nothing else in the project touched.** Every metric in Step 02 is validation-set performance — and validation data was used to select a champion from fourteen candidates, so those numbers are optimistically biased by construction.

### Test-set results

| Metric | Validation | Test |
|---|---|---|
| `roc_auc` | 0.8583 | **0.8589** |
| `f1` | 0.6154 | 0.6071 |
| `log_loss` | ~0.4125 | **0.3997** |

The expected outcome was a drop on test, from selection bias. Two of three metrics came out marginally *better*.

The honest reading is not that selection bias is absent, but that **it is smaller than the variance between splits.** That is consistent with everything else observed in this module: four model families within 0.004 of each other, and promotion decisions turning on differences of 0.0001. The stable claim is that this model achieves roughly 0.86 `roc_auc`, and finer distinctions are not measurable on a single split.

`f1` dropping while `roc_auc` and `log_loss` improved is expected — `f1` depends on the 0.5 threshold, so it is more sensitive to the class balance of a particular split than the two threshold-independent metrics.

### Evaluate scores production, not the candidate

`dvc metrics show` reports two different models:

| File | Framework | Run |
|---|---|---|
| `metrics/train.json` | logistic_regression | `7033f80f…` |
| `metrics/eval.json` | pytorch | `530d02ca…` |

Not a bug — the promotion gate refused the newly trained logistic regression, so `evaluate` scored the PyTorch model still holding the `production` alias. `train` reports what was just built; `evaluate` reports what is actually deployed.

Defensible either way, and worth naming as a deliberate choice: "how good is the model we are actually serving" is a real question. The cost is that a rejected candidate's test score is never measured, so there is no way to check whether the gate made the right call.

### Caching and selective re-execution

`dvc repro` a second time with nothing changed: all four stages report `didn't change, skipping`.

Changing `mlp.epochs` in `pipeline_config.yaml` and re-running: `prepare` and `featurize` skip; `train` and `evaluate` execute. That file is in `train`'s dependency list and not in the earlier stages', so DVC re-runs exactly the affected subgraph.

Source files are listed as dependencies throughout — `src/prodml/data.py` under `prepare`, `train.py`/`sweep.py`/`ChurnModelWrapper.py`/`registry.py` under `train`. Without them, changing training logic would not trigger a re-run, because DVC has no way to know code changed unless told.

### Lineage closed

`get_data_version()` reads the md5 out of the tracked `.dvc` file and logs it as an MLflow tag on every run:

```python
def get_data_version(dvc_file: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc") -> str:
    with open(dvc_file) as f:
        return yaml.safe_load(f)["outs"][0]["md5"]
```

This replaces the placeholder that had been open since Step 02. Given any run ID, its `data_version` tag identifies the exact dataset; finding the commit whose `.dvc` file carries that hash and running `dvc checkout` recovers the precise bytes that produced the model.

That closed loop — run ID → data hash → exact file — is what the acceptance check means by reproducible.