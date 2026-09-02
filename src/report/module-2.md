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

## Open items

- `data_version` tag is a placeholder until DVC lands in Step 04
- Promotion margin (above)
- ONNX conversion at promotion time — the artifact layout is ready for it; the conversion isn't written
- Cross-validation settings exist in `pipeline_config.yaml` but nothing uses them. Given three models separated by less than the noise floor on a single split, CV is the right way to establish whether any difference is real.

---

## Screenshots

<!-- TODO
- MLflow experiment view, sorted by roc_auc descending, showing all 14 runs
- Registry view showing churn-predictor versions with the production alias badge
- /metadata responses before and after the alias move
- Confusion matrix and feature importance from the artifacts tab
-->
