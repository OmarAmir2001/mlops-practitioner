# Module 3 — Serve it three ways, then release it safely

**Project:** Churn prediction (`prodml`) — binary classification, not the handbook's ride-duration regression.

Metric translation throughout: RMSE/MAE/R² → **`roc_auc`, `f1`, `log_loss`**.

---

## Step 01 — Orchestration with Airflow

`churn_train_pipeline` runs weekly on a LocalExecutor, wrapping the same steps the DVC pipeline already ran:

```
extract → validate → featurize → train → evaluate → branch → [register | skip] → notify
```

### Design decisions

**XCom carries references, never data.** Every task returns a path or an ID — `extract` returns the parquet path, `train` returns the MLflow `run_id`, `evaluate` returns a float. XCom values are serialised into Airflow's Postgres metadata database and touched on every scheduler heartbeat; putting a DataFrame there would bloat the database and slow the scheduler for every DAG in the deployment, not just this one. Each task reads its input file from the path it was handed.

**`trigger_rule="none_failed_min_one_success"` on `notify`.** After a `BranchPythonOperator`, exactly one downstream path is *skipped* rather than successful. The default `all_success` treats a skipped upstream as unsatisfied, so `notify` would never fire regardless of which branch ran. This is the most common branching bug in Airflow and it fails silently — the task simply never appears.

**Retries and timeouts.** `retries=2` with a 5-minute delay at DAG level; `execution_timeout=45min` on `train` specifically, since it is the only task that can plausibly hang.

### Airflow 2 → 3 deviations

The handbook was written against Airflow 2, which reached end of life in April 2026. This project uses **Airflow 3.3.0**, which forced three departures:

| Handbook (Airflow 2) | Airflow 3 |
|---|---|
| `airflow webserver`, one service | `airflow api-server`, and the DAG processor is now separate — six services total |
| `from airflow.operators.python import ...` | `from airflow.providers.standard.operators.python import ...`; DAG objects come from `airflow.sdk` |
| `airflow users create` | Fails with `AttributeError: 'AirflowSecurityManagerV2' object has no attribute 'find_role'`. Airflow 3 defaults to `SimpleAuthManager`, which generates a password at startup rather than storing users in the database |

Services required: `airflow-postgres`, `airflow-init`, `airflow-apiserver`, `airflow-scheduler`, `airflow-dag-processor`, `airflow-triggerer`.

Two setup problems worth recording, both cost real time:

- **`airflow-init` run as root fails** with `ModuleNotFoundError: No module named 'airflow'`. Airflow is installed under `/home/airflow/.local/`, on the `airflow` user's path. The init container must run as UID 50000 like the others.
- **Mounted log directories must be owned by UID 50000.** The dag-processor crash-loops with `FileNotFoundError` on its own log file otherwise — a permissions failure disguised as a missing file.

### The FileSensor could not be made to work

The handbook requires a sensor. A `FileSensor` was written and the DAG parsed cleanly, but every execution failed:

```
AirflowNotFoundException: The conn_id `fs_default` isn't defined
```

`FileSensor` resolves its path through an Airflow **Connection** rather than reading the filesystem directly. In Airflow 2, `fs_default` shipped by default; in Airflow 3 the database starts empty. Creating it explicitly succeeded — `airflow connections list` confirms `fs_default | fs` — but the task worker still reported it as undefined.

The cause is architectural. **Airflow 3 task workers no longer read the metadata database directly**; they fetch connections over an Execution API served by the apiserver, configured through `AIRFLOW__CORE__EXECUTION_API_SERVER_URL`. A misconfigured or unreachable endpoint produces the identical "not defined" error as a genuinely missing connection, which makes it hard to diagnose from the message alone.

The sensor was removed so the remaining seven tasks could be proven. Worth noting it was of limited value regardless: the input is a single committed CSV that is always present, so it never waits for anything. A meaningful sensor would watch for a *new* partition, which requires date-partitioned data this project does not have.

### Backfill

Running `airflow dags backfill` over three past dates produces three **identical** runs. `DATA_PATH` is a fixed file, so "the data as of 2026-08-01" does not exist — each run reads the same 7,043 rows and trains the same models on the same seed.

This is a limitation of the design rather than a successful demonstration. Backfill is only meaningful when a run's logical date selects a different slice of data. Making it work would require date-partitioned inputs (`data/dt=2026-08-01/`) and a DAG that reads `{{ ds }}` to choose its partition.

### Division of labour: CI vs Airflow

**GitHub Actions tests code changes; Airflow runs recurring data workflows.**

CI answers "does this commit work?" — triggered by a push, finishes in minutes, output is pass/fail on a diff. Airflow answers "did this week's training complete?" — triggered by time or data, runs indefinitely, output is a state machine you can inspect, retry from the failed task, and backfill.

Putting training in CI, which Module 2 attempted, produces a job that cannot retry from a partial failure, cannot be backfilled, has no dependency graph, and is capped by the runner's time limit. Putting tests in Airflow means learning about a broken commit a day later. Module 2's quality gate was the right logic in the wrong lane; this DAG is where it belongs.

---

## Step 02 — Choosing inference patterns

Written before any serving code, deliberately.

### The three failure modes

**Cost blowout.** Scoring all 7,043 customers hourly through the web service: 169,000 predictions a day, each paying full per-request HTTP overhead, to answer a question whose inputs change weekly at most. Most of those predictions are identical to the previous hour's.

**Latency SLA breach.** A support agent has the customer on the phone and needs the churn risk before deciding whether to offer a retention discount. A p99 of 3 seconds is dead air in a live conversation. The agent stops waiting, the feature goes unused, and it is switched off — the model was accurate and still failed.

**Over-engineering.** Standing up Kafka, Triton and a GPU node to score 7,043 rows nightly. The dataset fits in memory, the model serialises to 1 KB, and a pandas loop finishes in four seconds. Every added component is another thing to operate, monitor, and be paged about.

### Decision table

| Scenario | Pattern | SLA that drives it | Why not the others |
|---|---|---|---|
| Agent sees churn risk mid-call | **Online** (web service) | p95 < 200 ms — a human is waiting | Batch is hours stale; streaming adds queue latency for a synchronous request |
| Nightly retention scoring of the full base | **Batch** | Complete before 06:00; per-row latency irrelevant | Online costs 50–100× for identical output; streaming has no events to react to |
| Risk recomputed on account events (failed payment, support ticket, downgrade) | **Streaming** | End-to-end < 5 s from event to updated risk | Batch misses the window — the customer churns before the nightly job runs; online needs something to *ask*, and nothing does |

The third row is what justifies streaming's complexity: the trigger is an event nobody waits on synchronously, but which goes stale within minutes.

### Vocabulary

**Latency** is how long one request takes. **Throughput** is how many complete per second. They trade against each other — micro-batching deliberately delays individual requests to serve more per model call.

**p50 / p95 / p99** are percentiles of the latency distribution. p95 = 200 ms means 95% of requests finished within 200 ms. The mean is close to useless here: with p50 at 40 ms and p99 at 4 s, one request in a hundred takes 100× the median and the mean hides it entirely. p95 is the usual SLA anchor — high enough to catch real tail pain, low enough not to be dominated by single outliers.

**Cold start** is the first-request penalty after a process starts: loading the model, downloading artifacts, initialising the runtime. This project's is unusually visible — `ChurnPredictor.__init__` downloads twelve artifacts from MinIO before serving anything.

**GPU utilization** is the fraction of time the device computes rather than waits. Low utilization with queued requests means the bottleneck is elsewhere — data transfer, preprocessing, or device concurrency — and a bigger GPU will not help.

**Batch size** is how many inputs go through the model per call. Larger batches amortise fixed per-call overhead across more rows, which is why 100 rows in one call is dramatically faster than 100 calls of one row — measured below at roughly 70×.

### The 4-layer serving stack

```
┌──────────────────────────────────────────────────────────┐
│  4. DEPLOYMENT     nginx weighted upstreams, Compose,    │
│                    canary + automatic rollback           │
│                    → routing, scaling, release strategy  │
├──────────────────────────────────────────────────────────┤
│  3. SERVING        BentoML Runner + adaptive batching    │
│                    (currently: FastAPI + uvicorn)        │
│                    → request lifecycle, concurrency      │
├──────────────────────────────────────────────────────────┤
│  2. RUNTIME        ONNX Runtime, OpenVINO                │
│                    (currently: eager sklearn / torch)    │
│                    → graph optimisation, precision       │
├──────────────────────────────────────────────────────────┤
│  1. MODEL          churn-predictor @production           │
│                    via mlflow.pyfunc + ChurnModelWrapper │
│                    → the weights, plus dv and scaler     │
└──────────────────────────────────────────────────────────┘
```

The value of the layering is independence: swapping eager sklearn for ONNX at layer 2 should not touch layer 3, and shifting canary weights at layer 4 should not touch layer 1. This project already tests that boundary — the `ChurnModelWrapper` factory means layer 1 can be logistic regression, XGBoost, PyTorch or ONNX with no change above it.

---

## Step 03 — CAT 1: the FastAPI baseline and its four problems

The Module 1 API is already CAT 1: FastAPI + uvicorn, eager runtime, one request at a time. It was deployed and measured rather than assumed.

### Baseline load test

Locust, 50 concurrent users, 80% single predictions / 15% batch / 5% metadata, `between(1, 3)` wait times, randomised payloads. 8,517 requests, **zero failures**.

| Endpoint | Requests | p50 | p95 | p99 | Mean | Max | RPS |
|---|---|---|---|---|---|---|---|
| `POST /predict` | 6,808 | 7 ms | 26 ms | 46 ms | 10.83 ms | 150 ms | 40.9 |
| `POST /predict/batch` | 1,311 | 9 ms | 28 ms | 52 ms | 12.25 ms | 133 ms | 8.0 |
| `GET /metadata` | 398 | 4 ms | 20 ms | 39 ms | 7.68 ms | 77 ms | 1.2 |
| **Aggregated** | **8,517** | **8 ms** | **26 ms** | **47 ms** | **10.9 ms** | **150 ms** | **50.1** |

Three things worth reading out of this table.

**The service is comfortably inside its SLA and nowhere near saturation.** p95 of 26 ms against the 200 ms target from Step 02, with no failures. 50 users at 1–3 second waits is only ~50 rps, well below the knee.

**`/predict/batch` costs almost nothing extra per request.** 9 ms versus 7 ms at p50, while carrying 5–50 customers instead of one and returning 28× more data (1,636 vs 58 bytes average). Scoring 50 rows costs roughly 2 ms more than scoring one — which is the batch-size argument from Problem 1 restated from the server's side.

**The tail is 6× the median.** p50 of 8 ms against p99 of 47 ms, with a 150 ms maximum. Not alarming at this load, but it is the shape that gets worse under contention, and it is what Step 08's ramp to 100 users should expose.

The payload generator randomises every field and derives `totalcharges` from `monthlycharges × tenure`, so the synthetic customers could plausibly exist. Sending the same row repeatedly would let any caching anywhere in the stack invalidate the measurement.

### Problem 1 — batch size is always 1

Measured directly, 100 predictions each way:

| Approach | Wall clock | Per prediction |
|---|---|---|
| 100 single requests to `/predict` | **1.340 s** | 13.4 ms |
| One batch of 100 to `/predict/batch` | **0.019 s** | 0.19 ms |

**Roughly 70× faster batched.**

The gap is not all server-side, and it would be dishonest to present it as such. Of the 1.340 s, `user 0.214s + sys 0.489s` is client-side CPU — curl starting 100 processes. Over half the single-request time is the client.

Discounting that, the remaining gap is still large, and it comes from four places paid 100 times instead of once: HTTP connection setup, Pydantic validation, DataFrame construction (`pd.DataFrame([X])` per request rather than one `pd.DataFrame(X)`), and the model's own fixed per-call cost.

This is precisely what adaptive micro-batching recovers for clients that *cannot* batch. The `/predict/batch` endpoint already gets the benefit; a support agent hitting `/predict` gets none of it, because their request arrives alone. Micro-batching makes the server accumulate concurrent single requests into one model call — the same win, with no client change.

### Problem 2 — the GIL blocks concurrency

Under the 50-user ramp on a **12-core** machine, the uvicorn process peaked at **82% CPU** — roughly 7% of the 1200% available.

Per-core: core 5 carried 77% of the load while the other eleven sat between 2% and 21%.

Latency climbed while eleven cores idled. That is the GIL serialising Python bytecode across concurrent requests.

The effect is unusually clean here. Normally numpy and sklearn release the GIL during their C-level work, which muddies the picture — but the per-request cost in this service is dominated by *Python-level* overhead: Pydantic validation, DataFrame construction, dict handling. A logistic regression's matrix multiply releases the GIL and takes microseconds; everything around it does not.

**The model is not the bottleneck. The Python around it is.** That is what a BentoML Runner addresses — it moves inference into a separate worker process, so the API's interpreter and the model's execution stop contending.

### Problem 3 — eager runtime

The model was converted to ONNX (deferred from Module 2 Step 03, where the wrapper's three-artifact layout was designed for exactly this) and served through ONNX Runtime with `ORT_ENABLE_ALL` graph optimisation, `CPUExecutionProvider`.

| Runtime | roc_auc | Throughput |
|---|---|---|
| Eager sklearn | 0.859365 | 25,218 rows/sec |
| ONNX Runtime | 0.859365 | **16,665 rows/sec** |

**Accuracy is identical to six decimal places** — the conversion preserved the model exactly, which is the result you want from the accuracy-delta check.

**Throughput is 34% worse**, and this is the more interesting finding.

Logistic regression is a single matrix multiply. There is nothing for a graph optimiser to fuse — no chains of operations to collapse, no redundant nodes to eliminate. What the measurement captures instead is ONNX Runtime's per-call overhead: marshalling the numpy array into the session's input format, crossing the C++ boundary, and unpacking the output list. Against sklearn's `predict_proba`, a thin wrapper over a BLAS call on data already in memory, that overhead dominates.

Accelerated runtimes pay off on deep networks with many fusable operations. On a one-node graph they are pure overhead. **This is a case where the tool the module teaches is the wrong choice for the workload**, which is itself the lesson: measure before adopting.


### Problem 4 — no model versioning in the artifact

CAT 1 bakes `model.pkl` into the container image. The consequences:

- **Rollback requires a rebuild.** Discovering a bad model in production means rebuilding the image, pushing it, and redeploying — minutes at best, during which the bad model keeps serving.
- **The image and the model are one unit.** You cannot update the model without a new image, or the code without re-shipping the model.
- **There is no record of which model is running.** The image tag identifies a commit, not a model version, so "which model produced this prediction?" is unanswerable from the container alone.

This project fixed the problem in Module 2 Step 03: `ChurnPredictor` loads `models:/churn-predictor@production` from the MLflow registry at startup. Moving the alias and restarting swaps the served model with zero code changes and zero rebuild — demonstrated there by switching a logistic regression for a PyTorch MLP via a click in the MLflow UI.

The tradeoff is that the API now depends on a reachable registry at startup and will not boot without one. That is the correct behaviour — failing loudly beats serving a stale model — but it is a real operational dependency the CAT 1 design did not have.

### An unrelated bug this step surfaced

The containerised API would not start:

```
ImportError: Failed to initialize: Bad git executable.
```

`helpers.py` had `from git import Repo` at module level for `get_git_commit_hash()`, a training-time provenance function. GitPython refuses to import at all without a `git` binary on PATH, and the slim serving image has no reason to include one. Since `model.py` imports `helpers` for the `@timed` decorator and `main.py` imports `model`, the entire app died at import.

Same shape as the `torch` problem that inflated the Airflow image to 12 GB: **a module-level import forcing a dependency the code path never uses.** Moving the import inside the function fixes both the container and, applied to `wrapper.py`'s `torch` and `xgboost` imports, most of the image size.

---

## Baseline comparison table

To be completed as each category is measured.

| Category | Runtime | p50 | p95 | p99 | RPS | Failure % |
|---|---|---|---|---|---|---|
| CAT 1 | FastAPI eager | 8 ms | 26 ms | 47 ms | 50.1 | 0 |
| CAT 2 | BentoML + micro-batching | | | | | |
| CAT 4 | ONNX Runtime | | | | | |
| CAT 4 | OpenVINO | | | | | |
| CAT 3 | TensorRT + Triton | | | | | |

CAT 1 measured at 50 users; later rows should be measured at 100 per the Step 08 requirement, so the CAT 1 row will need re-running at that level before the comparison is apples-to-apples.

---

## Carried-over items

| Item | Status |
|---|---|
| ONNX conversion (deferred from MP2 Step 03) | **Done** — Step 03 above |
| Terraform (MP2 Step 06) | Skipped; noted as an incomplete deliverable |
| Module-level `torch` / `xgboost` imports in `wrapper.py` | Open — same fix as the git import above |
| Docker image size (12 GB, mostly CUDA-bundled torch) | Open — CPU-only wheel plus lazy imports |

---

