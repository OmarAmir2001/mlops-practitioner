# Module 3 — Serve it three ways, then release it safely

**Project:** Churn prediction (`prodml`) — binary classification, not the handbook's ride-duration regression.

---

## Step 01 — Orchestration with Airflow

`churn_train_pipeline` runs weekly on a LocalExecutor, wrapping the same steps the DVC pipeline already ran: `extract → validate → featurize → train → evaluate → branch → [register | skip] → notify`.

### Design decisions

**XCom carries references, never data.** Every task returns a path or a run ID. `extract` returns the parquet path, `train` returns the MLflow `run_id`, `evaluate` returns a float. XCom values are serialised into Airflow's Postgres metadata database and read on every scheduler heartbeat — putting a DataFrame there would bloat the database and slow the scheduler for every DAG in the deployment, not just this one. The task reads the file itself from the path it was handed.

**`trigger_rule="none_failed_min_one_success"` on `notify`.** After a `BranchPythonOperator`, exactly one downstream path is *skipped* rather than successful. The default `all_success` treats a skipped upstream as unsatisfied, so `notify` would never fire regardless of which branch was taken. This is the single most common branching bug in Airflow.

**Retries and timeouts.** `retries=2` with a 5-minute delay at the DAG level; `execution_timeout=45min` on `train` specifically, since it is the only task that can plausibly hang.

### Airflow 2 → 3 deviations

The handbook was written against Airflow 2, which reached end of life in April 2026. This project uses **Airflow 3.3.0**, which required three departures from the handbook's instructions:

| Handbook (Airflow 2) | Airflow 3 |
|---|---|
| `airflow webserver` | `airflow api-server`, and the DAG processor is now a separate service |
| `from airflow.operators.python import ...` | `from airflow.providers.standard.operators.python import ...`; DAG objects come from `airflow.sdk` |
| `airflow users create` | Fails with `AttributeError: 'AirflowSecurityManagerV2' object has no attribute 'find_role'` — Airflow 3 defaults to `SimpleAuthManager`, which generates a password at startup rather than storing users in the database |

Six services are required rather than the handbook's implied one: `airflow-postgres`, `airflow-init`, `airflow-apiserver`, `airflow-scheduler`, `airflow-dag-processor`, `airflow-triggerer`.

### The FileSensor could not be made to work

The handbook requires a sensor. A `FileSensor` was written and the DAG parsed cleanly, but every execution failed with:

```
AirflowNotFoundException: The conn_id `fs_default` isn't defined
```

`FileSensor` resolves its path through an Airflow **Connection** rather than reading the filesystem directly. In Airflow 2 `fs_default` shipped by default; in Airflow 3 the database starts empty. Creating it explicitly (`airflow connections add fs_default --conn-type fs --conn-extra '{"path": "/"}'`) succeeded — `airflow connections list` confirms it exists — but the task worker still reported it as undefined.

The cause is architectural: **Airflow 3 task workers no longer read the metadata database directly.** They fetch connections over the Execution API served by the apiserver, configured via `AIRFLOW__CORE__EXECUTION_API_SERVER_URL`. A misconfigured or unreachable endpoint produces the same "not defined" error as a genuinely missing connection, which makes it hard to diagnose.

The sensor was removed so the remaining seven tasks could be proven. Worth noting the sensor was of limited value here anyway: the input is a single committed CSV that is always present, so it never actually waits for anything. A meaningful sensor would watch for a *new* partition, which requires date-partitioned data this project does not have.

### Backfill

Running `airflow dags backfill` over three past dates produces three runs that are **identical**. The `DATA_PATH` is a fixed file, so "the data as of 2026-08-01" does not exist — each run reads the same 7,043 rows and trains the same models on the same seed.

This is a real limitation of the design rather than a successful demonstration. Backfill is only meaningful when a run's logical date selects a different slice of data. Making it work would require date-partitioned inputs (`data/dt=2026-08-01/`) and a DAG that reads `{{ ds }}` to pick its partition.

### Division of labour: CI vs Airflow

**GitHub Actions tests code changes; Airflow runs recurring data workflows.**

CI answers "does this commit work?" It is triggered by a push, finishes in minutes, and its output is a pass/fail on a diff. Airflow answers "did this week's training complete?" It is triggered by time or data, runs indefinitely, and its output is a state machine you can inspect, retry from the failed task, and backfill.

Putting training in CI — which Module 2 attempted — produces a job that cannot be retried from a partial failure, cannot be backfilled, has no dependency graph, and is capped by the runner's time limits. Putting tests in Airflow means learning about a broken commit a day later. Module 2's quality gate was the right logic in the wrong lane; this DAG is where it belongs.

---

## Step 02 — Choosing inference patterns

Written before any serving code, deliberately.

### The three failure modes

**Cost blowout.** Scoring all 7,043 customers hourly through the web service: 169,000 predictions a day, at per-request HTTP overhead, to answer a question whose inputs change weekly at most. The same work as a nightly batch job costs orders of magnitude less, and most of those predictions are identical to the previous hour's.

**Latency SLA breach.** A support agent has the customer on the phone and needs the churn risk before deciding whether to offer a retention discount. A p99 of 3 seconds means dead air in a live conversation. The agent stops waiting for it, the feature goes unused, and it is switched off — the model was accurate and still failed.

**Over-engineering.** Standing up Kafka, Triton, and a GPU node to score 7,043 rows nightly. The dataset fits in memory, the model is a logistic regression that serialises to 1 KB, and a pandas loop finishes in four seconds. Every added component is another thing to operate, monitor, and page someone about at 3am.

### Decision table

| Scenario | Pattern | SLA that drives it | Why not the others |
|---|---|---|---|
| Agent sees churn risk mid-call | **Online** (web service) | p95 < 200 ms — a human is waiting | Batch is hours stale; streaming adds queue latency for no benefit when the request is synchronous |
| Nightly retention scoring of the full base | **Batch** | Complete before 06:00; per-row latency irrelevant | Online would cost 50–100× for identical output; streaming has no events to react to |
| Risk recomputed on account events (failed payment, support ticket, downgrade) | **Streaming** | End-to-end < 5 s from event to updated risk | Batch misses the window entirely — the customer churns before the nightly job runs; online requires something to *ask*, and nothing does |

The third row is the one that justifies streaming's complexity: the trigger is an event nobody is waiting on synchronously, but which becomes stale within minutes.

### Vocabulary

**Latency** is how long one request takes. **Throughput** is how many complete per second. They trade against each other — micro-batching deliberately delays individual requests to serve more of them per model call.

**p50 / p95 / p99** are percentiles of the latency distribution. p95 = 200 ms means 95% of requests finished within 200 ms. **The mean is close to useless here:** if p50 is 40 ms and p99 is 4 s, one request in a hundred takes 100× the median, and the mean hides it entirely. p95 is the usual SLA anchor — high enough to catch real tail pain, not so high that a single outlier dominates.

**Cold start** is the first-request penalty after a process starts or scales up: loading the model from the registry, downloading artifacts from MinIO, initialising the runtime. This project's cold start is unusually visible — `ChurnPredictor.__init__` downloads twelve artifacts before serving anything.

**GPU utilization** is the fraction of time the device is actually computing rather than waiting for data. Low utilization with queued requests means the bottleneck is elsewhere — data transfer, preprocessing, or insufficient device concurrency — and buying a bigger GPU will not help.

**Batch size** is how many inputs go through the model in one call. Larger batches amortise fixed per-call overhead across more rows, which is why 100 rows in one call is dramatically faster than 100 calls of one row.

### The 4-layer serving stack

```
┌──────────────────────────────────────────────────────────┐
│  4. DEPLOYMENT     nginx weighted upstreams,             │
│                    Docker Compose, canary + rollback     │
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
│                    → the weights, and the dv + scaler    │
└──────────────────────────────────────────────────────────┘
```

The value of the layering is independence: swapping eager sklearn for ONNX at layer 2 should not touch layer 3, and shifting canary weights at layer 4 should not touch layer 1. This project already tests that boundary — the `ChurnModelWrapper` factory means layer 1 can be logistic regression, XGBoost, PyTorch, or ONNX with no change above it.
