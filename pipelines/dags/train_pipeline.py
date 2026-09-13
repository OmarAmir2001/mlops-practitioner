"""Weekly churn model training pipeline.

Orchestrates the same steps the DVC pipeline runs, but on a schedule, with
retries, branching on the promotion decision, and a sensor waiting for data.

Runs inside the Airflow container, where `prodml` is pip-installed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.python import (
    BranchPythonOperator,
    PythonOperator,
)
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.sdk import DAG

RAW_DATA = "/opt/airflow/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_PARQUET = "/opt/airflow/data/interim/clean.parquet"
FEATURES_PKL = "/opt/airflow/data/processed/features.pkl"

METRIC = "roc_auc"
PROMOTION_MARGIN = 0.005
MODEL_NAME = "churn-predictor"

DEFAULT_ARGS = {
    "owner": "omar",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


# ────────────────────────────── task callables ──────────────────────────────
# Each returns a small value. Airflow pushes it to XCom automatically.
# Return PATHS and IDs — never DataFrames. XCom is stored in Airflow's
# metadata database, and a dataset in a Postgres row is a bad idea for both
# size and performance reasons.


def extract(**context) -> str:
    """Raw CSV -> cleaned parquet. Returns the output path."""
    import pandas as pd  # noqa: F401  (imported for the side effect of failing early)

    from prodml.data import prepare

    df = prepare(RAW_DATA)
    Path(CLEAN_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, index=False)
    print(f"extract: {len(df)} rows -> {CLEAN_PARQUET}")
    return CLEAN_PARQUET


def validate(**context) -> dict:
    """Fail loudly if the cleaned data is not what training expects."""
    import pandas as pd

    path = context["ti"].xcom_pull(task_ids="extract")
    df = pd.read_parquet(path)

    problems = []
    if len(df) < 1000:
        problems.append(f"only {len(df)} rows")
    if "churn" not in df.columns:
        problems.append("target column 'churn' missing")
    if df["churn"].nunique() < 2:
        problems.append("target has fewer than two classes")
    if df.isna().any().any():
        problems.append("nulls present after cleaning")

    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))

    stats = {"rows": len(df), "columns": df.shape[1]}
    print(f"validate: OK {stats}")
    return stats


def featurize_task(**context) -> str:
    """Cleaned parquet -> pickled feature bundle. Returns the output path."""
    import pickle

    import pandas as pd

    from prodml.data import featurize, split

    df = pd.read_parquet(CLEAN_PARQUET)
    bundle = featurize(*split(df))

    Path(FEATURES_PKL).parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_PKL, "wb") as f:
        pickle.dump(bundle, f)

    print(f"featurize: X_train {bundle[0].shape} -> {FEATURES_PKL}")
    return FEATURES_PKL


def train(**context) -> str:
    """Train all families, log to MLflow, register the champion.

    Returns the champion's run_id — the reference everything downstream uses.
    """
    import pickle

    from prodml.ml_flow.train import run_training

    with open(FEATURES_PKL, "rb") as f:
        bundle = pickle.load(f)

    summary = run_training(bundle)
    print(f"train: {json.dumps(summary, indent=2)}")

    context["ti"].xcom_push(key="summary", value=summary)
    return summary["run_id"]


def evaluate(**context) -> float:
    """Read the champion's metric back from MLflow. Returns the value."""
    from prodml.registry import client

    run_id = context["ti"].xcom_pull(task_ids="train")
    value = client.get_run(run_id).data.metrics[METRIC]
    print(f"evaluate: run {run_id} scored {METRIC}={value:.6f}")
    return value


def choose_branch(**context) -> str:
    """Decide whether to promote. Returns the task_id to run next.

    This is what makes it a BranchPythonOperator: the returned string names
    the downstream task Airflow should follow. Everything else is skipped.
    """
    from mlflow.exceptions import RestException

    from prodml.registry import PRODUCTION_ALIAS, client

    candidate = context["ti"].xcom_pull(task_ids="evaluate")

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
        production = client.get_run(current.run_id).data.metrics[METRIC]
    except (RestException, Exception) as exc:  # noqa: BLE001
        print(f"No production baseline ({exc.__class__.__name__}). Promoting.")
        return "register"

    floor = production + PROMOTION_MARGIN
    print(f"branch: candidate={candidate:.6f} production={production:.6f} floor={floor:.6f}")

    if candidate > floor:
        return "register"
    return "skip_registration"


def register(**context) -> str:
    """Point the production alias at the champion's registered version."""
    from prodml.registry import PRODUCTION_ALIAS, _find_version_for_run, client

    run_id = context["ti"].xcom_pull(task_ids="train")
    version = _find_version_for_run(MODEL_NAME, run_id)
    client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version)

    print(f"register: version {version} is now @{PRODUCTION_ALIAS}")
    return version


def skip_registration(**context) -> None:
    """The candidate did not clear the margin. Recorded, not an error."""
    candidate = context["ti"].xcom_pull(task_ids="evaluate")
    print(f"skip: candidate {METRIC}={candidate:.6f} did not clear the margin")


def notify(**context) -> None:
    """Summarise the run, whichever branch was taken."""
    ti = context["ti"]
    summary = ti.xcom_pull(task_ids="train", key="summary") or {}
    version = ti.xcom_pull(task_ids="register")

    if version:
        print(f"NOTIFY: promoted version {version} — {summary}")
    else:
        print(f"NOTIFY: no promotion this run — {summary}")


# ─────────────────────────────────── DAG ────────────────────────────────────

with DAG(
    dag_id="churn_train_pipeline",
    description="Weekly churn model training, evaluation and conditional promotion",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 8, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["churn", "training", "mlflow"],
) as dag:

    wait_for_data = FileSensor(
        task_id="wait_for_data",
        filepath=RAW_DATA,
        poke_interval=30,
        timeout=600,
        mode="reschedule",  # frees the worker slot between checks
    )

    t_extract = PythonOperator(task_id="extract", python_callable=extract)

    t_validate = PythonOperator(task_id="validate", python_callable=validate)

    t_featurize = PythonOperator(task_id="featurize", python_callable=featurize_task)

    t_train = PythonOperator(
        task_id="train",
        python_callable=train,
        execution_timeout=timedelta(minutes=45),
    )

    t_evaluate = PythonOperator(task_id="evaluate", python_callable=evaluate)

    t_branch = BranchPythonOperator(task_id="branch", python_callable=choose_branch)

    t_register = PythonOperator(task_id="register", python_callable=register)

    t_skip = PythonOperator(task_id="skip_registration", python_callable=skip_registration)

    t_notify = PythonOperator(
        task_id="notify",
        python_callable=notify,
        # After a branch, one path is always SKIPPED, not successful. The
        # default trigger_rule (all_success) would skip notify too.
        trigger_rule="none_failed_min_one_success",
    )

    (
        wait_for_data
        >> t_extract
        >> t_validate
        >> t_featurize
        >> t_train
        >> t_evaluate
        >> t_branch
    )
    t_branch >> [t_register, t_skip] >> t_notify