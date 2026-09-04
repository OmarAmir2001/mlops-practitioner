"""DVC stage: score the production model on the held-out test set."""
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, log_loss
from prodml.model import ChurnPredictor
from prodml.config import get_settings
import structlog

log = structlog.get_logger(__name__)
settings = get_settings()

TEST_RAW = "data/processed/test_raw.parquet"
OUTPUT = "metrics/eval.json"


def main():
    df_test = pd.read_parquet(TEST_RAW)
    y_test = df_test["churn"].values

    predictor = ChurnPredictor()
    results = predictor.predict_batch(df_test.to_dict(orient="records"))
    y_proba = [r["churn_probability"] for r in results]
    y_pred = [int(r["churn"]) for r in results]

    metrics = {
        "test_roc_auc": roc_auc_score(y_test, y_proba),
        "test_f1": f1_score(y_test, y_pred),
        "test_log_loss": log_loss(y_test, y_proba),
        **predictor.get_metadata(),
    }

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("evaluate_done", **{k: v for k, v in metrics.items() if k.startswith("test_")})


if __name__ == "__main__":
    main()