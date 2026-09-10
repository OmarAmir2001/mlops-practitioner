"""Compare eager and ONNX predictions on the test set."""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prodml.model import ChurnPredictor

TEST_RAW = "data/processed/test_raw.parquet"


def main():
    df = pd.read_parquet(TEST_RAW)
    y = df["churn"].values

    predictor = ChurnPredictor()
    rows = df.to_dict(orient="records")

    start = time.perf_counter()
    results = predictor.predict_batch(rows)
    elapsed = time.perf_counter() - start

    proba = np.array([r["churn_probability"] for r in results])

    print(f"rows      : {len(df)}")
    print(f"framework : {predictor.get_metadata()['framework']}")
    print(f"roc_auc   : {roc_auc_score(y, proba):.6f}")
    print(f"wall clock: {elapsed:.4f}s  ({len(df)/elapsed:,.0f} rows/sec)")


if __name__ == "__main__":
    main()