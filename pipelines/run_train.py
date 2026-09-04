"""DVC stage: features -> trained models, registered champion."""
import pickle
import json
from prodml.ml_flow.train import run_training
import structlog

log = structlog.get_logger(__name__)

INPUT = "data/processed/features.pkl"
OUTPUT = "metrics/train.json"


def main():
    with open(INPUT, "rb") as f:
        bundle = pickle.load(f)

    summary = run_training(bundle)

    with open(OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("train_done", **summary)


if __name__ == "__main__":
    main()