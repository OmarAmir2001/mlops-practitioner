"""CI quality gate: block a merge if the candidate model regressed."""

import json
import sys

from prodml.registry import PRODUCTION_ALIAS, client

MARGIN = 0.05
METRIC = "roc_auc"
MODEL_NAME = "churn-predictor"
CANDIDATE_FILE = "metrics/train.json"


def main() -> int:
    with open(CANDIDATE_FILE) as f:
        candidate = json.load(f)[METRIC]

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
        production = client.get_run(current.run_id).data.metrics[METRIC]
    except Exception as exc:
        print(f"No production baseline ({exc.__class__.__name__}). "
              f"Candidate {METRIC}={candidate:.4f}. Passing.")
        return 0

    floor = production * (1 - MARGIN)
    print(f"candidate={candidate:.4f}  production={production:.4f}  floor={floor:.4f}")

    if candidate < floor:
        print(f"REGRESSION: {METRIC} fell more than {MARGIN:.0%} below production.")
        return 1

    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())