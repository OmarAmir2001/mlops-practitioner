from mlflow import MlflowClient
from mlflow.exceptions import RestException
from prodml.config import get_settings
import structlog

log = structlog.get_logger(__name__)
client = MlflowClient(tracking_uri=get_settings().MLFLOW_TRACKING_URI)

PRODUCTION_ALIAS = "production"


def _find_version_for_run(model_name: str, run_id: str) -> str:
    for v in client.search_model_versions(f"name='{model_name}'"):
        if v.run_id == run_id:
            return v.version
    raise ValueError(f"No registered version found for run {run_id}")


def promote_if_better(
    candidate_run_id: str,
    model_name: str = "churn-predictor",
    metric: str = "roc_auc",
) -> bool:
    candidate_metric = client.get_run(candidate_run_id).data.metrics[metric]
    new_version = _find_version_for_run(model_name, candidate_run_id)

    try:
        current = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except RestException:
        # nothing carries the production alias yet
        current = None

    if current is None:
        client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, new_version)
        log.info("promoted_first_production", version=new_version, metric=metric, value=candidate_metric)
        return True

    prod_metric = client.get_run(current.run_id).data.metrics[metric]

    if candidate_metric > prod_metric:   # higher is better — reverse for log_loss
        client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, new_version)
        log.info("promoted", version=new_version, candidate=candidate_metric, previous=prod_metric)
        return True

    log.info("not_promoted", version=new_version, candidate=candidate_metric, production=prod_metric)
    return False