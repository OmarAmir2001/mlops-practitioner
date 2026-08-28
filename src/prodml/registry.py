from mlflow import MlflowClient

client = MlflowClient()

def promote_if_better(candidate_run_id, model_name="churn-predictor", metric="roc_auc"):
    candidate_metric = client.get_run(candidate_run_id).data.metrics[metric]

    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        # nothing in production yet — promote unconditionally
        new_version = _find_version_for_run(model_name, candidate_run_id)
        client.transition_model_version_stage(model_name, new_version, "Production")
        return True

    prod_run_id = prod_versions[0].run_id
    prod_metric = client.get_run(prod_run_id).data.metrics[metric]

    if candidate_metric > prod_metric:  # higher AUC is better — reverse this for a loss metric
        new_version = _find_version_for_run(model_name, candidate_run_id)
        client.transition_model_version_stage(model_name, new_version, "Production")
        return True
    return False

def _find_version_for_run(model_name, run_id):
    for v in client.search_model_versions(f"name='{model_name}'"):
        if v.run_id == run_id:
            return v.version
    raise ValueError(f"No registered version found for run {run_id}")