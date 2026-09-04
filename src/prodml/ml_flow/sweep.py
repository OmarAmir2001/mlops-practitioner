import mlflow
import optuna
import structlog
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from ..helpers import get_git_commit_hash as get_git_commit
from ..helpers import timer
from ..helpers import model_size_mb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from ..helpers import get_data_version


log = structlog.get_logger(__name__)


def _objective(trial, ranges, X_train, y_train, X_val, y_val):
    params = {
        "max_depth": trial.suggest_int("max_depth", *ranges.max_depth),
        "learning_rate": trial.suggest_float("learning_rate", *ranges.learning_rate, log=True),
        "n_estimators": trial.suggest_int("n_estimators", *ranges.n_estimators),
        "subsample": trial.suggest_float("subsample", *ranges.subsample),
    }

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        mlflow.log_metric("roc_auc", roc_auc)
        return roc_auc


def sweep_xgboost(sweep_config, X_train, y_train, X_val, y_val, threshold):
    ranges = sweep_config.xgboost

    with mlflow.start_run(run_name="xgb-sweep") as run:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t: _objective(t, ranges, X_train, y_train, X_val, y_val),
            n_trials=sweep_config.n_trials,
        )

        log.info("sweep_finished", best_params=study.best_params, best_roc_auc=study.best_value)

        # retrain once with the winning params — this is the model that competes
        model = XGBClassifier(**study.best_params)
        with timer() as t:
            model.fit(X_train, y_train)

        y_proba =  model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        roc_auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_importance(model, ax=ax, max_num_features=20)
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        

        mlflow.log_params(study.best_params)
        mlflow.log_metrics({"roc_auc": roc_auc, "f1": f1})
        mlflow.log_metric("train_duration_sec", t["elapsed"])
        mlflow.log_metric("model_size_mb", model_size_mb(model))
        mlflow.xgboost.log_model(model, name="model")
        mlflow.set_tags({"framework": "xgboost", "author": "omar", "swept": "true", "git_commit": get_git_commit(),
                         "data_version": get_data_version()})

        return model, "xgboost", roc_auc, f1, run.info.run_id