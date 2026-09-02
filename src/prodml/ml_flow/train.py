import mlflow
import os
from prodml.config import get_settings, apply_aws_env
from prodml.data import prepare, split, featurize
from prodml.ml_flow.pipeline_schema import load_ml_pipeline
from .ChurnModelWrapper import log_wrapped_model
import structlog
from .sweep import sweep_xgboost
from ..helpers import get_git_commit_hash as get_git_commit
from ..helpers import timer
from ..helpers import model_size_mb
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import plot_importance
from prodml.registry import promote_if_better

log = structlog.get_logger(__name__)

settings = get_settings()
ml_config=load_ml_pipeline()
settings = get_settings()
apply_aws_env()

def main():
    mlflow.set_tracking_uri(ml_config.tracking_uri)
    mlflow.set_experiment("churn-prediction")

    df = prepare(settings.DATA_PATH)
    df_train, df_val, df_test = split(df)
    X_train, X_val, X_test, y_train, y_val, y_test, dv, scaler = featurize(df_train, df_val, df_test)

    results = [
        train_logistic_regression(X_train, y_train, X_val, y_val),
        train_xgboost(X_train, y_train, X_val, y_val),
        train_mlp(X_train, y_train, X_val, y_val),
        sweep_xgboost(ml_config.sweep, X_train, y_train, X_val, y_val, settings.CHURN_THRESHOLD),
    ]

    champion = max(results, key=lambda r: r[2])
    model, framework, roc_auc,f1, run_id = champion
    log.info("champion_selected", framework=framework, roc_auc=roc_auc, f1=f1, run_id=run_id)

    with mlflow.start_run(run_id=run_id):        # reopen the champion's run
        log_wrapped_model(model, framework, dv, scaler)
    promoted = promote_if_better(run_id)
    log.info("promotion_result", promoted=promoted, framework=framework, roc_auc=roc_auc)


def train_logistic_regression(X_train, y_train, X_val, y_val):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score, log_loss

    ml_config.active_model="logistic"
    params = ml_config.models[ml_config.active_model]
    

    with mlflow.start_run(run_name="lr-baseline") as run:
        mlflow.log_params(params)
        mlflow.log_param("split_seed", 42)

        model = LogisticRegression(**params)
        with timer() as t:
            model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]   # probabilities, not labels
        y_pred = (y_proba >= settings.CHURN_THRESHOLD).astype(int)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        roc_auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)
        
        mlflow.log_metrics({
            "roc_auc":roc_auc,
            "f1": f1,
            "log_loss": log_loss(y_val, y_proba),
            "train_duration_sec": t["elapsed"],
            "model_size_mb": model_size_mb(model),
        })
        
        mlflow.sklearn.log_model(model, name="model")
        mlflow.set_tags({"framework": "logistic_regression", "author": "omar","git_commit": get_git_commit()})
        return model , "logistic_regression",roc_auc , f1 , run.info.run_id

def train_xgboost(X_train, y_train, X_val, y_val):
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, f1_score, log_loss
    
    ml_config.active_model="xgboost"
    params = ml_config.models[ml_config.active_model]

    with mlflow.start_run(run_name="xgboost-baseline") as run:
        mlflow.log_params(params)
        mlflow.log_param("split_seed", 42)

        model = XGBClassifier(**params)
        mlflow.xgboost.autolog()
        with timer() as t:
            model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= settings.CHURN_THRESHOLD).astype(int)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_importance(model, ax=ax, max_num_features=20)
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

        roc_auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)

        mlflow.log_metrics({
            "roc_auc":roc_auc,
            "f1": f1,
            "log_loss": log_loss(y_val, y_proba),
            "train_duration_sec": t["elapsed"],
            "model_size_mb": model_size_mb(model),
        })
        mlflow.xgboost.log_model(model, name="model")
        mlflow.set_tags({"framework": "xgboost", "author": "omar","git_commit": get_git_commit()})
        return model , "xgboost",roc_auc , f1 , run.info.run_id


def train_mlp(X_train, y_train, X_val, y_val):
    import torch
    from torch import nn
    from torch import optim
    from sklearn.metrics import roc_auc_score, f1_score, log_loss
    
    ml_config.active_model="mlp"
    params = ml_config.models[ml_config.active_model]

    with mlflow.start_run(run_name="mlp-baseline") as run:
        mlflow.log_params(params)
        mlflow.log_param("split_seed", 42)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

        layers = []
        input_size = X_train.shape[1]

        for _ in range(params["n_layers"]):
            layers.append(nn.Linear(input_size, params["hidden_size"]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(params["dropout"]))
            input_size = params["hidden_size"]   

        layers.append(nn.Linear(input_size, 1))

        model = nn.Sequential(*layers)
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = nn.BCEWithLogitsLoss()

        with timer() as t:
            for epoch in range(params["epochs"]):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_train_t)
                loss = criterion(y_pred, y_train_t)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    y_logits = model(X_val_t)
                    y_proba = torch.sigmoid(y_logits)
                    y_pred = (y_proba >= settings.CHURN_THRESHOLD).to(torch.int)
                    roc_auc = roc_auc_score(y_val_t.numpy(), y_proba.numpy())
                    f1 = f1_score(y_val_t.numpy(), y_pred.numpy())

                mlflow.log_metrics({
                    "roc_auc": roc_auc,
                    "f1": f1,
                    "log_loss": log_loss(y_val_t.numpy(), y_proba.numpy()),
                }, step=epoch)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_val_t.numpy(), y_pred.numpy(), ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        mlflow.log_metrics({
            "train_duration_sec": t["elapsed"],
            "model_size_mb": model_size_mb(model),
        })

        mlflow.pytorch.log_model(model, name="model", input_example=X_train_t[:5].numpy())
        mlflow.set_tags({"framework": "pytorch", "author": "omar","git_commit": get_git_commit()})
        return model , "pytorch",roc_auc , f1 , run.info.run_id

if __name__ == "__main__":
    main()
