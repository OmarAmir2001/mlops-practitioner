import mlflow
import os
from prodml.config import get_settings
from prodml.data import prepare, split, featurize
from prodml.ml_flow.pipeline_schema import load_ml_pipeline

settings = get_settings()
ml_config=load_ml_pipeline()
settings = get_settings()
os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT_URL

def main():
    mlflow.set_tracking_uri(ml_config.tracking_uri)
    mlflow.set_experiment("churn-prediction")

    df = prepare(settings.DATA_PATH)
    df_train, df_val, df_test = split(df)
    X_train, X_val, X_test, y_train, y_val, y_test, dv, scaler = featurize(df_train, df_val, df_test)

    train_logistic_regression(X_train, y_train, X_val, y_val)
    train_xgboost(X_train, y_train, X_val, y_val)
    train_mlp(X_train, y_train, X_val, y_val)
    

def train_logistic_regression(X_train, y_train, X_val, y_val):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score, log_loss

    ml_config.active_model="logistic"
    params = ml_config.models[ml_config.active_model]
    

    with mlflow.start_run(run_name="lr-baseline"):
        mlflow.log_params(params)
        mlflow.log_param("split_seed", 42)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]   # probabilities, not labels
        y_pred = (y_proba >= settings.CHURN_THRESHOLD).astype(int)

        mlflow.log_metrics({
            "roc_auc": roc_auc_score(y_val, y_proba),
            "f1": f1_score(y_val, y_pred),
            "log_loss": log_loss(y_val, y_proba),
        })
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.set_tags({"framework": "logistic_regression", "author": "omar"})

def train_xgboost(X_train, y_train, X_val, y_val):
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, f1_score, log_loss
    
    ml_config.active_model="xgboost"
    params = ml_config.models[ml_config.active_model]

    with mlflow.start_run(run_name="xgboost-baseline"):
        mlflow.log_params(params)
        mlflow.log_param("split_seed", 42)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]   # probabilities, not labels
        y_pred = (y_proba >= settings.CHURN_THRESHOLD).astype(int)

        mlflow.log_metrics({
            "roc_auc": roc_auc_score(y_val, y_proba),
            "f1": f1_score(y_val, y_pred),
            "log_loss": log_loss(y_val, y_proba),
        })
        mlflow.xgboost.log_model(model, artifact_path="model")
        mlflow.set_tags({"framework": "xgboost", "author": "omar"})

def train_mlp(X_train, y_train, X_val, y_val):
    import torch
    from torch import nn
    from torch import optim
    from sklearn.metrics import roc_auc_score, f1_score, log_loss
    
    ml_config.active_model="mlp"
    params = ml_config.models[ml_config.active_model]

    with mlflow.start_run(run_name="mlp-baseline"):
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
            input_size = params["hidden_size"]   # <-- fill this in

        layers.append(nn.Linear(input_size, 1))   # <-- fill this in

        model = nn.Sequential(*layers)
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = nn.BCEWithLogitsLoss()

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

            mlflow.log_metrics({
                "roc_auc": roc_auc_score(y_val_t.numpy(), y_proba.numpy()),
                "f1": f1_score(y_val_t.numpy(), y_pred.numpy()),
                "log_loss": log_loss(y_val_t.numpy(), y_proba.numpy()),
            })

        mlflow.pytorch.log_model(model, artifact_path="model", input_example=X_train_t[:5].numpy())
        mlflow.set_tags({"framework": "pytorch", "author": "omar"})

if __name__ == "__main__":
    main()
