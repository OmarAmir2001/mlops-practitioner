import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CATEGORICAL = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
]
NUMERICAL = ["tenure", "monthlycharges", "totalcharges"]


def prepare(raw_path: str) -> pd.DataFrame:
    """Raw CSV -> cleaned DataFrame. Maps to the DVC 'prepare' stage."""
    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    categorical_columns = list(df.dtypes[df.dtypes == "object"].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(" ", "_")

    df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce").fillna(0)
    df.churn = (df.churn == "yes").astype(int)
    return df


def split(df: pd.DataFrame, seed: int = 42):
    """Cleaned DataFrame -> train/val/test splits. Same seed = same split, every model family."""
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=seed)
    df_test, df_val = train_test_split(df_test, test_size=0.5, random_state=seed)
    return df_train, df_val, df_test


def featurize(df_train, df_val, df_test):
    """Splits -> (X_train, X_val, X_test, y_train, y_val, y_test, dv).
    Maps to the DVC 'featurize' stage. Scaler and DictVectorizer are fit on
    train only, then applied (never re-fit) to val/test."""
    from sklearn.feature_extraction import DictVectorizer

    df_train, df_val, df_test = df_train.copy(), df_val.copy(), df_test.copy()

    scaler = StandardScaler()
    df_train[NUMERICAL] = scaler.fit_transform(df_train[NUMERICAL])
    df_val[NUMERICAL] = scaler.transform(df_val[NUMERICAL])
    df_test[NUMERICAL] = scaler.transform(df_test[NUMERICAL])

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(
        df_train[CATEGORICAL + NUMERICAL].to_dict(orient="records")
    )
    X_val = dv.transform(df_val[CATEGORICAL + NUMERICAL].to_dict(orient="records"))
    X_test = dv.transform(df_test[CATEGORICAL + NUMERICAL].to_dict(orient="records"))

    y_train, y_val, y_test = (
        df_train.churn.values,
        df_val.churn.values,
        df_test.churn.values,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, dv, scaler
