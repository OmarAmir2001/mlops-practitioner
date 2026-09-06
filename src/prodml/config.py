# config.py
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

""" Configuration settings for the application. """


class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str
    PORT: int = 8000
    MODEL_NAME: str = "churn-predictor"
    DATA_PATH: str
    CHURN_THRESHOLD: float = 0.5
    MLFLOW_TRACKING_URI: str
    ML_CONFIG_PATH: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    MLFLOW_S3_ENDPOINT_URL: str

    model_config = SettingsConfigDict(env_file=".env")


def get_settings() -> Settings:
    return Settings()


def apply_aws_env() -> None:
    """Push S3/MinIO credentials into os.environ for boto3 to find."""
    s = get_settings()
    os.environ["AWS_ACCESS_KEY_ID"] = s.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = s.AWS_SECRET_ACCESS_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s.MLFLOW_S3_ENDPOINT_URL
x=1
