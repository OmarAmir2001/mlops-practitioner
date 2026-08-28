# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

""" Configuration settings for the application. """

class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str
    PORT: int = 8000
    MODEL_FILE: str
    METADATA_FILE: str
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