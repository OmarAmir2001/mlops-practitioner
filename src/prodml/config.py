# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

""" Configuration settings for the application. """

class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str
    PORT: int = 8000
    MODEL_FILE: str
    METADATA_FILE: str
    CHURN_THRESHOLD: float = 0.5

    model_config = SettingsConfigDict(env_file=".env")


def get_settings() -> Settings:
    return Settings()