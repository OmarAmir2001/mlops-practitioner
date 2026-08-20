# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    NAME: str
    VERSION: str

    MODEL_FILE: str
    CHURN_THRESHOLD: float = 0.5

    model_config = SettingsConfigDict(env_file=".env")


def get_settings() -> Settings:
    return Settings()