from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    celery_concurrency: int = 4
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"
    massive_api_key: str = ""
    massive_base_url: str = "https://api.massive.com"
    dataset_path: str = "backtest_dataset"
    postgrest_url: str = "http://localhost:3031"
    postgrest_token: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
