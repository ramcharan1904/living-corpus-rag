from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "corpus"
    postgres_password: str = "corpus"
    postgres_db: str = "living_corpus"

    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_grpc_port: int = 50051

    # Corpus worktrees: sibling checkouts of the Pydantic repo at
    # different points in its v2 history.
    corpus_path_early: Path = Path("../pyd-v2early")
    corpus_path_mid: Path = Path("../pyd-v2mid")
    corpus_path_late: Path = Path("../pyd-v2late")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def corpus_paths(self) -> dict[str, Path]:
        return {
            "early": self.corpus_path_early,
            "mid": self.corpus_path_mid,
            "late": self.corpus_path_late,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
