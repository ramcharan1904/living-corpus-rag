from pathlib import Path

from config import Settings


def test_settings_load_with_defaults():
    settings = Settings(_env_file=None)

    assert settings.postgres_dsn.startswith("postgresql://")
    assert settings.weaviate_url.startswith("http")

    corpus_paths = settings.corpus_paths
    assert set(corpus_paths) == {"early", "mid", "late"}
    assert all(isinstance(p, Path) for p in corpus_paths.values())
