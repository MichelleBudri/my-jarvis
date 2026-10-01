import os

import pytest

from backend.config import Settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Keep tests away from the personal .env and the real data/ directory."""
    for key in list(os.environ):
        if key.startswith("JARVIS_"):
            monkeypatch.delenv(key)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "data"))


@pytest.fixture
def owner_env(monkeypatch):
    monkeypatch.setenv("JARVIS_OWNER_NAME", "Alice")
    monkeypatch.setenv("JARVIS_OWNER_GENDER", "feminino")
    monkeypatch.setenv("JARVIS_LATITUDE", "-25.43")
    monkeypatch.setenv("JARVIS_LONGITUDE", "-49.27")
    monkeypatch.setenv("JARVIS_TIMEZONE", "America/Sao_Paulo")
    monkeypatch.setenv("JARVIS_CITY", "Curitiba, PR, Brasil")
