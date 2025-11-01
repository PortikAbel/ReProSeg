import tempfile
from pathlib import Path
from typing import Generator

import pytest

from config import ReProSegConfig


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(autouse=True)
def mock_env_variables(temp_dir: Path, monkeypatch):
    """Mock environment variables for testing."""
    # Set environment variables directly
    monkeypatch.setenv("DATA_ROOT", str(temp_dir))
    monkeypatch.setenv("LOG_ROOT", str(temp_dir))
    monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))
    yield


@pytest.fixture
def mock_config() -> ReProSegConfig:
    """Create a mock ReProSegConfig for testing with reasonable defaults."""
    return ReProSegConfig()
