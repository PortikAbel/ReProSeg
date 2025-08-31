import tempfile
from pathlib import Path
from typing import Generator

import pytest


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
