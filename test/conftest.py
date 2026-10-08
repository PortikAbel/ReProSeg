import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest

from config import TrainConfig


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


@pytest.fixture(autouse=True)
def mock_run_context(temp_dir: Path, monkeypatch):
    """Provide a stubbed run context so tests never touch real run artifacts."""
    import utils.run_context as run_context

    stub = MagicMock(spec=run_context.RunContext)
    stub.log_dir = temp_dir
    stub.checkpoint_dir = temp_dir / "checkpoints"
    stub.tensorboard_dir = temp_dir / "tensorboard"
    stub.hparams_dir = temp_dir / "tensorboard" / "hparams"
    stub.prototypes_dir = temp_dir / "prototypes"
    stub.consistency_dir.side_effect = (
        lambda official_parts_only=False: temp_dir / f"consistency{'_official_parts' if official_parts_only else ''}"
    )
    monkeypatch.setattr(run_context, "_run_context", stub)
    yield stub


@pytest.fixture
def mock_config() -> TrainConfig:
    """Create a mock TrainConfig for testing with reasonable defaults."""
    cfg = TrainConfig()
    cfg.data.filter_classes = True
    return cfg


@pytest.fixture
def data_config(mock_config):
    """Expose only the DataConfig portion of mock_config."""
    return mock_config.data
