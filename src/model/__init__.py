from pathlib import Path

from utils.environment import get_env


def get_pretrained_models_dir() -> Path:
    """Get the pretrained models directory path."""
    return Path(get_env("PROJECT_ROOT")) / "pretrained"
