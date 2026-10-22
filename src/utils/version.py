from collections.abc import Mapping
from importlib.metadata import version


def get_version() -> str:
    return version("reproseg")


def major_minor(full_version: str) -> str:
    return ".".join(full_version.split(".")[:2])


def compat_version() -> str:
    """Checkpoints are compatible when major.minor matches; patch releases never break them."""
    return major_minor(get_version())


def check_checkpoint_version(checkpoint: Mapping, path: object) -> None:
    saved = checkpoint.get("version")
    if saved is None:
        raise ValueError(
            f"Checkpoint {path} has no version (saved before versioning was introduced). "
            "Retrain, or check out the code that produced it."
        )
    current = compat_version()
    if saved != current:
        raise ValueError(
            f"Checkpoint {path} was saved by version {saved}.x but this code is {current}.x. "
            f"Retrain, or run `git checkout v{saved}.<patch>` to load it."
        )
