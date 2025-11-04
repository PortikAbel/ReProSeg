"""Base configuration schemas."""

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base configuration with common validation logic."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._post_init_setup()

    def _post_init_setup(self):
        """Additional setup and processing after initialization."""
        pass
