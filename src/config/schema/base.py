"""Base configuration schemas."""

from pydantic import BaseModel


class BaseConfig(BaseModel):
    """Base configuration with common validation logic."""

    class Config:
        extra = "allow"  # Allow extra fields for flexibility with Hydra
        use_enum_values = True
        validate_assignment = True

    def __init__(self, **data):
        super().__init__(**data)
        self._post_init_validation()

    def _post_init_validation(self):
        """Additional validation after initialization."""
        pass
