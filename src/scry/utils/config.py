# Description: Configuration management using Pydantic Settings.
# Description: Loads config from environment variables for the data source and model settings.

"""Configuration management for Scry."""


from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration loaded from environment variables.

    The data source is an object store selected by ``SCRY_DATA_URI`` (the scheme
    picks the backend: file/s3/gs/az).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Object-store data source. Scheme selects the backend (file/s3/gs/az).
    data_uri: str | None = Field(
        None,
        alias="SCRY_DATA_URI",
        description="Object-store URI for the data source",
    )

    # Model configuration
    model_path: str = Field(
        "models/xdec_model.pt",
        description="Path to saved model weights",
    )
    num_clusters: int = Field(5, description="Number of operational state clusters", gt=0)
    sequence_length: int = Field(30, description="Length of input time windows", gt=0)
    numerical_hidden_dim: int = Field(64, description="GRU hidden dim for numerical branch", gt=0)
    categorical_hidden_dim: int = Field(32, description="GRU hidden dim for categorical branch", gt=0)
    latent_dim: int = Field(8, description="VAE latent embedding dimension", gt=0)
    batch_size: int = Field(64, description="Training batch size", gt=0)

    # Training configuration
    pretrain_epochs: int = Field(500, description="XVAE pretraining epochs", gt=0)
    cluster_epochs: int = Field(300, description="Clustering training epochs", gt=0)
    learning_rate: float = Field(1e-3, description="Adam learning rate", gt=0)
    beta: float = Field(1.0, description="VAE KL regularization weight", ge=0)
    lambda_cluster: float = Field(0.1, description="Fixed clustering loss weight (IDEC-style)", ge=0)
    lambda_balance: float = Field(0.0, description="Cluster balance entropy regularization weight", ge=0)
    cluster_lr: float = Field(1e-4, description="Separate learning rate for clustering phase", gt=0)
    window_step: int = Field(10, description="Step size for sliding window generation", gt=0)

    @field_validator("num_clusters", "sequence_length", "batch_size", "latent_dim", mode="before")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that integer fields are positive."""
        if isinstance(v, str):
            v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


# Singleton instance
_config: Config | None = None


def get_config() -> Config:
    """Get the singleton Config instance.

    Returns:
        Config: The configuration instance.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset the singleton config instance. Useful for testing."""
    global _config
    _config = None


# Alias for convenience
ScryConfig = Config
