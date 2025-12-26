"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    # Navigate up from src/wp_translation/utils to project root
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file (absolute or relative to project root)

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep merge multiple configuration dictionaries.

    Later configs override earlier ones for conflicting keys.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration dictionary
    """
    result: dict[str, Any] = {}
    for config in configs:
        _deep_merge(result, config)
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge override into base dictionary."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


class BaseConfig(BaseModel):
    """Base configuration model with common functionality."""

    class Config:
        extra = "allow"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseConfig":
        """Load configuration from a YAML file."""
        config_dict = load_config(path)
        return cls(**config_dict)
