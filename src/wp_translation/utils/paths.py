"""Path management utilities."""

from pathlib import Path
from typing import Optional

from .config_loader import get_project_root


class PathManager:
    """Manages project paths for data, models, and outputs."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize path manager.

        Args:
            base_dir: Base project directory. If None, uses project root.
        """
        self.base_dir = base_dir or get_project_root()

    @property
    def data_dir(self) -> Path:
        """Root data directory."""
        return self.base_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        """Directory for raw downloaded PO files."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Directory for processed translation pairs."""
        return self.data_dir / "processed"

    @property
    def datasets_dir(self) -> Path:
        """Directory for HuggingFace datasets."""
        return self.data_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        """Root models directory."""
        return self.base_dir / "models"

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for training checkpoints."""
        return self.models_dir / "checkpoints"

    @property
    def adapters_dir(self) -> Path:
        """Directory for LoRA adapter weights."""
        return self.models_dir / "adapters"

    @property
    def final_models_dir(self) -> Path:
        """Directory for final merged models."""
        return self.models_dir / "final"

    @property
    def logs_dir(self) -> Path:
        """Directory for logs."""
        return self.base_dir / "logs"

    @property
    def configs_dir(self) -> Path:
        """Directory for configuration files."""
        return self.base_dir / "configs"

    def get_locale_raw_dir(self, locale: str) -> Path:
        """Get raw data directory for a specific locale."""
        path = self.raw_data_dir / locale
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_locale_processed_dir(self, locale: str) -> Path:
        """Get processed data directory for a specific locale."""
        path = self.processed_data_dir / locale
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_locale_dataset_dir(self, locale: str) -> Path:
        """Get dataset directory for a specific locale."""
        path = self.datasets_dir / locale
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_locale_adapter_dir(self, locale: str) -> Path:
        """Get adapter directory for a specific locale."""
        path = self.adapters_dir / locale
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_pairs_file(self, locale: str) -> Path:
        """Get path to translation pairs JSONL file for a locale."""
        return self.get_locale_processed_dir(locale) / "pairs.jsonl"

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        dirs = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.datasets_dir,
            self.checkpoints_dir,
            self.adapters_dir,
            self.final_models_dir,
            self.logs_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
