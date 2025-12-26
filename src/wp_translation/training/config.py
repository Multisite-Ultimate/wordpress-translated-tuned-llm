"""Training configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QuantizationConfig:
    """BitsAndBytes quantization configuration."""

    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    r: int = 64
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Model settings
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer_name: Optional[str] = None
    trust_remote_code: bool = False

    # Quantization
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Training hyperparameters
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 512

    # Memory optimization
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    tf32: bool = True

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # Multi-GPU settings
    ddp_find_unused_parameters: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Reproducibility
    seed: int = 42

    # Output paths
    output_dir: str = "./models/checkpoints"
    final_model_dir: str = "./models/final"
    adapter_dir: str = "./models/adapters"

    # Evaluation
    evaluation_strategy: str = "steps"
    eval_batch_size: int = 4

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TrainingConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            TrainingConfig instance
        """
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            TrainingConfig instance
        """
        # Handle nested configs
        if "quantization" in config_dict:
            config_dict["quantization"] = QuantizationConfig(
                **config_dict["quantization"]
            )

        if "lora" in config_dict:
            config_dict["lora"] = LoRAConfig(**config_dict["lora"])

        # Handle nested 'model' section
        if "model" in config_dict:
            model_config = config_dict.pop("model")
            if "name" in model_config:
                config_dict["model_name"] = model_config["name"]
            if "tokenizer" in model_config:
                config_dict["tokenizer_name"] = model_config["tokenizer"]
            if "trust_remote_code" in model_config:
                config_dict["trust_remote_code"] = model_config["trust_remote_code"]

        # Handle nested 'training' section
        if "training" in config_dict:
            training_config = config_dict.pop("training")
            for key, value in training_config.items():
                # Map YAML keys to dataclass fields
                if key == "num_epochs":
                    config_dict["num_epochs"] = value
                elif key == "per_device_batch_size":
                    config_dict["per_device_batch_size"] = value
                elif key == "gradient_accumulation_steps":
                    config_dict["gradient_accumulation_steps"] = value
                elif key == "learning_rate":
                    config_dict["learning_rate"] = value
                elif key == "warmup_ratio":
                    config_dict["warmup_ratio"] = value
                elif key == "weight_decay":
                    config_dict["weight_decay"] = value
                elif key == "max_grad_norm":
                    config_dict["max_grad_norm"] = value
                elif key == "lr_scheduler_type":
                    config_dict["lr_scheduler_type"] = value
                elif key == "max_seq_length":
                    config_dict["max_seq_length"] = value
                elif key == "gradient_checkpointing":
                    config_dict["gradient_checkpointing"] = value
                elif key == "optim":
                    config_dict["optim"] = value
                elif key == "bf16":
                    config_dict["bf16"] = value
                elif key == "tf32":
                    config_dict["tf32"] = value
                elif key == "logging_steps":
                    config_dict["logging_steps"] = value
                elif key == "save_steps":
                    config_dict["save_steps"] = value
                elif key == "eval_steps":
                    config_dict["eval_steps"] = value
                elif key == "save_total_limit":
                    config_dict["save_total_limit"] = value
                elif key == "ddp_find_unused_parameters":
                    config_dict["ddp_find_unused_parameters"] = value
                elif key == "dataloader_num_workers":
                    config_dict["dataloader_num_workers"] = value
                elif key == "dataloader_pin_memory":
                    config_dict["dataloader_pin_memory"] = value
                elif key == "seed":
                    config_dict["seed"] = value

        # Handle nested 'output' section
        if "output" in config_dict:
            output_config = config_dict.pop("output")
            if "dir" in output_config:
                config_dict["output_dir"] = output_config["dir"]
            if "final_model_dir" in output_config:
                config_dict["final_model_dir"] = output_config["final_model_dir"]
            if "adapter_dir" in output_config:
                config_dict["adapter_dir"] = output_config["adapter_dir"]

        # Handle nested 'evaluation' section
        if "evaluation" in config_dict:
            eval_config = config_dict.pop("evaluation")
            if "strategy" in eval_config:
                config_dict["evaluation_strategy"] = eval_config["strategy"]
            if "batch_size" in eval_config:
                config_dict["eval_batch_size"] = eval_config["batch_size"]

        # Filter out unknown keys
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}

        return cls(**filtered_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "trust_remote_code": self.trust_remote_code,
            "quantization": {
                "load_in_4bit": self.quantization.load_in_4bit,
                "bnb_4bit_compute_dtype": self.quantization.bnb_4bit_compute_dtype,
                "bnb_4bit_quant_type": self.quantization.bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": self.quantization.bnb_4bit_use_double_quant,
            },
            "lora": {
                "r": self.lora.r,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "target_modules": self.lora.target_modules,
                "bias": self.lora.bias,
                "task_type": self.lora.task_type,
            },
            "num_epochs": self.num_epochs,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_seq_length": self.max_seq_length,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "bf16": self.bf16,
            "seed": self.seed,
            "output_dir": self.output_dir,
        }

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        import torch

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        return (
            self.per_device_batch_size
            * self.gradient_accumulation_steps
            * num_gpus
        )
