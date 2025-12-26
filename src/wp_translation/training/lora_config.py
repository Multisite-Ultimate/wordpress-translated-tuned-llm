"""LoRA configuration and model preparation."""

from typing import Optional

from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel

from ..utils.logging import get_logger
from .config import TrainingConfig

logger = get_logger(__name__)


class LoRAConfigBuilder:
    """Build and apply LoRA configuration for fine-tuning."""

    def __init__(self, config: TrainingConfig):
        """Initialize the LoRA config builder.

        Args:
            config: Training configuration
        """
        self.config = config

    def build_lora_config(self) -> PeftLoraConfig:
        """Create PEFT LoRA configuration.

        Returns:
            PeftLoraConfig for LoRA fine-tuning
        """
        # Map task type string to enum
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
            "SEQ_CLS": TaskType.SEQ_CLS,
        }

        task_type = task_type_map.get(
            self.config.lora.task_type,
            TaskType.CAUSAL_LM,
        )

        lora_config = PeftLoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            bias=self.config.lora.bias,
            task_type=task_type,
        )

        logger.info(
            f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
            f"dropout={lora_config.lora_dropout}, "
            f"targets={len(lora_config.target_modules)} modules"
        )

        return lora_config

    def prepare_model(
        self,
        model: PreTrainedModel,
        use_gradient_checkpointing: bool = True,
        use_quantization: bool = True,
    ) -> PreTrainedModel:
        """Prepare model for training.

        When using quantization, this applies the necessary modifications for training quantized models:
        - Casts layer norms to FP32
        - Makes embedding layer trainable
        - Enables gradient checkpointing

        Args:
            model: Base model to prepare
            use_gradient_checkpointing: Whether to enable gradient checkpointing
            use_quantization: Whether quantization is being used

        Returns:
            Prepared model
        """
        if use_quantization:
            logger.info("Preparing model for k-bit training")
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        else:
            logger.info("Skipping k-bit preparation (not using quantization)")
            # Just enable gradient checkpointing if requested
            if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        return model

    def apply_lora(
        self,
        model: PreTrainedModel,
        lora_config: Optional[PeftLoraConfig] = None,
    ) -> PreTrainedModel:
        """Apply LoRA adapters to the model.

        Args:
            model: Prepared base model
            lora_config: LoRA configuration (builds from config if None)

        Returns:
            Model with LoRA adapters applied
        """
        if lora_config is None:
            lora_config = self.build_lora_config()

        logger.info("Applying LoRA adapters to model")

        model = get_peft_model(model, lora_config)

        # Log trainable parameters
        self._log_trainable_params(model)

        return model

    def prepare_and_apply_lora(
        self,
        model: PreTrainedModel,
        use_quantization: bool = True,
    ) -> PreTrainedModel:
        """Prepare model and apply LoRA in one step.

        Args:
            model: Base model
            use_quantization: Whether quantization is being used

        Returns:
            Model ready for fine-tuning with LoRA
        """
        # Prepare for training (with or without k-bit preparation)
        model = self.prepare_model(
            model,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
            use_quantization=use_quantization,
        )

        # Apply LoRA adapters
        model = self.apply_lora(model)

        return model

    def _log_trainable_params(self, model: PreTrainedModel) -> None:
        """Log information about trainable parameters.

        Args:
            model: Model with LoRA applied
        """
        trainable_params = 0
        all_params = 0

        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        trainable_percent = 100 * trainable_params / all_params

        logger.info(
            f"Trainable params: {trainable_params:,} / {all_params:,} "
            f"({trainable_percent:.2f}%)"
        )

    def get_target_modules_for_model(self, model_name: str) -> list[str]:
        """Get recommended target modules for a specific model.

        Args:
            model_name: Model name or path

        Returns:
            List of recommended target module names
        """
        model_lower = model_name.lower()

        # Mistral/Mixtral
        if "mistral" in model_lower or "mixtral" in model_lower:
            return [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        # LLaMA
        if "llama" in model_lower:
            return [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        # Qwen
        if "qwen" in model_lower:
            return [
                "c_attn",
                "c_proj",
                "w1",
                "w2",
            ]

        # Falcon
        if "falcon" in model_lower:
            return [
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ]

        # Default (works for most models)
        logger.warning(
            f"Unknown model architecture for {model_name}, "
            "using default target modules"
        )
        return [
            "q_proj",
            "v_proj",
        ]

    def estimate_lora_params(self) -> dict[str, int]:
        """Estimate the number of LoRA parameters.

        Returns:
            Dictionary with parameter counts
        """
        # Rough estimates based on typical transformer architectures
        # Actual counts depend on model architecture

        hidden_size = 4096  # Typical for 7B models
        num_layers = 32  # Typical for 7B models
        num_target_modules = len(self.config.lora.target_modules)

        # Each target module gets A and B matrices
        # A: (r, in_features), B: (out_features, r)
        params_per_module = 2 * self.config.lora.r * hidden_size

        # Total across all layers and modules
        total_lora_params = num_layers * num_target_modules * params_per_module

        return {
            "r": self.config.lora.r,
            "alpha": self.config.lora.alpha,
            "num_target_modules": num_target_modules,
            "estimated_lora_params": total_lora_params,
            "estimated_lora_params_millions": round(total_lora_params / 1e6, 2),
        }
