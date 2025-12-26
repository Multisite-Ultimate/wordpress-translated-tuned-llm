"""Model loading with quantization support."""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..utils.logging import get_logger
from .config import TrainingConfig

logger = get_logger(__name__)


class ModelLoader:
    """Load models with QLoRA quantization for fine-tuning."""

    def __init__(self, config: TrainingConfig):
        """Initialize the model loader.

        Args:
            config: Training configuration
        """
        self.config = config

    def get_quantization_config(self, use_8bit: bool = False) -> BitsAndBytesConfig:
        """Create BitsAndBytes quantization configuration.

        Args:
            use_8bit: If True, use 8-bit quantization instead of 4-bit

        Returns:
            BitsAndBytesConfig for quantization
        """
        if use_8bit:
            # 8-bit quantization works on older GPUs (compute capability >= 6.0)
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )

        # Map string dtype to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        compute_dtype = dtype_map.get(
            self.config.quantization.bnb_4bit_compute_dtype,
            torch.bfloat16,
        )

        return BitsAndBytesConfig(
            load_in_4bit=self.config.quantization.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
        )

    def load_base_model(
        self,
        device_map: Optional[str | dict] = "auto",
        use_quantization: bool = True,
    ) -> AutoModelForCausalLM:
        """Load base model with optional 4-bit quantization.

        Args:
            device_map: Device mapping strategy
            use_quantization: Whether to use 4-bit quantization (requires compute >= 7.5)

        Returns:
            Model ready for fine-tuning
        """
        logger.info(f"Loading model: {self.config.model_name}")

        # Check GPU compute capability for quantization support
        use_8bit = False
        skip_quantization = False
        if use_quantization and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if props.major < 7 or (props.major == 7 and props.minor < 5):
                # Try to use 8-bit quantization, but if bitsandbytes CUDA isn't working, skip
                try:
                    from bitsandbytes.cuda_setup.main import CUDASetup
                    cuda_setup = CUDASetup.get_instance()
                    if cuda_setup.cuda_available and not cuda_setup.binary_name.endswith('cpu.so'):
                        logger.warning(
                            f"GPU compute capability {props.major}.{props.minor} < 7.5, "
                            "switching to 8-bit quantization (slower but works on older GPUs)."
                        )
                        use_8bit = True
                    else:
                        logger.warning(
                            f"GPU compute capability {props.major}.{props.minor} < 7.5 and "
                            "bitsandbytes CUDA not available. Using FP16 without quantization."
                        )
                        skip_quantization = True
                except Exception as e:
                    logger.warning(f"bitsandbytes not properly configured: {e}. Skipping quantization.")
                    skip_quantization = True

        if skip_quantization:
            use_quantization = False

        # Prepare model loading kwargs
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float16,
        }

        # Add flash attention if available
        if self._supports_flash_attn():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Add quantization config if supported
        if use_quantization:
            try:
                bnb_config = self.get_quantization_config(use_8bit=use_8bit)
                model_kwargs["quantization_config"] = bnb_config
                if use_8bit:
                    logger.info("Using 8-bit quantization")
                else:
                    logger.info("Using 4-bit quantization")
            except Exception as e:
                logger.warning(f"Quantization not available: {e}. Using FP16.")
        else:
            logger.info("Using FP16 without quantization")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            logger.info("Enabled gradient checkpointing")

        # Log model info
        self._log_model_info(model)

        return model

    def load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer for the model.

        Returns:
            Configured tokenizer
        """
        tokenizer_name = self.config.tokenizer_name or self.config.model_name

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Set padding side for causal LM
        tokenizer.padding_side = "right"

        logger.info(f"Loaded tokenizer with vocab size: {len(tokenizer)}")

        return tokenizer

    def _supports_flash_attn(self) -> bool:
        """Check if Flash Attention 2 is available.

        Returns:
            True if Flash Attention 2 can be used
        """
        try:
            import flash_attn  # noqa: F401

            return True
        except ImportError:
            return False

    def _log_model_info(self, model: AutoModelForCausalLM) -> None:
        """Log information about the loaded model.

        Args:
            model: Loaded model
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Total parameters: {total_params / 1e9:.2f}B")
        logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        logger.info(
            f"Trainable %: {trainable_params / total_params * 100:.2f}%"
        )

        # Memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_used = torch.cuda.memory_allocated(i) / 1e9
                memory_reserved = torch.cuda.memory_reserved(i) / 1e9
                logger.info(
                    f"GPU {i} memory: {memory_used:.2f}GB used, "
                    f"{memory_reserved:.2f}GB reserved"
                )

    def estimate_memory_usage(self, use_quantization: bool = True) -> dict[str, float]:
        """Estimate GPU memory usage for training.

        Args:
            use_quantization: Whether 4-bit quantization will be used

        Returns:
            Dictionary with memory estimates in GB
        """
        model_params = 7e9  # 7B parameters

        if use_quantization:
            # 4-bit quantization: 0.5 bytes per parameter
            base_model_memory = model_params * 0.5 / 1e9
        else:
            # FP16: 2 bytes per parameter
            base_model_memory = model_params * 2 / 1e9

        # LoRA adapters (r=64, ~1% of model)
        lora_memory = model_params * 0.01 * 2 / 1e9  # FP16

        # Optimizer states (AdamW for LoRA params only)
        optimizer_memory = model_params * 0.01 * 8 / 1e9  # 2 states * 4 bytes each

        # Gradients and activations (rough estimate)
        batch_memory = (
            self.config.per_device_batch_size
            * self.config.max_seq_length
            * 4096  # Hidden size estimate
            * 4  # FP32 for gradients
            / 1e9
        )

        total = base_model_memory + lora_memory + optimizer_memory + batch_memory

        return {
            "base_model_gb": round(base_model_memory, 2),
            "lora_adapters_gb": round(lora_memory, 2),
            "optimizer_states_gb": round(optimizer_memory, 2),
            "batch_memory_gb": round(batch_memory, 2),
            "total_estimated_gb": round(total, 2),
            "quantization": use_quantization,
            "note": "With FP16 and 2 GPUs, model is split across devices" if not use_quantization else "",
        }

    def load_for_inference(
        self,
        adapter_path: Optional[str] = None,
        merge_adapters: bool = True,
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model for inference (without training overhead).

        Args:
            adapter_path: Path to LoRA adapter weights
            merge_adapters: Whether to merge adapters into base model

        Returns:
            Tuple of (model, tokenizer)
        """
        from peft import PeftModel

        # Load base model (still quantized for memory efficiency)
        model = self.load_base_model()
        tokenizer = self.load_tokenizer()

        # Load and optionally merge LoRA adapters
        if adapter_path:
            logger.info(f"Loading adapters from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)

            if merge_adapters:
                logger.info("Merging adapters into base model")
                model = model.merge_and_unload()

        model.eval()

        return model, tokenizer
