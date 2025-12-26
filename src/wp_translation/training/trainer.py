"""Translation fine-tuning trainer."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from ..utils.logging import get_logger
from .config import TrainingConfig
from .lora_config import LoRAConfigBuilder
from .model_loader import ModelLoader

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Result of a training run."""

    final_loss: float
    total_steps: int
    epochs_completed: float
    output_path: Path
    adapter_path: Path
    training_time_seconds: float
    best_checkpoint: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "final_loss": round(self.final_loss, 4),
            "total_steps": self.total_steps,
            "epochs_completed": round(self.epochs_completed, 2),
            "output_path": str(self.output_path),
            "adapter_path": str(self.adapter_path),
            "training_time_seconds": round(self.training_time_seconds, 2),
            "training_time_hours": round(self.training_time_seconds / 3600, 2),
            "best_checkpoint": self.best_checkpoint,
        }


class TranslationTrainer:
    """Trainer for fine-tuning LLMs on translation tasks."""

    def __init__(
        self,
        config: TrainingConfig,
        model_loader: Optional[ModelLoader] = None,
        lora_builder: Optional[LoRAConfigBuilder] = None,
    ):
        """Initialize the trainer.

        Args:
            config: Training configuration
            model_loader: Model loader (creates default if None)
            lora_builder: LoRA config builder (creates default if None)
        """
        self.config = config
        self.model_loader = model_loader or ModelLoader(config)
        self.lora_builder = lora_builder or LoRAConfigBuilder(config)

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> TrainingResult:
        """Run fine-tuning.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            TrainingResult with training metrics
        """
        import time

        start_time = time.time()

        # Load and prepare model
        logger.info("Loading and preparing model...")
        self.model = self.model_loader.load_base_model()

        # Check if model was loaded with quantization
        use_quantization = hasattr(self.model, 'quantization_config') or (
            hasattr(self.model.config, 'quantization_config') and
            self.model.config.quantization_config is not None
        )

        self.model = self.lora_builder.prepare_and_apply_lora(
            self.model,
            use_quantization=use_quantization,
        )
        self.tokenizer = self.model_loader.load_tokenizer()

        # Create training arguments
        training_args = self._create_training_args(eval_dataset is not None)

        # Create trainer
        logger.info("Creating SFT trainer...")
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
            packing=False,  # Don't pack examples for translation
        )

        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train(
            resume_from_checkpoint=resume_from_checkpoint,
        )

        # Save final model
        output_path = Path(self.config.output_dir)
        adapter_path = Path(self.config.adapter_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        adapter_path.mkdir(parents=True, exist_ok=True)

        # Save adapter weights
        logger.info(f"Saving adapter to {adapter_path}")
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        # Save training metrics
        metrics_path = output_path / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(train_result.metrics, f, indent=2)

        training_time = time.time() - start_time

        result = TrainingResult(
            final_loss=train_result.metrics.get("train_loss", 0.0),
            total_steps=train_result.global_step,
            epochs_completed=train_result.metrics.get("epoch", self.config.num_epochs),
            output_path=output_path,
            adapter_path=adapter_path,
            training_time_seconds=training_time,
            best_checkpoint=self.trainer.state.best_model_checkpoint,
        )

        logger.info(
            f"Training complete! "
            f"Loss: {result.final_loss:.4f}, "
            f"Steps: {result.total_steps}, "
            f"Time: {result.training_time_seconds/3600:.2f}h"
        )

        return result

    def _create_training_args(self, has_eval: bool) -> TrainingArguments:
        """Create training arguments from config.

        Args:
            has_eval: Whether evaluation dataset is provided

        Returns:
            TrainingArguments for the trainer
        """
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps" if has_eval else "no",
            eval_steps=self.config.eval_steps if has_eval else None,
            save_total_limit=self.config.save_total_limit,
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim=self.config.optim,
            bf16=self.config.bf16,
            tf32=self.config.tf32,
            ddp_find_unused_parameters=self.config.ddp_find_unused_parameters,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            seed=self.config.seed,
            report_to=["tensorboard"],
            logging_dir=str(Path(self.config.output_dir) / "logs"),
            load_best_model_at_end=has_eval,
            metric_for_best_model="eval_loss" if has_eval else None,
            greater_is_better=False,
            group_by_length=True,
            remove_unused_columns=False,
        )

    def save_adapter(
        self,
        output_path: Path,
        save_tokenizer: bool = True,
    ) -> None:
        """Save LoRA adapter weights.

        Args:
            output_path: Path to save adapter
            save_tokenizer: Whether to also save tokenizer
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Run train() first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving adapter to {output_path}")
        self.model.save_pretrained(output_path)

        if save_tokenizer and self.tokenizer:
            self.tokenizer.save_pretrained(output_path)

    def merge_and_save(
        self,
        output_path: Path,
        safe_serialization: bool = True,
    ) -> None:
        """Merge adapter with base model and save.

        Args:
            output_path: Path to save merged model
            safe_serialization: Use safetensors format
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Run train() first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Merging adapter with base model...")
        merged_model = self.model.merge_and_unload()

        logger.info(f"Saving merged model to {output_path}")
        merged_model.save_pretrained(
            output_path,
            safe_serialization=safe_serialization,
        )

        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)

    def evaluate(
        self,
        eval_dataset: Dataset,
    ) -> dict:
        """Evaluate the model on a dataset.

        Args:
            eval_dataset: Dataset to evaluate on

        Returns:
            Dictionary with evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Run train() first.")

        logger.info("Running evaluation...")
        metrics = self.trainer.evaluate(eval_dataset)

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def get_training_state(self) -> dict:
        """Get current training state.

        Returns:
            Dictionary with training state information
        """
        if self.trainer is None:
            return {"status": "not_started"}

        state = self.trainer.state

        return {
            "status": "training" if state.is_local_process_zero else "worker",
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "best_checkpoint": state.best_model_checkpoint,
            "log_history_length": len(state.log_history),
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.trainer is not None:
            del self.trainer
            self.trainer = None

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Cleaned up trainer resources")
