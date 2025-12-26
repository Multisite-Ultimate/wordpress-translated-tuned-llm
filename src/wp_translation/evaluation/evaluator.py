"""Model evaluation for translation quality."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..dataset.formatter import PromptFormatter, get_formatter
from ..utils.logging import get_logger
from .metrics import MetricsResult, TranslationMetrics

logger = get_logger(__name__)


@dataclass
class TranslationSample:
    """A single translation sample with source, reference, and hypothesis."""

    source: str
    reference: str
    hypothesis: str
    project_type: str = ""


@dataclass
class EvaluationResult:
    """Result of model evaluation."""

    metrics: MetricsResult
    num_samples: int
    samples: list[TranslationSample] = field(default_factory=list)
    generation_time_seconds: float = 0.0
    avg_generation_time_per_sample: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics.to_dict(),
            "num_samples": self.num_samples,
            "generation_time_seconds": round(self.generation_time_seconds, 2),
            "avg_generation_time_per_sample": round(
                self.avg_generation_time_per_sample, 4
            ),
            "sample_count": len(self.samples),
        }


class ModelEvaluator:
    """Evaluate fine-tuned translation models."""

    def __init__(
        self,
        model_path: str | Path,
        formatter: Optional[PromptFormatter] = None,
        metrics: Optional[TranslationMetrics] = None,
        source_lang: str = "English",
        target_lang: str = "Dutch",
        batch_size: int = 8,
        max_new_tokens: int = 256,
        device: str = "cuda",
    ):
        """Initialize the evaluator.

        Args:
            model_path: Path to fine-tuned model or adapter
            formatter: Prompt formatter (auto-detected if None)
            metrics: Translation metrics computer
            source_lang: Source language name
            target_lang: Target language name
            batch_size: Batch size for generation
            max_new_tokens: Maximum tokens to generate
            device: Device to use
        """
        self.model_path = Path(model_path)
        self.formatter = formatter
        self.metrics = metrics or TranslationMetrics()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = device

        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load the fine-tuned model for evaluation."""
        logger.info(f"Loading model from {self.model_path}")

        # Check if this is an adapter or full model
        adapter_config = self.model_path / "adapter_config.json"
        is_adapter = adapter_config.exists()

        if is_adapter:
            # Load base model and adapter
            from peft import PeftModel

            # Read adapter config to get base model
            import json
            with open(adapter_config) as f:
                config = json.load(f)

            base_model_name = config.get(
                "base_model_name_or_path",
                "mistralai/Mistral-7B-Instruct-v0.2"
            )

            logger.info(f"Loading base model: {base_model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )

            logger.info("Loading adapter weights")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            self.model = self.model.merge_and_unload()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Set up tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set up formatter if not provided
        if self.formatter is None:
            model_name = getattr(self.model.config, "_name_or_path", "mistral")
            self.formatter = get_formatter(model_name)

        self.model.eval()
        logger.info("Model loaded and ready for evaluation")

    def evaluate(
        self,
        test_dataset: Dataset,
        num_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """Run full evaluation on test dataset.

        Args:
            test_dataset: Dataset with source and target fields
            num_samples: Number of samples to evaluate (None = all)
            show_progress: Whether to show progress bar

        Returns:
            EvaluationResult with metrics and samples
        """
        import time

        if self.model is None:
            self.load_model()

        # Limit samples if specified
        if num_samples and num_samples < len(test_dataset):
            test_dataset = test_dataset.select(range(num_samples))

        logger.info(f"Evaluating on {len(test_dataset)} samples")

        sources = []
        references = []
        hypotheses = []
        samples = []

        start_time = time.time()

        # Generate translations in batches
        iterator = range(0, len(test_dataset), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating translations")

        for i in iterator:
            batch = test_dataset[i:i + self.batch_size]

            # Get sources and references
            batch_sources = batch["source"]
            batch_refs = batch["target"]
            batch_project_types = batch.get("project_type", [""] * len(batch_sources))

            # Generate translations
            batch_hypotheses = self._generate_batch(batch_sources)

            sources.extend(batch_sources)
            references.extend(batch_refs)
            hypotheses.extend(batch_hypotheses)

            # Create samples
            for src, ref, hyp, proj in zip(
                batch_sources, batch_refs, batch_hypotheses, batch_project_types
            ):
                samples.append(TranslationSample(
                    source=src,
                    reference=ref,
                    hypothesis=hyp,
                    project_type=proj,
                ))

        generation_time = time.time() - start_time

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = self.metrics.compute_all(sources, hypotheses, references)

        result = EvaluationResult(
            metrics=metrics,
            num_samples=len(test_dataset),
            samples=samples,
            generation_time_seconds=generation_time,
            avg_generation_time_per_sample=generation_time / len(test_dataset),
        )

        logger.info(f"Evaluation complete: {metrics}")

        return result

    def _generate_batch(self, sources: list[str]) -> list[str]:
        """Generate translations for a batch of sources.

        Args:
            sources: List of source texts

        Returns:
            List of generated translations
        """
        # Format prompts
        prompts = [
            self.formatter.format_inference_prompt(
                source=src,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )
            for src in sources
        ]

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy for consistency
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and extract translations
        translations = []
        for i, output in enumerate(outputs):
            # Get only the generated part (after the prompt)
            generated = output[inputs["input_ids"][i].shape[0]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            translations.append(text.strip())

        return translations

    def evaluate_single(self, source: str) -> tuple[str, dict]:
        """Translate a single text and return with timing info.

        Args:
            source: Source text to translate

        Returns:
            Tuple of (translation, timing_info)
        """
        import time

        if self.model is None:
            self.load_model()

        start_time = time.time()
        translations = self._generate_batch([source])
        elapsed = time.time() - start_time

        return translations[0], {
            "generation_time_seconds": round(elapsed, 4),
            "source_length": len(source),
            "translation_length": len(translations[0]),
        }

    def get_best_and_worst_samples(
        self,
        result: EvaluationResult,
        n: int = 10,
    ) -> tuple[list[TranslationSample], list[TranslationSample]]:
        """Get best and worst translation samples based on similarity.

        Uses simple heuristics since per-sample COMET is expensive.

        Args:
            result: Evaluation result
            n: Number of samples to return

        Returns:
            Tuple of (best_samples, worst_samples)
        """
        from difflib import SequenceMatcher

        # Score each sample by reference similarity
        scored = []
        for sample in result.samples:
            ratio = SequenceMatcher(
                None,
                sample.reference.lower(),
                sample.hypothesis.lower(),
            ).ratio()
            scored.append((ratio, sample))

        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)

        best = [s for _, s in scored[:n]]
        worst = [s for _, s in scored[-n:]]

        return best, worst

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.metrics.cleanup()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Evaluator resources cleaned up")
