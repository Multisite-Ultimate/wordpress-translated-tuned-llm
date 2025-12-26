"""Translation interface for GlotPress integration."""

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..dataset.formatter import PromptFormatter, get_formatter, get_language_name
from ..utils.logging import get_logger

logger = get_logger(__name__)


class WordPressTranslator:
    """Translation interface for fine-tuned models.

    Provides a simple API for translating WordPress strings
    using the fine-tuned model.
    """

    def __init__(
        self,
        model_path: str | Path,
        source_lang: str = "en",
        target_lang: str = "nl",
        formatter: Optional[PromptFormatter] = None,
        device: str = "cuda",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        use_quantization: bool = True,
    ):
        """Initialize the translator.

        Args:
            model_path: Path to fine-tuned model or adapter
            source_lang: Source language code
            target_lang: Target language code
            formatter: Prompt formatter (auto-detected if None)
            device: Device to use
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            use_quantization: Whether to use 4-bit quantization
        """
        self.model_path = Path(model_path)
        self.source_lang = get_language_name(source_lang)
        self.target_lang = get_language_name(target_lang)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_quantization = use_quantization

        self.model = None
        self.tokenizer = None
        self.formatter = formatter

        self._load_model()

    def _load_model(self) -> None:
        """Load the fine-tuned model."""
        logger.info(f"Loading model from {self.model_path}")

        # Check if this is an adapter or full model
        adapter_config = self.model_path / "adapter_config.json"
        is_adapter = adapter_config.exists()

        if is_adapter:
            self._load_adapter_model()
        else:
            self._load_full_model()

        # Set up tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set up formatter if not provided
        if self.formatter is None:
            model_name = getattr(self.model.config, "_name_or_path", "mistral")
            self.formatter = get_formatter(model_name)

        self.model.eval()
        logger.info("Model loaded and ready for translation")

    def _load_adapter_model(self) -> None:
        """Load model with LoRA adapter."""
        import json
        from peft import PeftModel

        # Read adapter config to get base model
        with open(self.model_path / "adapter_config.json") as f:
            config = json.load(f)

        base_model_name = config.get(
            "base_model_name_or_path",
            "mistralai/Mistral-7B-Instruct-v0.2"
        )

        logger.info(f"Loading base model: {base_model_name}")

        if self.use_quantization:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map=self.device,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )

        logger.info("Loading adapter weights")
        self.model = PeftModel.from_pretrained(self.model, self.model_path)

        # Optionally merge for faster inference
        # self.model = self.model.merge_and_unload()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def _load_full_model(self) -> None:
        """Load full merged model."""
        if self.use_quantization:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map=self.device,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def translate(self, text: str) -> str:
        """Translate a single text.

        Args:
            text: Source text to translate

        Returns:
            Translated text
        """
        prompt = self.formatter.format_inference_prompt(
            source=text,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        translation = self.tokenizer.decode(generated, skip_special_tokens=True)

        return translation.strip()

    def translate_batch(
        self,
        texts: list[str],
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> list[str]:
        """Translate multiple texts.

        Args:
            texts: List of source texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of translated texts
        """
        from tqdm import tqdm

        translations = []
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            iterator = tqdm(iterator, desc="Translating")

        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_translations = self._translate_batch(batch)
            translations.extend(batch_translations)

        return translations

    def _translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a batch of texts.

        Args:
            texts: List of source texts

        Returns:
            List of translations
        """
        # Format prompts
        prompts = [
            self.formatter.format_inference_prompt(
                source=text,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )
            for text in texts
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
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Extract translations
        translations = []
        for i, output in enumerate(outputs):
            generated = output[inputs["input_ids"][i].shape[0]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            translations.append(text.strip())

        return translations

    def translate_po_file(
        self,
        input_path: Path,
        output_path: Path,
        skip_translated: bool = True,
    ) -> dict:
        """Translate an entire PO file.

        Args:
            input_path: Path to input PO file
            output_path: Path to output PO file
            skip_translated: Whether to skip already translated entries

        Returns:
            Dictionary with translation statistics
        """
        import polib

        po = polib.pofile(str(input_path))
        translated_count = 0
        skipped_count = 0

        for entry in po:
            # Skip header
            if not entry.msgid:
                continue

            # Skip already translated if requested
            if skip_translated and entry.msgstr:
                skipped_count += 1
                continue

            # Translate
            translation = self.translate(entry.msgid)
            entry.msgstr = translation
            translated_count += 1

        po.save(str(output_path))

        return {
            "translated": translated_count,
            "skipped": skipped_count,
            "total": len(po),
        }

    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_path": str(self.model_path),
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "quantized": self.use_quantization,
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Translator resources cleaned up")
