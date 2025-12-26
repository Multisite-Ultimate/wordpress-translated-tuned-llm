"""Prompt formatting for LLM fine-tuning."""

from abc import ABC, abstractmethod
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Language code to full name mapping
LANGUAGE_NAMES = {
    "en": "English",
    "nl": "Dutch",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "hu": "Hungarian",
    "tr": "Turkish",
    "el": "Greek",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
}


def get_language_name(code: str) -> str:
    """Get full language name from code.

    Args:
        code: Language code (e.g., 'nl', 'de')

    Returns:
        Full language name
    """
    # Handle compound codes like 'pt_BR'
    base_code = code.split("_")[0].lower()
    return LANGUAGE_NAMES.get(base_code, code.capitalize())


class PromptFormatter(ABC):
    """Base class for prompt formatting."""

    @abstractmethod
    def format_training_example(
        self,
        source: str,
        target: str,
        source_lang: str = "English",
        target_lang: str = "Dutch",
    ) -> str:
        """Format a single training example.

        Args:
            source: Source text to translate
            target: Target translation
            source_lang: Source language name
            target_lang: Target language name

        Returns:
            Formatted training example string
        """
        pass

    @abstractmethod
    def format_inference_prompt(
        self,
        source: str,
        source_lang: str = "English",
        target_lang: str = "Dutch",
    ) -> str:
        """Format prompt for inference (no target).

        Args:
            source: Source text to translate
            source_lang: Source language name
            target_lang: Target language name

        Returns:
            Formatted inference prompt
        """
        pass

    def get_response_prefix(self) -> str:
        """Get the prefix that marks the start of the model response.

        Returns:
            Response prefix string
        """
        return ""


class MistralFormatter(PromptFormatter):
    """Mistral instruction format.

    Uses the Mistral instruct format:
    <s>[INST] instruction [/INST] response</s>
    """

    SYSTEM_PROMPT = (
        "You are a professional translator specializing in WordPress content. "
        "Translate the following text accurately while preserving any placeholders "
        "like %s, %d, %1$s, {name}, or HTML tags."
    )

    INSTRUCTION_TEMPLATE = (
        "Translate the following WordPress text from {source_lang} to {target_lang}. "
        "Preserve any placeholders like %s, %d, or {{name}}.\n\n"
        "{source}"
    )

    TRAINING_TEMPLATE = "<s>[INST] {instruction} [/INST]{target}</s>"

    INFERENCE_TEMPLATE = "<s>[INST] {instruction} [/INST]"

    def format_training_example(
        self,
        source: str,
        target: str,
        source_lang: str = "English",
        target_lang: str = "Dutch",
    ) -> str:
        """Format a training example in Mistral instruct format."""
        instruction = self.INSTRUCTION_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source=source,
        )

        return self.TRAINING_TEMPLATE.format(
            instruction=instruction,
            target=target,
        )

    def format_inference_prompt(
        self,
        source: str,
        source_lang: str = "English",
        target_lang: str = "Dutch",
    ) -> str:
        """Format an inference prompt in Mistral instruct format."""
        instruction = self.INSTRUCTION_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source=source,
        )

        return self.INFERENCE_TEMPLATE.format(instruction=instruction)

    def get_response_prefix(self) -> str:
        """Get the response prefix for Mistral format."""
        return "[/INST]"


class Llama3Formatter(PromptFormatter):
    """LLaMA 3 chat format.

    Uses the LLaMA 3 instruct format with special tokens.
    """

    SYSTEM_PROMPT = (
        "You are a professional translator specializing in WordPress content. "
        "Translate accurately while preserving placeholders and HTML tags."
    )

    INSTRUCTION_TEMPLATE = (
        "Translate the following WordPress text from {source_lang} to {target_lang}. "
        "Preserve any placeholders like %s, %d, or {{name}}.\n\n"
        "{source}"
    )

    TRAINING_TEMPLATE = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "{target}<|eot_id|>"
    )

    INFERENCE_TEMPLATE = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    def format_training_example(
        self,
        source: str,
        target: str,
        source_lang: str = "English",
        target_lang: str = "Dutch",
    ) -> str:
        """Format a training example in LLaMA 3 chat format."""
        instruction = self.INSTRUCTION_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source=source,
        )

        return self.TRAINING_TEMPLATE.format(
            system=self.SYSTEM_PROMPT,
            instruction=instruction,
            target=target,
        )

    def format_inference_prompt(
        self,
        source: str,
        source_lang: str = "English",
        target_lang: str = "Dutch",
    ) -> str:
        """Format an inference prompt in LLaMA 3 chat format."""
        instruction = self.INSTRUCTION_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source=source,
        )

        return self.INFERENCE_TEMPLATE.format(
            system=self.SYSTEM_PROMPT,
            instruction=instruction,
        )

    def get_response_prefix(self) -> str:
        """Get the response prefix for LLaMA 3 format."""
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"


class SimpleFormatter(PromptFormatter):
    """Simple format for testing or basic models.

    Uses a straightforward prompt format without special tokens.
    """

    TRAINING_TEMPLATE = (
        "### Translate from {source_lang} to {target_lang}:\n"
        "{source}\n\n"
        "### Translation:\n"
        "{target}"
    )

    INFERENCE_TEMPLATE = (
        "### Translate from {source_lang} to {target_lang}:\n"
        "{source}\n\n"
        "### Translation:\n"
    )

    def format_training_example(
        self,
        source: str,
        target: str,
        source_lang: str = "English",
        target_lang: str = "Dutch",
    ) -> str:
        """Format a training example in simple format."""
        return self.TRAINING_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source=source,
            target=target,
        )

    def format_inference_prompt(
        self,
        source: str,
        source_lang: str = "English",
        target_lang: str = "Dutch",
    ) -> str:
        """Format an inference prompt in simple format."""
        return self.INFERENCE_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source=source,
        )

    def get_response_prefix(self) -> str:
        """Get the response prefix for simple format."""
        return "### Translation:\n"


def get_formatter(model_name: str) -> PromptFormatter:
    """Get the appropriate formatter for a model.

    Args:
        model_name: Model name or identifier

    Returns:
        Appropriate PromptFormatter instance
    """
    model_lower = model_name.lower()

    if "mistral" in model_lower:
        return MistralFormatter()
    elif "llama-3" in model_lower or "llama3" in model_lower:
        return Llama3Formatter()
    else:
        logger.warning(f"Unknown model {model_name}, using Mistral format")
        return MistralFormatter()
