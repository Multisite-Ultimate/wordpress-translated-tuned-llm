"""Dataset building for fine-tuning."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer

from ..parser.po_parser import TranslationPair
from ..utils.logging import get_logger
from .formatter import PromptFormatter, get_formatter, get_language_name
from .splitter import DatasetSplitter, SplitStats

logger = get_logger(__name__)


@dataclass
class DatasetStats:
    """Statistics about a built dataset."""

    num_train_examples: int
    num_test_examples: int
    avg_source_length: float
    avg_target_length: float
    avg_prompt_tokens: float
    max_prompt_tokens: int
    project_distribution: dict[str, int]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "num_train_examples": self.num_train_examples,
            "num_test_examples": self.num_test_examples,
            "avg_source_length": round(self.avg_source_length, 2),
            "avg_target_length": round(self.avg_target_length, 2),
            "avg_prompt_tokens": round(self.avg_prompt_tokens, 2),
            "max_prompt_tokens": self.max_prompt_tokens,
            "project_distribution": self.project_distribution,
        }


class DatasetBuilder:
    """Build HuggingFace datasets from translation pairs."""

    def __init__(
        self,
        model_name: str,
        formatter: Optional[PromptFormatter] = None,
        splitter: Optional[DatasetSplitter] = None,
        max_seq_length: int = 512,
        source_lang: str = "en",
        target_lang: str = "nl",
    ):
        """Initialize the dataset builder.

        Args:
            model_name: Model name for tokenizer
            formatter: Prompt formatter (auto-detected if None)
            splitter: Dataset splitter (default 80/20 if None)
            max_seq_length: Maximum sequence length
            source_lang: Source language code
            target_lang: Target language code
        """
        self.model_name = model_name
        self.formatter = formatter or get_formatter(model_name)
        self.splitter = splitter or DatasetSplitter(test_size=0.2)
        self.max_seq_length = max_seq_length
        self.source_lang = get_language_name(source_lang)
        self.target_lang = get_language_name(target_lang)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_from_pairs(
        self,
        pairs: list[TranslationPair],
        output_dir: Optional[Path] = None,
        save_raw: bool = True,
    ) -> tuple[DatasetDict, DatasetStats, SplitStats]:
        """Build dataset from translation pairs.

        Args:
            pairs: List of translation pairs
            output_dir: Directory to save dataset (optional)
            save_raw: Whether to save raw pairs as JSONL

        Returns:
            Tuple of (dataset_dict, dataset_stats, split_stats)
        """
        logger.info(f"Building dataset from {len(pairs)} pairs")

        # Split into train/test
        train_pairs, test_pairs, split_stats = self.splitter.split(pairs)

        # Format and create datasets
        train_dataset = self._create_dataset(train_pairs, "train")
        test_dataset = self._create_dataset(test_pairs, "test")

        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
        })

        # Compute statistics
        stats = self._compute_stats(train_pairs, test_pairs, train_dataset)

        # Save if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save HuggingFace dataset
            dataset_dict.save_to_disk(str(output_dir))
            logger.info(f"Saved dataset to {output_dir}")

            # Save raw pairs
            if save_raw:
                self._save_raw_pairs(train_pairs, output_dir / "train_pairs.jsonl")
                self._save_raw_pairs(test_pairs, output_dir / "test_pairs.jsonl")

            # Save stats
            stats_path = output_dir / "stats.json"
            with open(stats_path, "w") as f:
                json.dump({
                    "dataset_stats": stats.to_dict(),
                    "split_stats": split_stats.to_dict(),
                }, f, indent=2)

        return dataset_dict, stats, split_stats

    def build_from_file(
        self,
        pairs_file: Path,
        output_dir: Optional[Path] = None,
    ) -> tuple[DatasetDict, DatasetStats, SplitStats]:
        """Build dataset from JSONL file of pairs.

        Args:
            pairs_file: Path to JSONL file with pairs
            output_dir: Directory to save dataset (optional)

        Returns:
            Tuple of (dataset_dict, dataset_stats, split_stats)
        """
        pairs = []
        with open(pairs_file, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                pairs.append(TranslationPair.from_dict(data))

        return self.build_from_pairs(pairs, output_dir)

    def _create_dataset(
        self,
        pairs: list[TranslationPair],
        split_name: str,
    ) -> Dataset:
        """Create HuggingFace Dataset from pairs.

        Args:
            pairs: List of translation pairs
            split_name: Name of the split (train/test)

        Returns:
            HuggingFace Dataset
        """
        records = []

        for pair in tqdm(pairs, desc=f"Formatting {split_name}"):
            # Format the training example
            text = self.formatter.format_training_example(
                source=pair.source,
                target=pair.target,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )

            # Create inference prompt for evaluation
            prompt = self.formatter.format_inference_prompt(
                source=pair.source,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )

            records.append({
                "text": text,
                "prompt": prompt,
                "source": pair.source,
                "target": pair.target,
                "project_type": pair.project_type,
                "project_name": pair.project_name,
            })

        dataset = Dataset.from_list(records)

        # Tokenize for length analysis (but don't store tokenized data)
        logger.info(f"Created {split_name} dataset with {len(dataset)} examples")

        return dataset

    def _compute_stats(
        self,
        train_pairs: list[TranslationPair],
        test_pairs: list[TranslationPair],
        train_dataset: Dataset,
    ) -> DatasetStats:
        """Compute dataset statistics.

        Args:
            train_pairs: Training pairs
            test_pairs: Test pairs
            train_dataset: Training dataset

        Returns:
            DatasetStats object
        """
        all_pairs = train_pairs + test_pairs

        # Text lengths
        source_lengths = [len(p.source) for p in all_pairs]
        target_lengths = [len(p.target) for p in all_pairs]

        # Token lengths (sample for efficiency)
        sample_size = min(1000, len(train_dataset))
        sample = train_dataset.select(range(sample_size))

        token_lengths = []
        for text in sample["text"]:
            tokens = self.tokenizer.encode(text, truncation=False)
            token_lengths.append(len(tokens))

        # Project distribution
        from collections import Counter
        project_dist = Counter(p.project_type for p in all_pairs)

        return DatasetStats(
            num_train_examples=len(train_pairs),
            num_test_examples=len(test_pairs),
            avg_source_length=sum(source_lengths) / len(source_lengths),
            avg_target_length=sum(target_lengths) / len(target_lengths),
            avg_prompt_tokens=sum(token_lengths) / len(token_lengths),
            max_prompt_tokens=max(token_lengths),
            project_distribution=dict(project_dist),
        )

    def _save_raw_pairs(
        self,
        pairs: list[TranslationPair],
        output_path: Path,
    ) -> None:
        """Save raw pairs to JSONL file.

        Args:
            pairs: List of translation pairs
            output_path: Output file path
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    def filter_by_length(
        self,
        dataset: Dataset,
        max_tokens: Optional[int] = None,
    ) -> Dataset:
        """Filter dataset by sequence length.

        Args:
            dataset: Dataset to filter
            max_tokens: Maximum token length (uses max_seq_length if None)

        Returns:
            Filtered dataset
        """
        max_tokens = max_tokens or self.max_seq_length

        def is_valid_length(example):
            tokens = self.tokenizer.encode(
                example["text"],
                truncation=False,
            )
            return len(tokens) <= max_tokens

        original_size = len(dataset)
        filtered = dataset.filter(is_valid_length)
        removed = original_size - len(filtered)

        if removed > 0:
            logger.info(
                f"Filtered {removed} examples exceeding {max_tokens} tokens "
                f"({removed/original_size*100:.1f}%)"
            )

        return filtered

    def get_collator(self):
        """Get data collator for training.

        Returns:
            DataCollatorForLanguageModeling instance
        """
        from transformers import DataCollatorForLanguageModeling

        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked
        )
