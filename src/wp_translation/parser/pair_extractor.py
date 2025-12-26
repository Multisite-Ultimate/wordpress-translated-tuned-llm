"""Extract and filter high-quality translation pairs."""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

from ..utils.logging import get_logger
from .cleaner import TextCleaner
from .po_parser import POParser, TranslationPair

logger = get_logger(__name__)


@dataclass
class ExtractionStats:
    """Statistics from pair extraction."""

    total_parsed: int = 0
    valid_pairs: int = 0
    duplicates_removed: int = 0
    fuzzy_filtered: int = 0
    empty_filtered: int = 0
    length_filtered: int = 0
    same_text_filtered: int = 0
    placeholder_mismatch_filtered: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_parsed": self.total_parsed,
            "valid_pairs": self.valid_pairs,
            "duplicates_removed": self.duplicates_removed,
            "fuzzy_filtered": self.fuzzy_filtered,
            "empty_filtered": self.empty_filtered,
            "length_filtered": self.length_filtered,
            "same_text_filtered": self.same_text_filtered,
            "placeholder_mismatch_filtered": self.placeholder_mismatch_filtered,
            "acceptance_rate": (
                self.valid_pairs / self.total_parsed * 100
                if self.total_parsed > 0
                else 0
            ),
        }


class PairExtractor:
    """Extract and filter high-quality translation pairs from PO files."""

    def __init__(
        self,
        parser: Optional[POParser] = None,
        cleaner: Optional[TextCleaner] = None,
        min_source_length: int = 3,
        max_source_length: int = 512,
        filter_duplicates: bool = True,
        filter_fuzzy: bool = True,
        check_placeholders: bool = True,
    ):
        """Initialize the extractor.

        Args:
            parser: PO file parser (creates default if None)
            cleaner: Text cleaner (creates default if None)
            min_source_length: Minimum source text length
            max_source_length: Maximum source text length
            filter_duplicates: Whether to remove duplicate source texts
            filter_fuzzy: Whether to filter fuzzy translations
            check_placeholders: Whether to check placeholder consistency
        """
        self.parser = parser or POParser(include_fuzzy=not filter_fuzzy)
        self.cleaner = cleaner or TextCleaner()
        self.min_source_length = min_source_length
        self.max_source_length = max_source_length
        self.filter_duplicates = filter_duplicates
        self.filter_fuzzy = filter_fuzzy
        self.check_placeholders = check_placeholders

    def extract_from_directory(
        self,
        locale_dir: Path,
        show_progress: bool = True,
    ) -> tuple[list[TranslationPair], ExtractionStats]:
        """Extract all translation pairs from a locale directory.

        Args:
            locale_dir: Directory containing PO files for a locale
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (list of pairs, extraction statistics)
        """
        stats = ExtractionStats()
        pairs: list[TranslationPair] = []
        seen_sources: dict[str, TranslationPair] = {}

        # Get all PO files
        po_files = list(Path(locale_dir).rglob("*.po"))
        logger.info(f"Processing {len(po_files)} PO files from {locale_dir}")

        iterator = tqdm(po_files, desc="Extracting pairs") if show_progress else po_files

        for po_file in iterator:
            for pair in self.parser.parse_file(po_file):
                stats.total_parsed += 1

                # Apply cleaning
                cleaned_source = self.cleaner.clean(pair.source)
                cleaned_target = self.cleaner.clean(pair.target)

                # Update pair with cleaned text
                pair.source = cleaned_source
                pair.target = cleaned_target

                # Validate pair
                is_valid, reason = self._validate_pair(pair, stats)
                if not is_valid:
                    continue

                # Handle deduplication
                if self.filter_duplicates:
                    dedup_key = self.cleaner.normalize_for_dedup(pair.source)

                    if dedup_key in seen_sources:
                        stats.duplicates_removed += 1
                        # Keep the one with more context or from a more reputable source
                        existing = seen_sources[dedup_key]
                        if self._should_replace(existing, pair):
                            # Remove old pair from list
                            pairs = [p for p in pairs if p.source != existing.source]
                            seen_sources[dedup_key] = pair
                            pairs.append(pair)
                        continue

                    seen_sources[dedup_key] = pair

                pairs.append(pair)
                stats.valid_pairs += 1

        logger.info(
            f"Extraction complete: {stats.valid_pairs} valid pairs "
            f"from {stats.total_parsed} total ({stats.valid_pairs/max(stats.total_parsed, 1)*100:.1f}%)"
        )

        return pairs, stats

    def _validate_pair(
        self,
        pair: TranslationPair,
        stats: ExtractionStats,
    ) -> tuple[bool, Optional[str]]:
        """Validate a translation pair.

        Args:
            pair: Translation pair to validate
            stats: Statistics object to update

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check fuzzy
        if self.filter_fuzzy and pair.is_fuzzy:
            stats.fuzzy_filtered += 1
            return False, "Fuzzy translation"

        # Use cleaner's validation
        is_valid, reason = self.cleaner.is_valid_pair(
            pair.source,
            pair.target,
            min_length=self.min_source_length,
            max_length=self.max_source_length,
            check_placeholders=self.check_placeholders,
        )

        if not is_valid:
            # Update specific stat counters
            if reason and "Empty" in reason:
                stats.empty_filtered += 1
            elif reason and "length" in reason.lower():
                stats.length_filtered += 1
            elif reason and "equals" in reason.lower():
                stats.same_text_filtered += 1
            elif reason and "placeholder" in reason.lower():
                stats.placeholder_mismatch_filtered += 1

            return False, reason

        return True, None

    def _should_replace(
        self,
        existing: TranslationPair,
        new: TranslationPair,
    ) -> bool:
        """Determine if new pair should replace existing duplicate.

        Args:
            existing: Currently stored pair
            new: New pair to consider

        Returns:
            True if new pair should replace existing
        """
        # Prefer pairs with context
        if new.context and not existing.context:
            return True

        # Prefer core translations over plugins/themes
        priority = {"wordpress": 3, "wp-themes": 2, "wp-plugins": 1, "unknown": 0}
        new_priority = priority.get(new.project_type, 0)
        existing_priority = priority.get(existing.project_type, 0)

        if new_priority > existing_priority:
            return True

        # Prefer non-fuzzy over fuzzy
        if existing.is_fuzzy and not new.is_fuzzy:
            return True

        return False

    def save_pairs(
        self,
        pairs: list[TranslationPair],
        output_path: Path,
        include_metadata: bool = True,
    ) -> None:
        """Save translation pairs to JSONL file.

        Args:
            pairs: List of translation pairs
            output_path: Path for output file
            include_metadata: Whether to include full metadata
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                if include_metadata:
                    data = pair.to_dict()
                else:
                    # Minimal format for training
                    data = {
                        "source": pair.source,
                        "target": pair.target,
                        "project_type": pair.project_type,
                    }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(pairs)} pairs to {output_path}")

    def load_pairs(self, input_path: Path) -> Iterator[TranslationPair]:
        """Load translation pairs from JSONL file.

        Args:
            input_path: Path to JSONL file

        Yields:
            TranslationPair objects
        """
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                yield TranslationPair.from_dict(data)

    def get_project_distribution(
        self,
        pairs: list[TranslationPair],
    ) -> dict[str, int]:
        """Get distribution of pairs by project type.

        Args:
            pairs: List of translation pairs

        Returns:
            Dictionary mapping project type to count
        """
        distribution: dict[str, int] = defaultdict(int)
        for pair in pairs:
            distribution[pair.project_type] += 1
        return dict(distribution)

    def get_length_distribution(
        self,
        pairs: list[TranslationPair],
        bins: list[int] = None,
    ) -> dict[str, int]:
        """Get distribution of source text lengths.

        Args:
            pairs: List of translation pairs
            bins: Length bin boundaries

        Returns:
            Dictionary mapping length range to count
        """
        if bins is None:
            bins = [0, 10, 25, 50, 100, 200, 500, 1000]

        distribution: dict[str, int] = defaultdict(int)

        for pair in pairs:
            length = len(pair.source)
            for i, bin_max in enumerate(bins[1:], 1):
                if length <= bin_max:
                    range_str = f"{bins[i-1]+1}-{bin_max}"
                    distribution[range_str] += 1
                    break
            else:
                distribution[f">{bins[-1]}"] += 1

        return dict(distribution)
