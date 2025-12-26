"""Dataset splitting utilities."""

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from sklearn.model_selection import train_test_split

from ..parser.po_parser import TranslationPair
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SplitStats:
    """Statistics about a dataset split."""

    train_size: int
    test_size: int
    train_ratio: float
    test_ratio: float
    train_project_distribution: dict[str, int]
    test_project_distribution: dict[str, int]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "train_size": self.train_size,
            "test_size": self.test_size,
            "train_ratio": round(self.train_ratio, 4),
            "test_ratio": round(self.test_ratio, 4),
            "train_project_distribution": self.train_project_distribution,
            "test_project_distribution": self.test_project_distribution,
        }


class DatasetSplitter:
    """Split translation pairs into train/test sets."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_by_project: bool = True,
        min_samples_per_class: int = 2,
    ):
        """Initialize the splitter.

        Args:
            test_size: Fraction of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility
            stratify_by_project: Whether to stratify by project type
            min_samples_per_class: Minimum samples required per class for stratification
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify_by_project = stratify_by_project
        self.min_samples_per_class = min_samples_per_class

    def split(
        self,
        pairs: list[TranslationPair],
    ) -> tuple[list[TranslationPair], list[TranslationPair], SplitStats]:
        """Split pairs into train and test sets.

        Args:
            pairs: List of translation pairs to split

        Returns:
            Tuple of (train_pairs, test_pairs, split_stats)
        """
        if len(pairs) < 10:
            raise ValueError(f"Not enough pairs to split: {len(pairs)}")

        # Determine stratification
        stratify = None
        if self.stratify_by_project:
            stratify = self._get_stratification_labels(pairs)

        try:
            train_pairs, test_pairs = train_test_split(
                pairs,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError as e:
            # Stratification failed, fall back to random split
            logger.warning(f"Stratification failed ({e}), using random split")
            train_pairs, test_pairs = train_test_split(
                pairs,
                test_size=self.test_size,
                random_state=self.random_state,
            )

        stats = self._compute_stats(train_pairs, test_pairs)

        logger.info(
            f"Split complete: {stats.train_size} train, {stats.test_size} test "
            f"({stats.train_ratio:.1%}/{stats.test_ratio:.1%})"
        )

        return train_pairs, test_pairs, stats

    def _get_stratification_labels(
        self,
        pairs: list[TranslationPair],
    ) -> Optional[list[str]]:
        """Get stratification labels for pairs.

        Args:
            pairs: List of translation pairs

        Returns:
            List of labels for stratification, or None if not possible
        """
        labels = [p.project_type for p in pairs]
        label_counts = Counter(labels)

        # Check if stratification is possible
        # Need at least min_samples_per_class in each class
        min_test_samples = max(1, int(len(pairs) * self.test_size))

        for label, count in label_counts.items():
            if count < self.min_samples_per_class:
                logger.warning(
                    f"Project type '{label}' has only {count} samples, "
                    f"merging with 'other' for stratification"
                )
                # Merge rare labels into 'other'
                labels = [
                    l if label_counts[l] >= self.min_samples_per_class else "other"
                    for l in labels
                ]
                break

        # Final check
        final_counts = Counter(labels)
        if any(c < 2 for c in final_counts.values()):
            logger.warning("Cannot stratify, some classes too small")
            return None

        return labels

    def _compute_stats(
        self,
        train_pairs: list[TranslationPair],
        test_pairs: list[TranslationPair],
    ) -> SplitStats:
        """Compute statistics about the split.

        Args:
            train_pairs: Training set pairs
            test_pairs: Test set pairs

        Returns:
            SplitStats object
        """
        total = len(train_pairs) + len(test_pairs)

        train_dist = Counter(p.project_type for p in train_pairs)
        test_dist = Counter(p.project_type for p in test_pairs)

        return SplitStats(
            train_size=len(train_pairs),
            test_size=len(test_pairs),
            train_ratio=len(train_pairs) / total,
            test_ratio=len(test_pairs) / total,
            train_project_distribution=dict(train_dist),
            test_project_distribution=dict(test_dist),
        )

    def validate_split(
        self,
        train_pairs: list[TranslationPair],
        test_pairs: list[TranslationPair],
    ) -> tuple[bool, list[str]]:
        """Validate the quality of a split.

        Checks for:
        - No overlap between train and test
        - Reasonable distribution of project types
        - No data leakage

        Args:
            train_pairs: Training set pairs
            test_pairs: Test set pairs

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check for source text overlap
        train_sources = {p.source for p in train_pairs}
        test_sources = {p.source for p in test_pairs}
        overlap = train_sources & test_sources

        if overlap:
            issues.append(
                f"Found {len(overlap)} overlapping source texts between train and test"
            )

        # Check project type representation
        train_types = set(p.project_type for p in train_pairs)
        test_types = set(p.project_type for p in test_pairs)

        missing_in_test = train_types - test_types
        if missing_in_test:
            issues.append(
                f"Project types in train but not test: {missing_in_test}"
            )

        missing_in_train = test_types - train_types
        if missing_in_train:
            issues.append(
                f"Project types in test but not train: {missing_in_train}"
            )

        # Check minimum test set size
        if len(test_pairs) < 100:
            issues.append(f"Test set may be too small: {len(test_pairs)} samples")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Split validation passed")
        else:
            for issue in issues:
                logger.warning(f"Split validation issue: {issue}")

        return is_valid, issues

    def get_length_balanced_split(
        self,
        pairs: list[TranslationPair],
        length_bins: int = 5,
    ) -> tuple[list[TranslationPair], list[TranslationPair], SplitStats]:
        """Split with balanced length distribution.

        Ensures both train and test have similar distributions of
        source text lengths.

        Args:
            pairs: List of translation pairs
            length_bins: Number of length bins for stratification

        Returns:
            Tuple of (train_pairs, test_pairs, split_stats)
        """
        # Create length-based labels
        lengths = [len(p.source) for p in pairs]
        min_len, max_len = min(lengths), max(lengths)
        bin_size = (max_len - min_len) / length_bins

        labels = []
        for length in lengths:
            bin_idx = min(int((length - min_len) / bin_size), length_bins - 1)
            labels.append(f"len_{bin_idx}")

        try:
            train_pairs, test_pairs = train_test_split(
                pairs,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=labels,
            )
        except ValueError:
            # Fall back to regular split
            return self.split(pairs)

        stats = self._compute_stats(train_pairs, test_pairs)
        return train_pairs, test_pairs, stats
