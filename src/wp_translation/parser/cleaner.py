"""Text cleaning and normalization for translation pairs."""

import html
import re
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """Clean and normalize translation text.

    Handles whitespace normalization, HTML entity conversion,
    and placeholder preservation.
    """

    # Common placeholder patterns in WordPress translations
    PLACEHOLDER_PATTERNS = [
        r"%\d*\$?[sdfg]",  # Printf-style: %s, %d, %1$s, %2$d
        r"%[sdfg]",  # Simple printf: %s, %d
        r"\{[^}]+\}",  # Mustache/WordPress: {name}, {count}
        r"\[\[[^\]]+\]\]",  # Double bracket: [[link]]
        r"<[^>]+>",  # HTML tags: <strong>, </a>
        r"&[a-zA-Z]+;",  # HTML entities: &nbsp;, &mdash;
        r"&#\d+;",  # Numeric HTML entities: &#8217;
    ]

    def __init__(
        self,
        normalize_whitespace: bool = True,
        decode_html_entities: bool = True,
        preserve_placeholders: bool = True,
    ):
        """Initialize the cleaner.

        Args:
            normalize_whitespace: Whether to normalize whitespace
            decode_html_entities: Whether to decode HTML entities
            preserve_placeholders: Whether to preserve placeholders
        """
        self.normalize_whitespace = normalize_whitespace
        self.decode_html_entities = decode_html_entities
        self.preserve_placeholders = preserve_placeholders

        # Compile placeholder regex
        self._placeholder_pattern = re.compile(
            "|".join(self.PLACEHOLDER_PATTERNS)
        )

    def clean(self, text: str) -> str:
        """Apply all cleaning steps to text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return text

        result = text

        # Decode HTML entities (but preserve placeholder entities)
        if self.decode_html_entities:
            result = self._decode_html_entities(result)

        # Normalize whitespace
        if self.normalize_whitespace:
            result = self._normalize_whitespace(result)

        return result

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.

        - Converts all whitespace to single spaces
        - Strips leading/trailing whitespace
        - Preserves intentional newlines in certain contexts

        Args:
            text: Text to normalize

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]

        # Join back, collapsing empty lines
        text = "\n".join(line for line in lines if line)

        return text.strip()

    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities in text.

        Args:
            text: Text with potential HTML entities

        Returns:
            Text with decoded entities
        """
        try:
            return html.unescape(text)
        except Exception:
            return text

    def extract_placeholders(self, text: str) -> list[str]:
        """Extract all placeholders from text.

        Args:
            text: Text to analyze

        Returns:
            List of placeholder strings found
        """
        return self._placeholder_pattern.findall(text)

    def placeholders_match(self, source: str, target: str) -> bool:
        """Check if placeholders in source and target match.

        Args:
            source: Source text
            target: Target text

        Returns:
            True if placeholders match (same set)
        """
        source_ph = set(self.extract_placeholders(source))
        target_ph = set(self.extract_placeholders(target))

        # For printf-style, we need to check count matches
        # Some languages reorder placeholders, so we check set equality
        return source_ph == target_ph

    def is_valid_pair(
        self,
        source: str,
        target: str,
        min_length: int = 1,
        max_length: int = 512,
        check_placeholders: bool = True,
    ) -> tuple[bool, Optional[str]]:
        """Validate a translation pair.

        Args:
            source: Source text
            target: Target text
            min_length: Minimum source length
            max_length: Maximum source length
            check_placeholders: Whether to verify placeholder consistency

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check for empty strings
        if not source or not source.strip():
            return False, "Empty source"

        if not target or not target.strip():
            return False, "Empty target"

        # Check source length
        source_len = len(source.strip())
        if source_len < min_length:
            return False, f"Source too short ({source_len} < {min_length})"

        if source_len > max_length:
            return False, f"Source too long ({source_len} > {max_length})"

        # Check for identical source and target
        if source.strip() == target.strip():
            # Allow if it's a proper noun, URL, or technical term
            if not self._is_likely_untranslatable(source):
                return False, "Source equals target"

        # Check placeholder consistency
        if check_placeholders:
            if not self.placeholders_match(source, target):
                return False, "Placeholder mismatch"

        return True, None

    def _is_likely_untranslatable(self, text: str) -> bool:
        """Check if text is likely untranslatable.

        Args:
            text: Text to check

        Returns:
            True if text appears to be untranslatable
        """
        # URLs
        if text.startswith(("http://", "https://", "www.")):
            return True

        # File paths
        if "/" in text and "." in text.split("/")[-1]:
            return True

        # Email addresses
        if "@" in text and "." in text:
            return True

        # Pure numbers
        if text.strip().replace(".", "").replace(",", "").isdigit():
            return True

        # Version numbers
        if re.match(r"^\d+\.\d+(\.\d+)?$", text.strip()):
            return True

        # Single words that are likely technical terms
        if len(text.split()) == 1 and text.isupper():
            return True

        return False

    def normalize_for_dedup(self, text: str) -> str:
        """Normalize text for deduplication.

        Creates a canonical form for comparing translations.

        Args:
            text: Text to normalize

        Returns:
            Normalized text for comparison
        """
        # Lowercase
        text = text.lower()

        # Remove all whitespace
        text = re.sub(r"\s+", "", text)

        # Remove punctuation for comparison
        text = re.sub(r"[^\w]", "", text)

        return text
