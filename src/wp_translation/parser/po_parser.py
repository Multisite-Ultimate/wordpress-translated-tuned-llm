"""PO file parser for extracting translation pairs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import polib

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranslationPair:
    """A single source-target translation pair."""

    source: str  # Original English text
    target: str  # Translated text
    context: Optional[str] = None  # PO context (msgctxt)
    source_file: str = ""  # Origin PO file path
    project_type: str = ""  # wp-plugins, wp-themes, or wordpress
    project_name: str = ""  # Plugin/theme slug or core component
    flags: list[str] = field(default_factory=list)  # PO flags (fuzzy, etc.)
    references: list[str] = field(default_factory=list)  # Source code references
    plural_source: Optional[str] = None  # msgid_plural if present
    plural_targets: list[str] = field(default_factory=list)  # msgstr[n] if plurals

    @property
    def is_plural(self) -> bool:
        """Check if this is a plural translation."""
        return self.plural_source is not None

    @property
    def is_fuzzy(self) -> bool:
        """Check if translation is marked as fuzzy."""
        return "fuzzy" in self.flags

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "context": self.context,
            "source_file": self.source_file,
            "project_type": self.project_type,
            "project_name": self.project_name,
            "flags": self.flags,
            "references": self.references,
            "plural_source": self.plural_source,
            "plural_targets": self.plural_targets,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TranslationPair":
        """Create from dictionary."""
        return cls(**data)


class POParser:
    """Parser for PO translation files."""

    def __init__(
        self,
        include_fuzzy: bool = False,
        include_obsolete: bool = False,
    ):
        """Initialize the parser.

        Args:
            include_fuzzy: Whether to include fuzzy translations
            include_obsolete: Whether to include obsolete entries
        """
        self.include_fuzzy = include_fuzzy
        self.include_obsolete = include_obsolete

    def parse_file(self, po_path: Path) -> Iterator[TranslationPair]:
        """Parse a single PO file, yielding translation pairs.

        Args:
            po_path: Path to the PO file

        Yields:
            TranslationPair objects for each valid entry
        """
        try:
            po = polib.pofile(str(po_path))
        except Exception as e:
            logger.error(f"Error parsing {po_path}: {e}")
            return

        # Extract project info from path
        project_type, project_name = self._extract_project_info(po_path)

        for entry in po:
            # Skip header entry
            if not entry.msgid:
                continue

            # Skip obsolete entries unless explicitly included
            if entry.obsolete and not self.include_obsolete:
                continue

            # Skip fuzzy entries unless explicitly included
            if "fuzzy" in entry.flags and not self.include_fuzzy:
                continue

            # Skip untranslated entries
            if not entry.msgstr and not entry.msgstr_plural:
                continue

            # Handle plural forms
            if entry.msgid_plural:
                # Create pairs for each plural form
                for idx, plural_str in entry.msgstr_plural.items():
                    if plural_str:  # Only if translated
                        yield TranslationPair(
                            source=entry.msgid if idx == 0 else entry.msgid_plural,
                            target=plural_str,
                            context=entry.msgctxt,
                            source_file=str(po_path),
                            project_type=project_type,
                            project_name=project_name,
                            flags=list(entry.flags),
                            references=[f"{r[0]}:{r[1]}" for r in entry.occurrences],
                            plural_source=entry.msgid_plural,
                            plural_targets=list(entry.msgstr_plural.values()),
                        )
            else:
                # Single form translation
                yield TranslationPair(
                    source=entry.msgid,
                    target=entry.msgstr,
                    context=entry.msgctxt,
                    source_file=str(po_path),
                    project_type=project_type,
                    project_name=project_name,
                    flags=list(entry.flags),
                    references=[f"{r[0]}:{r[1]}" for r in entry.occurrences],
                )

    def parse_directory(
        self,
        dir_path: Path,
        recursive: bool = True,
    ) -> Iterator[TranslationPair]:
        """Parse all PO files in a directory.

        Args:
            dir_path: Directory containing PO files
            recursive: Whether to search recursively

        Yields:
            TranslationPair objects from all PO files
        """
        pattern = "**/*.po" if recursive else "*.po"
        po_files = list(Path(dir_path).glob(pattern))

        logger.info(f"Found {len(po_files)} PO files in {dir_path}")

        for po_file in po_files:
            logger.debug(f"Parsing: {po_file}")
            yield from self.parse_file(po_file)

    def _extract_project_info(self, po_path: Path) -> tuple[str, str]:
        """Extract project type and name from file path.

        Args:
            po_path: Path to PO file

        Returns:
            Tuple of (project_type, project_name)
        """
        parts = po_path.parts

        # Look for known project type directories
        for i, part in enumerate(parts):
            if part in ("wp-plugins", "wp-themes", "wordpress", "wp"):
                project_type = "wordpress" if part == "wp" else part
                # Project name is typically the next part or filename
                if i + 1 < len(parts):
                    project_name = parts[i + 1]
                    if project_name.endswith(".po"):
                        project_name = project_name[:-3]
                else:
                    project_name = po_path.stem
                return project_type, project_name

        # Fallback: use parent dir and filename
        return "unknown", po_path.stem

    def get_file_stats(self, po_path: Path) -> dict:
        """Get statistics about a PO file.

        Args:
            po_path: Path to PO file

        Returns:
            Dictionary with file statistics
        """
        try:
            po = polib.pofile(str(po_path))
            return {
                "file": str(po_path),
                "total_entries": len(po),
                "translated": len(po.translated_entries()),
                "untranslated": len(po.untranslated_entries()),
                "fuzzy": len(po.fuzzy_entries()),
                "obsolete": len(po.obsolete_entries()),
                "percent_translated": po.percent_translated(),
            }
        except Exception as e:
            logger.error(f"Error getting stats for {po_path}: {e}")
            return {"file": str(po_path), "error": str(e)}
