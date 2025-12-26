"""PO file fetcher for downloading translations from translate.wordpress.org."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time

import aiofiles

from ..utils.logging import get_logger
from .client import RateLimitedClient
from .project_registry import Project, ProjectRegistry

logger = get_logger(__name__)


@dataclass
class FetchResult:
    """Result of a fetch operation."""

    locale: str
    files_downloaded: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    total_size_bytes: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.files_downloaded + self.files_failed
        if total == 0:
            return 0.0
        return (self.files_downloaded / total) * 100


@dataclass
class ProjectFetchResult:
    """Result of fetching a single project."""

    project: Project
    success: bool
    file_path: Optional[Path] = None
    size_bytes: int = 0
    error: Optional[str] = None


class POFileFetcher:
    """Fetches PO translation files from translate.wordpress.org.

    Downloads translation files for a given locale across all project types.
    """

    def __init__(
        self,
        client: RateLimitedClient,
        output_dir: Path,
        registry: ProjectRegistry,
    ):
        """Initialize the fetcher.

        Args:
            client: Rate-limited HTTP client
            output_dir: Base directory for saving downloaded files
            registry: Project registry for discovering projects
        """
        self.client = client
        self.output_dir = Path(output_dir)
        self.registry = registry

    async def fetch_locale(
        self,
        locale: str,
        project_types: Optional[list[str]] = None,
        max_projects_per_type: Optional[int] = None,
        skip_existing: bool = True,
        concurrent_downloads: int = 5,
    ) -> FetchResult:
        """Fetch all PO files for a locale.

        Args:
            locale: Locale code (e.g., 'nl', 'de', 'fr')
            project_types: List of project types to fetch (default: all)
            max_projects_per_type: Maximum projects per type (for testing)
            skip_existing: Skip files that already exist
            concurrent_downloads: Number of concurrent downloads

        Returns:
            FetchResult with statistics about the operation
        """
        start_time = time.time()
        result = FetchResult(locale=locale)

        if project_types is None:
            project_types = ["wp-plugins", "wp-themes", "wp"]

        # Collect all projects to fetch
        all_projects: list[Project] = []

        for project_type in project_types:
            if project_type == "wp-plugins":
                projects = await self.registry.get_plugins(
                    locale, limit=max_projects_per_type
                )
            elif project_type == "wp-themes":
                projects = await self.registry.get_themes(
                    locale, limit=max_projects_per_type
                )
            elif project_type in ("wordpress", "wp"):
                projects = await self.registry.get_core_projects(locale)
            else:
                logger.warning(f"Unknown project type: {project_type}")
                continue

            all_projects.extend(projects)
            logger.info(f"Found {len(projects)} {project_type} projects for {locale}")

        logger.info(f"Total projects to fetch: {len(all_projects)}")

        # Create output directory structure
        locale_dir = self.output_dir / locale
        for project_type in project_types:
            (locale_dir / project_type).mkdir(parents=True, exist_ok=True)

        # Fetch projects with concurrency limit
        semaphore = asyncio.Semaphore(concurrent_downloads)

        async def fetch_with_semaphore(project: Project) -> ProjectFetchResult:
            async with semaphore:
                return await self.fetch_project(project, locale, skip_existing)

        tasks = [fetch_with_semaphore(p) for p in all_projects]
        project_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        for proj_result in project_results:
            if isinstance(proj_result, Exception):
                result.files_failed += 1
                result.errors.append(str(proj_result))
            elif isinstance(proj_result, ProjectFetchResult):
                if proj_result.success:
                    if proj_result.size_bytes > 0:
                        result.files_downloaded += 1
                        result.total_size_bytes += proj_result.size_bytes
                    else:
                        result.files_skipped += 1
                else:
                    result.files_failed += 1
                    if proj_result.error:
                        result.errors.append(proj_result.error)

        result.duration_seconds = time.time() - start_time

        logger.info(
            f"Fetch complete for {locale}: "
            f"{result.files_downloaded} downloaded, "
            f"{result.files_skipped} skipped, "
            f"{result.files_failed} failed, "
            f"{result.total_size_bytes / 1e6:.2f} MB, "
            f"{result.duration_seconds:.1f}s"
        )

        return result

    async def fetch_project(
        self,
        project: Project,
        locale: str,
        skip_existing: bool = True,
        version: str = "stable",
    ) -> ProjectFetchResult:
        """Fetch a single project's translation file.

        Args:
            project: Project to fetch
            locale: Locale code
            skip_existing: Skip if file already exists
            version: Version to fetch ('stable' or 'dev')

        Returns:
            ProjectFetchResult with download details
        """
        # Determine output path
        output_path = self._get_output_path(project, locale)

        # Check if file exists
        if skip_existing and output_path.exists():
            logger.debug(f"Skipping existing: {output_path}")
            return ProjectFetchResult(
                project=project,
                success=True,
                file_path=output_path,
                size_bytes=0,  # Indicates skipped
            )

        # Build export URL
        url = self.registry.build_export_url(project, locale, version)

        try:
            # Download content
            content = await self.client.get(url)

            # Check if content is valid PO file
            if not self._is_valid_po_content(content):
                return ProjectFetchResult(
                    project=project,
                    success=False,
                    error=f"Invalid PO content for {project.slug}",
                )

            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(output_path, "wb") as f:
                await f.write(content)

            logger.debug(f"Downloaded: {project.slug} ({len(content)} bytes)")

            return ProjectFetchResult(
                project=project,
                success=True,
                file_path=output_path,
                size_bytes=len(content),
            )

        except Exception as e:
            error_msg = f"Error fetching {project.slug}: {str(e)}"
            logger.warning(error_msg)
            return ProjectFetchResult(
                project=project,
                success=False,
                error=error_msg,
            )

    def _get_output_path(self, project: Project, locale: str) -> Path:
        """Get the output file path for a project.

        Args:
            project: Project being downloaded
            locale: Locale code

        Returns:
            Path where file should be saved
        """
        locale_dir = self.output_dir / locale

        if project.project_type == "wp":
            # Core projects: locale/wp/dev.po, etc.
            safe_slug = project.slug.replace("/", "-")
            return locale_dir / "wp" / f"{safe_slug}.po"
        else:
            # Plugins/themes: locale/{type}/{slug}.po
            return locale_dir / project.project_type / f"{project.slug}.po"

    def _is_valid_po_content(self, content: bytes) -> bool:
        """Check if content appears to be valid PO file.

        Args:
            content: Downloaded content

        Returns:
            True if content appears to be valid PO
        """
        try:
            text = content.decode("utf-8", errors="ignore")
            # Valid PO files should contain msgid and msgstr
            return "msgid" in text and "msgstr" in text
        except Exception:
            return False

    async def get_download_stats(self, locale: str) -> dict:
        """Get statistics about downloaded files for a locale.

        Args:
            locale: Locale code

        Returns:
            Dictionary with file counts and sizes
        """
        locale_dir = self.output_dir / locale
        if not locale_dir.exists():
            return {"total_files": 0, "total_size_bytes": 0}

        total_files = 0
        total_size = 0

        for po_file in locale_dir.rglob("*.po"):
            total_files += 1
            total_size += po_file.stat().st_size

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1e6, 2),
        }
