"""WordPress.org API client for fetching translations via official APIs.

Uses the WordPress.org Plugin/Theme APIs and direct translation downloads
instead of scraping translate.wordpress.org pages.

API Endpoints:
- Plugin list: https://api.wordpress.org/plugins/info/1.2/?action=query_plugins
- Theme list: https://api.wordpress.org/themes/info/1.2/?action=query_themes
- Translations lookup: https://api.wordpress.org/translations/plugins/1.0/?slug={slug}
- Download URL: from translations API 'package' field
"""

import asyncio
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import aiofiles

from ..utils.logging import get_logger
from .client import RateLimitedClient

logger = get_logger(__name__)

# API endpoints
PLUGIN_API_URL = "https://api.wordpress.org/plugins/info/1.2/"
THEME_API_URL = "https://api.wordpress.org/themes/info/1.2/"

# Translations lookup API
PLUGIN_TRANSLATIONS_API = "https://api.wordpress.org/translations/plugins/1.0/"
THEME_TRANSLATIONS_API = "https://api.wordpress.org/translations/themes/1.0/"

# Common locale mappings (short code -> WordPress locale code)
LOCALE_MAP = {
    "nl": "nl_NL",
    "de": "de_DE",
    "fr": "fr_FR",
    "es": "es_ES",
    "it": "it_IT",
    "pt": "pt_PT",
    "ru": "ru_RU",
    "ja": "ja",
    "zh": "zh_CN",
    "ko": "ko_KR",
    "ar": "ar",
    "he": "he_IL",
    "pl": "pl_PL",
    "sv": "sv_SE",
    "da": "da_DK",
    "fi": "fi",
    "nb": "nb_NO",
    "cs": "cs_CZ",
    "tr": "tr_TR",
    "uk": "uk",
    "el": "el",
    "hu": "hu_HU",
    "ro": "ro_RO",
    "sk": "sk_SK",
    "bg": "bg_BG",
    "hr": "hr",
    "sr": "sr_RS",
    "sl": "sl_SI",
    "et": "et",
    "lv": "lv",
    "lt": "lt_LT",
}


@dataclass
class PluginInfo:
    """Information about a WordPress plugin from the API."""

    slug: str
    name: str
    version: str
    active_installs: int = 0
    rating: float = 0.0

    @property
    def project_type(self) -> str:
        return "wp-plugins"


@dataclass
class ThemeInfo:
    """Information about a WordPress theme from the API."""

    slug: str
    name: str
    version: str
    active_installs: int = 0
    rating: float = 0.0

    @property
    def project_type(self) -> str:
        return "wp-themes"


@dataclass
class FetchResult:
    """Result of a fetch operation."""

    locale: str
    files_downloaded: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    total_size_bytes: int = 0
    duration_seconds: float = 0.0
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.files_downloaded + self.files_failed
        if total == 0:
            return 0.0
        return (self.files_downloaded / total) * 100


class WordPressAPIClient:
    """Client for WordPress.org official APIs.

    Fetches plugin/theme lists and downloads translations via direct URLs.
    """

    def __init__(
        self,
        client: RateLimitedClient,
        output_dir: Path,
    ):
        """Initialize the API client.

        Args:
            client: Rate-limited HTTP client
            output_dir: Base directory for saving downloaded files
        """
        self.client = client
        self.output_dir = Path(output_dir)

    async def get_popular_plugins(
        self,
        limit: Optional[int] = None,
        browse: str = "popular",
    ) -> list[PluginInfo]:
        """Get list of popular plugins from WordPress.org API.

        Args:
            limit: Maximum number of plugins to return
            browse: Browse type (popular, new, updated, top-rated)

        Returns:
            List of plugin info objects
        """
        plugins: list[PluginInfo] = []
        page = 1
        per_page = 100

        while True:
            url = (
                f"{PLUGIN_API_URL}?action=query_plugins"
                f"&browse={browse}"
                f"&per_page={per_page}"
                f"&page={page}"
            )

            try:
                import json

                data = await self.client.get_text(url)
                response = json.loads(data)

                if "plugins" not in response or not response["plugins"]:
                    break

                for plugin in response["plugins"]:
                    plugins.append(
                        PluginInfo(
                            slug=plugin.get("slug", ""),
                            name=plugin.get("name", ""),
                            version=plugin.get("version", ""),
                            active_installs=plugin.get("active_installs", 0),
                            rating=plugin.get("rating", 0),
                        )
                    )

                logger.info(
                    f"Fetched page {page}: {len(response['plugins'])} plugins "
                    f"(total: {len(plugins)})"
                )

                if limit and len(plugins) >= limit:
                    plugins = plugins[:limit]
                    break

                # Check if we have all pages
                total_pages = response.get("info", {}).get("pages", 1)
                if page >= total_pages:
                    break

                page += 1

            except Exception as e:
                logger.warning(f"Error fetching plugins page {page}: {e}")
                break

        logger.info(f"Total plugins found: {len(plugins)}")
        return plugins

    async def get_popular_themes(
        self,
        limit: Optional[int] = None,
        browse: str = "popular",
    ) -> list[ThemeInfo]:
        """Get list of popular themes from WordPress.org API.

        Args:
            limit: Maximum number of themes to return
            browse: Browse type (popular, new, updated, top-rated)

        Returns:
            List of theme info objects
        """
        themes: list[ThemeInfo] = []
        page = 1
        per_page = 100

        while True:
            url = (
                f"{THEME_API_URL}?action=query_themes"
                f"&browse={browse}"
                f"&per_page={per_page}"
                f"&page={page}"
            )

            try:
                import json

                data = await self.client.get_text(url)
                response = json.loads(data)

                if "themes" not in response or not response["themes"]:
                    break

                for theme in response["themes"]:
                    themes.append(
                        ThemeInfo(
                            slug=theme.get("slug", ""),
                            name=theme.get("name", ""),
                            version=theme.get("version", ""),
                            active_installs=theme.get("active_installs", 0),
                            rating=theme.get("rating", 0),
                        )
                    )

                logger.info(
                    f"Fetched page {page}: {len(response['themes'])} themes "
                    f"(total: {len(themes)})"
                )

                if limit and len(themes) >= limit:
                    themes = themes[:limit]
                    break

                # Check if we have all pages
                total_pages = response.get("info", {}).get("pages", 1)
                if page >= total_pages:
                    break

                page += 1

            except Exception as e:
                logger.warning(f"Error fetching themes page {page}: {e}")
                break

        logger.info(f"Total themes found: {len(themes)}")
        return themes

    async def fetch_translations(
        self,
        locale: str,
        project_types: Optional[list[str]] = None,
        limit: Optional[int] = None,
        skip_existing: bool = True,
        concurrent_downloads: int = 5,
        progress_callback: Optional[callable] = None,
    ) -> FetchResult:
        """Fetch translation files for a locale.

        Downloads translation ZIP files and extracts PO files from them.

        Args:
            locale: Locale code (e.g., 'nl', 'de', 'fr')
            project_types: List of project types (wp-plugins, wp-themes)
            limit: Maximum projects per type
            skip_existing: Skip files that already exist
            concurrent_downloads: Number of concurrent downloads
            progress_callback: Optional callback(completed, total, bytes_downloaded, slug, status)
                               Called after each download with progress info

        Returns:
            FetchResult with statistics
        """
        start_time = time.time()
        result = FetchResult(locale=locale)

        if project_types is None:
            project_types = ["wp-plugins", "wp-themes"]

        # Create output directory
        locale_dir = self.output_dir / locale
        for project_type in project_types:
            (locale_dir / project_type).mkdir(parents=True, exist_ok=True)

        # Collect projects
        projects = []

        if "wp-plugins" in project_types:
            plugins = await self.get_popular_plugins(limit=limit)
            projects.extend(plugins)

        if "wp-themes" in project_types:
            themes = await self.get_popular_themes(limit=limit)
            projects.extend(themes)

        total_projects = len(projects)
        logger.info(f"Total projects to fetch translations for: {total_projects}")

        # Progress tracking
        completed_count = 0
        total_bytes = 0
        progress_lock = asyncio.Lock()

        # Fetch translations concurrently
        semaphore = asyncio.Semaphore(concurrent_downloads)

        async def fetch_with_semaphore(project):
            nonlocal completed_count, total_bytes
            async with semaphore:
                fetch_result = await self._fetch_translation(
                    project, locale, skip_existing
                )

                # Update progress
                async with progress_lock:
                    completed_count += 1
                    success, size, error = fetch_result
                    if success and size > 0:
                        total_bytes += size
                        status = "downloaded"
                    elif success:
                        status = "skipped"
                    else:
                        status = "failed"

                    if progress_callback:
                        try:
                            progress_callback(
                                completed_count,
                                total_projects,
                                total_bytes,
                                project.slug,
                                status,
                            )
                        except Exception:
                            pass  # Don't let callback errors break downloads

                return fetch_result

        tasks = [fetch_with_semaphore(p) for p in projects]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        for r in results:
            if isinstance(r, Exception):
                result.files_failed += 1
                result.errors.append(str(r))
            elif isinstance(r, tuple):
                success, size, error = r
                if success:
                    if size > 0:
                        result.files_downloaded += 1
                        result.total_size_bytes += size
                    else:
                        result.files_skipped += 1
                else:
                    result.files_failed += 1
                    if error:
                        result.errors.append(error)

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

    async def _fetch_translation(
        self,
        project: PluginInfo | ThemeInfo,
        locale: str,
        skip_existing: bool = True,
    ) -> tuple[bool, int, Optional[str]]:
        """Fetch translation for a single project.

        Args:
            project: Plugin or theme info
            locale: Locale code
            skip_existing: Skip if file already exists

        Returns:
            Tuple of (success, size_bytes, error_message)
        """
        # Determine output path
        output_path = (
            self.output_dir
            / locale
            / project.project_type
            / f"{project.slug}.po"
        )

        # Check if file exists
        if skip_existing and output_path.exists():
            logger.debug(f"Skipping existing: {output_path}")
            return (True, 0, None)

        # Map short locale to WordPress locale
        wp_locale = LOCALE_MAP.get(locale, locale)

        # Get download URL from translations API
        try:
            download_url = await self._get_translation_url(project, wp_locale)
            if not download_url:
                logger.debug(f"No {wp_locale} translation for {project.slug}")
                return (False, 0, f"No translation available for {project.slug}")
        except Exception as e:
            logger.debug(f"Error looking up translation for {project.slug}: {e}")
            return (False, 0, f"Lookup failed for {project.slug}: {e}")

        try:
            # Download ZIP file
            content = await self.client.get(download_url)

            # Extract PO file from ZIP
            po_content = self._extract_po_from_zip(content, project.slug)

            if po_content is None:
                return (False, 0, f"No PO file found in ZIP for {project.slug}")

            # Save PO file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(output_path, "wb") as f:
                await f.write(po_content)

            logger.debug(f"Downloaded: {project.slug} ({len(po_content)} bytes)")
            return (True, len(po_content), None)

        except Exception as e:
            # 404 errors are common for projects without translations
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str:
                logger.debug(f"No translation for {project.slug}: {e}")
            else:
                logger.warning(f"Error fetching {project.slug}: {e}")
            return (False, 0, f"Error fetching {project.slug}: {e}")

    async def _get_translation_url(
        self,
        project: PluginInfo | ThemeInfo,
        wp_locale: str,
    ) -> Optional[str]:
        """Get the download URL for a project's translation.

        Args:
            project: Plugin or theme info
            wp_locale: WordPress locale code (e.g., 'nl_NL')

        Returns:
            Download URL or None if not found
        """
        # Build translations API URL
        if isinstance(project, PluginInfo):
            url = f"{PLUGIN_TRANSLATIONS_API}?slug={project.slug}"
        else:
            url = f"{THEME_TRANSLATIONS_API}?slug={project.slug}"

        try:
            data = await self.client.get_text(url)
            response = json.loads(data)

            translations = response.get("translations", [])
            for trans in translations:
                if trans.get("language") == wp_locale:
                    return trans.get("package")

            # Also check for partial matches (nl matches nl_NL, nl_BE, etc.)
            for trans in translations:
                lang = trans.get("language", "")
                if lang.startswith(wp_locale.split("_")[0] + "_"):
                    return trans.get("package")

            return None

        except Exception as e:
            logger.debug(f"Error fetching translations for {project.slug}: {e}")
            return None

    def _extract_po_from_zip(
        self,
        zip_content: bytes,
        slug: str,
    ) -> Optional[bytes]:
        """Extract PO file from a translation ZIP archive.

        Args:
            zip_content: ZIP file content as bytes
            slug: Plugin/theme slug (for naming)

        Returns:
            PO file content as bytes, or None if not found
        """
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                # Look for .po files in the ZIP
                po_files = [n for n in zf.namelist() if n.endswith(".po")]

                if not po_files:
                    return None

                # Prefer the main translation file (slug-locale.po)
                # or just take the first .po file
                main_po = None
                for name in po_files:
                    if slug in name:
                        main_po = name
                        break

                if main_po is None:
                    main_po = po_files[0]

                return zf.read(main_po)

        except zipfile.BadZipFile:
            logger.warning(f"Invalid ZIP file for {slug}")
            return None
        except Exception as e:
            logger.warning(f"Error extracting ZIP for {slug}: {e}")
            return None

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
