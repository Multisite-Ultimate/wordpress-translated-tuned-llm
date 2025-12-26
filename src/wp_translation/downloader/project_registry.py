"""WordPress project registry for discovering translation projects."""

import re
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup

from ..utils.logging import get_logger
from .client import RateLimitedClient

logger = get_logger(__name__)

# Base URL
BASE_URL = "https://translate.wordpress.org"

# Locale-based URL for listing projects with translations
# Example: https://translate.wordpress.org/locale/nl/default/wp-plugins/
LOCALE_PROJECTS_URL = "{base_url}/locale/{locale}/default/{project_type}/"

# Export URL template
# Example: https://translate.wordpress.org/projects/wp-plugins/woocommerce/stable/nl/default/export-translations/?format=po
EXPORT_URL_TEMPLATE = (
    "{base_url}/projects/{project_path}/{locale}/default/"
    "export-translations/?format=po"
)


@dataclass
class Project:
    """WordPress translation project."""

    name: str
    slug: str
    project_type: str  # wp-plugins, wp-themes, or wp
    url: str
    percent_translated: float = 0.0

    @property
    def full_path(self) -> str:
        """Get full project path for URL construction."""
        if self.project_type == "wp":
            return f"wp/{self.slug}"
        return f"{self.project_type}/{self.slug}"


class ProjectRegistry:
    """Registry for discovering WordPress translation projects.

    Scrapes translate.wordpress.org locale pages to find projects
    that have translations for a specific locale.
    """

    def __init__(self, client: RateLimitedClient):
        """Initialize the registry.

        Args:
            client: Rate-limited HTTP client for making requests
        """
        self.client = client
        self._cache: dict[str, list[Project]] = {}

    async def get_projects_for_locale(
        self,
        locale: str,
        project_type: str,
        limit: Optional[int] = None,
        page_limit: Optional[int] = None,
    ) -> list[Project]:
        """Get list of projects with translations for a locale.

        Args:
            locale: Locale code (e.g., 'nl', 'de', 'fr')
            project_type: Type of projects (wp-plugins, wp-themes, wp)
            limit: Maximum number of projects to return
            page_limit: Maximum number of pages to scrape

        Returns:
            List of projects with translations
        """
        cache_key = f"{locale}:{project_type}"

        if cache_key in self._cache:
            projects = self._cache[cache_key]
        else:
            projects = await self._scrape_locale_projects(
                locale, project_type, page_limit
            )
            self._cache[cache_key] = projects

        if limit:
            return projects[:limit]
        return projects

    async def get_plugins(
        self,
        locale: str,
        limit: Optional[int] = None,
        page_limit: Optional[int] = None,
    ) -> list[Project]:
        """Get plugins with translations for a locale."""
        return await self.get_projects_for_locale(
            locale, "wp-plugins", limit, page_limit
        )

    async def get_themes(
        self,
        locale: str,
        limit: Optional[int] = None,
        page_limit: Optional[int] = None,
    ) -> list[Project]:
        """Get themes with translations for a locale."""
        return await self.get_projects_for_locale(
            locale, "wp-themes", limit, page_limit
        )

    async def get_core_projects(self, locale: str) -> list[Project]:
        """Get WordPress core translation projects for a locale.

        Returns:
            List of core projects (dev, admin, etc.)
        """
        # Core projects have known paths
        core_slugs = [
            ("dev", "Development"),
            ("dev/admin", "Administration"),
            ("dev/admin/network", "Network Admin"),
            ("dev/cc", "Continents & Cities"),
        ]

        return [
            Project(
                name=f"WordPress Core - {name}",
                slug=slug,
                project_type="wp",
                url=f"{BASE_URL}/locale/{locale}/default/wp/{slug}/",
            )
            for slug, name in core_slugs
        ]

    async def _scrape_locale_projects(
        self,
        locale: str,
        project_type: str,
        page_limit: Optional[int] = None,
    ) -> list[Project]:
        """Scrape project list from locale-specific page.

        Args:
            locale: Locale code (e.g., 'nl')
            project_type: Type of projects (wp-plugins or wp-themes)
            page_limit: Maximum number of pages to scrape

        Returns:
            List of discovered projects
        """
        projects: list[Project] = []
        page = 1
        max_pages = page_limit or 2000  # Safety limit

        base_url = LOCALE_PROJECTS_URL.format(
            base_url=BASE_URL,
            locale=locale,
            project_type=project_type,
        )

        while page <= max_pages:
            if page == 1:
                url = base_url
            else:
                url = f"{base_url}page/{page}/"

            logger.info(f"Scraping page {page}: {url}")

            try:
                html = await self.client.get_text(url)
                page_projects = self._parse_locale_project_page(
                    html, locale, project_type
                )

                if not page_projects:
                    logger.info(f"No more projects found on page {page}")
                    break

                projects.extend(page_projects)
                logger.info(f"Found {len(page_projects)} projects on page {page}")
                page += 1

            except Exception as e:
                # Check if it's a 404 (no more pages)
                error_str = str(e).lower()
                if "404" in error_str or "not found" in error_str:
                    logger.info(f"No more pages (page {page} returned 404)")
                else:
                    logger.warning(f"Error scraping page {page}: {e}")
                break

        logger.info(f"Total {project_type} projects found for {locale}: {len(projects)}")
        return projects

    def _parse_locale_project_page(
        self,
        html: str,
        locale: str,
        project_type: str,
    ) -> list[Project]:
        """Parse locale project page to extract project information.

        Args:
            html: HTML content of the locale project page
            locale: Locale code
            project_type: Type of projects being parsed

        Returns:
            List of projects found on the page
        """
        soup = BeautifulSoup(html, "lxml")
        projects: list[Project] = []

        # Pattern for project links: /locale/{locale}/default/{type}/{slug}/
        # Exclude ? to avoid matching pagination links like ?page=2
        pattern = re.compile(
            rf"/locale/{re.escape(locale)}/default/{re.escape(project_type)}/([^/?]+)/?$"
        )

        # Find all project links
        for link in soup.find_all("a", href=pattern):
            href = link.get("href", "")
            match = pattern.search(href)
            if match:
                slug = match.group(1)
                name = link.get_text(strip=True) or slug

                # Try to get translation percentage from parent element
                percent = 0.0
                parent = link.find_parent("div", class_="project")
                if parent:
                    progress_text = parent.find(string=re.compile(r"\d+%"))
                    if progress_text:
                        try:
                            percent = float(re.search(r"(\d+)%", progress_text).group(1))
                        except (AttributeError, ValueError):
                            pass

                projects.append(
                    Project(
                        name=name,
                        slug=slug,
                        project_type=project_type,
                        url=f"{BASE_URL}{href}",
                        percent_translated=percent,
                    )
                )

        # Deduplicate by slug
        seen = set()
        unique_projects = []
        for p in projects:
            if p.slug not in seen:
                seen.add(p.slug)
                unique_projects.append(p)

        return unique_projects

    def build_export_url(
        self,
        project: Project,
        locale: str,
        version: str = "stable",
    ) -> str:
        """Build the export URL for a project translation.

        Args:
            project: Project to export
            locale: Locale code (e.g., 'nl', 'de')
            version: Version to export ('stable' or 'dev')

        Returns:
            URL to download PO file
        """
        if project.project_type == "wp":
            # Core uses different path structure (no version)
            path = f"wp/{project.slug}"
        elif project.project_type == "wp-themes":
            # Themes don't have a version in the path
            path = f"wp-themes/{project.slug}"
        else:
            # Plugins use {type}/{slug}/{version}
            path = f"{project.project_type}/{project.slug}/{version}"

        return EXPORT_URL_TEMPLATE.format(
            base_url=BASE_URL,
            project_path=path,
            locale=locale,
        )

    async def check_translation_exists(
        self,
        project: Project,
        locale: str,
        version: str = "stable",
    ) -> bool:
        """Check if a translation exists for a project/locale combination.

        Args:
            project: Project to check
            locale: Locale code
            version: Version to check

        Returns:
            True if translation exists
        """
        url = self.build_export_url(project, locale, version)
        try:
            headers = await self.client.head(url)
            content_type = headers.get("Content-Type", "")
            # PO files should have text/plain or application/x-gettext
            return "text" in content_type or "gettext" in content_type
        except Exception:
            return False
