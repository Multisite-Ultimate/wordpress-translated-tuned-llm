"""Download command for fetching translation files."""

import asyncio
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

app = typer.Typer()
console = Console()


class DownloadProgress:
    """Track and display download progress."""

    def __init__(self):
        self.completed = 0
        self.total = 0
        self.bytes_downloaded = 0
        self.start_time = time.time()
        self.last_slug = ""
        self.last_status = ""
        self.downloaded_count = 0
        self.skipped_count = 0
        self.failed_count = 0

    def update(self, completed: int, total: int, bytes_downloaded: int, slug: str, status: str):
        """Update progress stats."""
        self.completed = completed
        self.total = total
        self.bytes_downloaded = bytes_downloaded
        self.last_slug = slug
        self.last_status = status

        if status == "downloaded":
            self.downloaded_count += 1
        elif status == "skipped":
            self.skipped_count += 1
        elif status == "failed":
            self.failed_count += 1

    def get_rate(self) -> float:
        """Get download rate in files per second."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.completed / elapsed
        return 0.0

    def get_bytes_rate(self) -> float:
        """Get download rate in bytes per second."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.bytes_downloaded / elapsed
        return 0.0

    def get_eta(self) -> str:
        """Get estimated time remaining."""
        rate = self.get_rate()
        if rate > 0 and self.total > 0:
            remaining = (self.total - self.completed) / rate
            if remaining < 60:
                return f"{remaining:.0f}s"
            elif remaining < 3600:
                return f"{remaining / 60:.1f}m"
            else:
                return f"{remaining / 3600:.1f}h"
        return "..."

    def render(self) -> Table:
        """Render progress as a Rich table."""
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="cyan")
        table.add_column(justify="left")

        # Progress bar
        pct = (self.completed / self.total * 100) if self.total > 0 else 0
        bar_width = 30
        filled = int(bar_width * self.completed / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        table.add_row("Progress", f"[green]{bar}[/green] {pct:.1f}%")
        table.add_row("Files", f"{self.completed:,} / {self.total:,}")
        table.add_row("Downloaded", f"[green]{self.downloaded_count:,}[/green] | Skipped: [yellow]{self.skipped_count:,}[/yellow] | Failed: [red]{self.failed_count:,}[/red]")
        table.add_row("Size", f"{self.bytes_downloaded / 1e6:.2f} MB")
        table.add_row("Rate", f"{self.get_rate():.1f} files/s ({self.get_bytes_rate() / 1e6:.2f} MB/s)")
        table.add_row("ETA", self.get_eta())

        if self.last_slug:
            status_color = {"downloaded": "green", "skipped": "yellow", "failed": "red"}.get(self.last_status, "white")
            table.add_row("Current", f"[{status_color}]{self.last_slug}[/{status_color}]")

        return Panel(table, title="[bold]Downloading Translations[/bold]", border_style="blue")


@app.command("locale")
def download_locale(
    locale: str = typer.Argument(..., help="Locale code (e.g., 'nl', 'de', 'fr')"),
    project_types: list[str] = typer.Option(
        ["wp-plugins", "wp-themes"],
        "--type",
        "-t",
        help="Project types to download (wp-plugins, wp-themes)",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum projects per type (for testing)",
    ),
    output_dir: Path = typer.Option(
        "./data/raw",
        "--output",
        "-o",
        help="Output directory for downloaded files",
    ),
    rate_limit: int = typer.Option(
        60,
        "--rate-limit",
        "-r",
        help="Requests per minute",
    ),
    concurrent: int = typer.Option(
        10,
        "--concurrent",
        "-c",
        help="Concurrent downloads",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip",
        help="Skip files that already exist",
    ),
    use_scrape: bool = typer.Option(
        False,
        "--use-scrape",
        help="Use old scraping method instead of API",
    ),
):
    """Download all PO files for a specific locale.

    Uses the WordPress.org API to fetch plugin/theme lists and downloads
    translation ZIP files directly from downloads.wordpress.org.
    """
    from src.wp_translation.downloader import RateLimitedClient
    from src.wp_translation.utils.logging import setup_logging

    setup_logging(level="INFO")

    console.print(f"\n[bold]Downloading translations for locale: {locale}[/bold]\n")
    console.print(f"Project types: {', '.join(project_types)}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Method: {'Scraping' if use_scrape else 'WordPress API'}")
    if limit:
        console.print(f"Limit per type: {limit}")
    console.print()

    async def run_download():
        async with RateLimitedClient(requests_per_minute=rate_limit) as client:
            if use_scrape:
                # Use old scraping method
                from src.wp_translation.downloader import (
                    ProjectRegistry,
                    POFileFetcher,
                )

                registry = ProjectRegistry(client)
                fetcher = POFileFetcher(client, output_dir, registry)

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Downloading (scraping)...", total=None)

                    result = await fetcher.fetch_locale(
                        locale=locale,
                        project_types=project_types,
                        max_projects_per_type=limit,
                        skip_existing=skip_existing,
                        concurrent_downloads=concurrent,
                    )
            else:
                # Use new API method (default) with progress display
                from src.wp_translation.downloader import WordPressAPIClient

                api_client = WordPressAPIClient(client, output_dir)

                # Set up progress tracking
                progress_tracker = DownloadProgress()
                live = None

                def on_progress(completed, total, bytes_downloaded, slug, status):
                    progress_tracker.update(completed, total, bytes_downloaded, slug, status)
                    if live:
                        live.update(progress_tracker.render())

                console.print("[bold blue]Fetching project list from WordPress.org API...[/bold blue]")

                # Download with progress display
                with Live(progress_tracker.render(), console=console, refresh_per_second=4) as live_display:
                    live = live_display
                    result = await api_client.fetch_translations(
                        locale=locale,
                        project_types=project_types,
                        limit=limit,
                        skip_existing=skip_existing,
                        concurrent_downloads=concurrent,
                        progress_callback=on_progress,
                    )

            return result

    result = asyncio.run(run_download())

    console.print("\n[bold]Download Complete![/bold]")
    console.print(f"  Files downloaded: {result.files_downloaded}")
    console.print(f"  Files skipped: {result.files_skipped}")
    console.print(f"  Files failed: {result.files_failed}")
    console.print(f"  Total size: {result.total_size_bytes / 1e6:.2f} MB")
    console.print(f"  Duration: {result.duration_seconds:.1f}s")
    console.print(f"  Success rate: {result.success_rate:.1f}%")

    if result.errors:
        console.print(f"\n[yellow]Errors ({len(result.errors)}):[/yellow]")
        for error in result.errors[:10]:
            console.print(f"  - {error}")
        if len(result.errors) > 10:
            console.print(f"  ... and {len(result.errors) - 10} more")


@app.command("stats")
def download_stats(
    locale: str = typer.Argument(..., help="Locale code"),
    data_dir: Path = typer.Option(
        "./data/raw",
        "--data-dir",
        "-d",
        help="Data directory",
    ),
):
    """Show statistics about downloaded files for a locale."""
    locale_dir = data_dir / locale

    if not locale_dir.exists():
        console.print(f"[red]No data found for locale: {locale}[/red]")
        raise typer.Exit(1)

    # Count files by type
    stats = {}
    total_size = 0

    for project_type in ["wp-plugins", "wp-themes", "wp"]:
        type_dir = locale_dir / project_type
        if type_dir.exists():
            files = list(type_dir.glob("*.po"))
            size = sum(f.stat().st_size for f in files)
            stats[project_type] = {
                "count": len(files),
                "size_mb": size / 1e6,
            }
            total_size += size

    console.print(f"\n[bold]Download Statistics for {locale}[/bold]\n")

    for project_type, data in stats.items():
        console.print(
            f"  {project_type}: {data['count']} files ({data['size_mb']:.2f} MB)"
        )

    console.print(f"\n  Total: {sum(s['count'] for s in stats.values())} files")
    console.print(f"  Total size: {total_size / 1e6:.2f} MB")
