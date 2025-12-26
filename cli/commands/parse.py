"""Parse command for extracting translation pairs from PO files."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("extract")
def extract_pairs(
    locale: str = typer.Argument(..., help="Locale code (e.g., 'nl', 'de')"),
    input_dir: Path = typer.Option(
        "./data/raw",
        "--input",
        "-i",
        help="Input directory with PO files",
    ),
    output_dir: Path = typer.Option(
        "./data/processed",
        "--output",
        "-o",
        help="Output directory for processed pairs",
    ),
    min_length: int = typer.Option(
        3,
        "--min-length",
        help="Minimum source text length",
    ),
    max_length: int = typer.Option(
        512,
        "--max-length",
        help="Maximum source text length",
    ),
    include_fuzzy: bool = typer.Option(
        False,
        "--include-fuzzy",
        help="Include fuzzy translations",
    ),
    no_dedup: bool = typer.Option(
        False,
        "--no-dedup",
        help="Disable deduplication",
    ),
):
    """Extract and filter translation pairs from PO files."""
    from src.wp_translation.parser import PairExtractor
    from src.wp_translation.utils.logging import setup_logging

    setup_logging(level="INFO")

    locale_dir = input_dir / locale

    if not locale_dir.exists():
        console.print(f"[red]No data found at: {locale_dir}[/red]")
        console.print("Run 'wp-translate download locale' first.")
        raise typer.Exit(1)

    console.print(f"\n[bold]Extracting pairs for locale: {locale}[/bold]\n")

    extractor = PairExtractor(
        min_source_length=min_length,
        max_source_length=max_length,
        filter_duplicates=not no_dedup,
        filter_fuzzy=not include_fuzzy,
    )

    pairs, stats = extractor.extract_from_directory(locale_dir)

    # Save pairs
    output_path = output_dir / locale / "pairs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extractor.save_pairs(pairs, output_path)

    # Show statistics
    console.print("\n[bold]Extraction Statistics[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Count", justify="right")

    table.add_row("Total parsed", str(stats.total_parsed))
    table.add_row("Valid pairs", str(stats.valid_pairs))
    table.add_row("Duplicates removed", str(stats.duplicates_removed))
    table.add_row("Fuzzy filtered", str(stats.fuzzy_filtered))
    table.add_row("Empty filtered", str(stats.empty_filtered))
    table.add_row("Length filtered", str(stats.length_filtered))
    table.add_row("Same text filtered", str(stats.same_text_filtered))
    table.add_row("Placeholder mismatch", str(stats.placeholder_mismatch_filtered))

    console.print(table)

    # Project distribution
    distribution = extractor.get_project_distribution(pairs)
    console.print("\n[bold]Project Distribution[/bold]\n")
    for project_type, count in sorted(distribution.items(), key=lambda x: -x[1]):
        console.print(f"  {project_type}: {count}")

    console.print(f"\n[bold green]Saved {len(pairs)} pairs to {output_path}[/bold green]")


@app.command("stats")
def parse_stats(
    locale: str = typer.Argument(..., help="Locale code"),
    data_dir: Path = typer.Option(
        "./data/processed",
        "--data-dir",
        "-d",
        help="Processed data directory",
    ),
):
    """Show statistics about parsed translation pairs."""
    import json

    pairs_file = data_dir / locale / "pairs.jsonl"

    if not pairs_file.exists():
        console.print(f"[red]No parsed data found for locale: {locale}[/red]")
        console.print("Run 'wp-translate parse extract' first.")
        raise typer.Exit(1)

    # Load and analyze pairs
    pairs = []
    source_lengths = []
    target_lengths = []
    project_types = {}

    with open(pairs_file) as f:
        for line in f:
            pair = json.loads(line.strip())
            pairs.append(pair)
            source_lengths.append(len(pair["source"]))
            target_lengths.append(len(pair["target"]))

            pt = pair.get("project_type", "unknown")
            project_types[pt] = project_types.get(pt, 0) + 1

    console.print(f"\n[bold]Parsed Data Statistics for {locale}[/bold]\n")

    console.print(f"  Total pairs: {len(pairs)}")
    console.print(f"  Avg source length: {sum(source_lengths)/len(source_lengths):.1f} chars")
    console.print(f"  Avg target length: {sum(target_lengths)/len(target_lengths):.1f} chars")
    console.print(f"  Max source length: {max(source_lengths)} chars")
    console.print(f"  Max target length: {max(target_lengths)} chars")

    console.print("\n[bold]By Project Type:[/bold]")
    for pt, count in sorted(project_types.items(), key=lambda x: -x[1]):
        pct = count / len(pairs) * 100
        console.print(f"  {pt}: {count} ({pct:.1f}%)")


@app.command("sample")
def show_samples(
    locale: str = typer.Argument(..., help="Locale code"),
    count: int = typer.Option(10, "--count", "-n", help="Number of samples"),
    data_dir: Path = typer.Option(
        "./data/processed",
        "--data-dir",
        "-d",
        help="Processed data directory",
    ),
):
    """Show sample translation pairs."""
    import json
    import random

    pairs_file = data_dir / locale / "pairs.jsonl"

    if not pairs_file.exists():
        console.print(f"[red]No parsed data found for locale: {locale}[/red]")
        raise typer.Exit(1)

    # Load pairs
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))

    # Random sample
    samples = random.sample(pairs, min(count, len(pairs)))

    console.print(f"\n[bold]Sample Translation Pairs ({locale})[/bold]\n")

    for i, pair in enumerate(samples, 1):
        console.print(f"[bold]#{i}[/bold] ({pair.get('project_type', 'unknown')})")
        console.print(f"  [blue]Source:[/blue] {pair['source'][:100]}...")
        console.print(f"  [green]Target:[/green] {pair['target'][:100]}...")
        console.print()
