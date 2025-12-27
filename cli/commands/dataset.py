"""Dataset building commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("build")
def build_dataset(
    locale: str = typer.Argument(..., help="Locale code (e.g., 'nl', 'de')"),
    model_name: str = typer.Option(
        "mistralai/Mistral-7B-Instruct-v0.2",
        "--model",
        "-m",
        help="Model name for tokenizer",
    ),
    input_file: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input pairs JSONL file (default: data/processed/{locale}/pairs.jsonl)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: data/datasets/{locale})",
    ),
    test_size: float = typer.Option(
        0.2,
        "--test-size",
        "-t",
        help="Fraction for test set (default: 0.2)",
    ),
    max_seq_length: int = typer.Option(
        512,
        "--max-length",
        help="Maximum sequence length",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
):
    """Build HuggingFace datasets from translation pairs."""
    from src.wp_translation.dataset import DatasetBuilder, DatasetSplitter
    from src.wp_translation.utils.logging import setup_logging
    from src.wp_translation.utils.paths import PathManager

    setup_logging(level="INFO")
    paths = PathManager()

    # Set default paths
    if input_file is None:
        input_file = paths.get_pairs_file(locale)

    if output_dir is None:
        output_dir = paths.get_locale_dataset_dir(locale)

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        console.print("Run 'wp-translate parse extract' first.")
        raise typer.Exit(1)

    console.print(f"\n[bold]Building dataset for locale: {locale}[/bold]\n")
    console.print(f"Model: {model_name}")
    console.print(f"Test size: {test_size}")
    console.print(f"Max sequence length: {max_seq_length}")
    console.print()

    # Build dataset
    splitter = DatasetSplitter(test_size=test_size, random_state=seed)
    builder = DatasetBuilder(
        model_name=model_name,
        splitter=splitter,
        max_seq_length=max_seq_length,
        source_lang="en",
        target_lang=locale,
    )

    dataset_dict, stats, split_stats = builder.build_from_file(
        input_file,
        output_dir=output_dir,
    )

    # Show results
    console.print("\n[bold]Dataset Statistics[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Train examples", str(stats.num_train_examples))
    table.add_row("Test examples", str(stats.num_test_examples))
    table.add_row("Avg source length", f"{stats.avg_source_length:.1f} chars")
    table.add_row("Avg target length", f"{stats.avg_target_length:.1f} chars")
    table.add_row("Avg prompt tokens", f"{stats.avg_prompt_tokens:.1f}")
    table.add_row("Max prompt tokens", str(stats.max_prompt_tokens))

    console.print(table)

    console.print("\n[bold]Split Statistics[/bold]")
    console.print(f"  Train: {split_stats.train_size} ({split_stats.train_ratio:.1%})")
    console.print(f"  Test: {split_stats.test_size} ({split_stats.test_ratio:.1%})")

    console.print(f"\n[bold green]Dataset saved to: {output_dir}[/bold green]")


@app.command("info")
def dataset_info(
    locale: str = typer.Argument(..., help="Locale code"),
    data_dir: Path = typer.Option(
        "./data/datasets",
        "--data-dir",
        "-d",
        help="Datasets directory",
    ),
):
    """Show information about a built dataset."""
    from datasets import load_from_disk

    dataset_path = data_dir / locale

    if not dataset_path.exists():
        console.print(f"[red]No dataset found for locale: {locale}[/red]")
        console.print("Run 'wp-translate dataset build' first.")
        raise typer.Exit(1)

    dataset = load_from_disk(str(dataset_path))

    console.print(f"\n[bold]Dataset Info for {locale}[/bold]\n")
    console.print(dataset)

    # Show column info
    if "train" in dataset:
        console.print("\n[bold]Columns:[/bold]")
        for col in dataset["train"].column_names:
            console.print(f"  - {col}")

        console.print("\n[bold]Sample:[/bold]")
        sample = dataset["train"][0]
        for key, value in sample.items():
            if isinstance(value, str):
                console.print(f"  {key}: {value[:100]}...")
            else:
                console.print(f"  {key}: {value}")


@app.command("export")
def export_dataset(
    locale: str = typer.Argument(..., help="Locale code"),
    output_file: Path = typer.Argument(..., help="Output file path"),
    split: str = typer.Option(
        "train",
        "--split",
        "-s",
        help="Split to export (train or test)",
    ),
    format: str = typer.Option(
        "jsonl",
        "--format",
        "-f",
        help="Export format (jsonl, csv)",
    ),
    data_dir: Path = typer.Option(
        "./data/datasets",
        "--data-dir",
        "-d",
        help="Datasets directory",
    ),
):
    """Export dataset to a file."""
    from datasets import load_from_disk

    dataset_path = data_dir / locale

    if not dataset_path.exists():
        console.print(f"[red]No dataset found for locale: {locale}[/red]")
        raise typer.Exit(1)

    dataset = load_from_disk(str(dataset_path))

    if split not in dataset:
        console.print(f"[red]Split '{split}' not found. Available: {list(dataset.keys())}[/red]")
        raise typer.Exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        dataset[split].to_json(str(output_file))
    elif format == "csv":
        dataset[split].to_csv(str(output_file))
    else:
        console.print(f"[red]Unknown format: {format}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Exported {len(dataset[split])} examples to {output_file}[/green]")


@app.command("push")
def push_to_hub(
    locale: str = typer.Argument(..., help="Locale code (e.g., 'nl', 'de')"),
    repo_name: str = typer.Option(
        None,
        "--repo",
        "-r",
        help="Repository name (default: wordpress-translations-{locale})",
    ),
    organization: str = typer.Option(
        None,
        "--org",
        help="Organization name (uses your username if not specified)",
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Make the dataset private",
    ),
    data_dir: Path = typer.Option(
        "./data/datasets",
        "--data-dir",
        "-d",
        help="Datasets directory",
    ),
):
    """Push dataset to Hugging Face Hub for AutoTrain."""
    import json
    from datasets import load_from_disk
    from huggingface_hub import HfApi, login

    dataset_path = data_dir / locale
    stats_path = dataset_path / "stats.json"

    if not dataset_path.exists():
        console.print(f"[red]No dataset found for locale: {locale}[/red]")
        console.print("Run 'wp-translate dataset build' first.")
        raise typer.Exit(1)

    # Login to Hugging Face
    console.print("[bold]Authenticating with Hugging Face...[/bold]")
    try:
        api = HfApi()
        user_info = api.whoami()
        console.print(f"Logged in as: {user_info['name']}")
    except Exception:
        console.print("Please login to Hugging Face Hub first:")
        console.print("  Run: huggingface-cli login")
        raise typer.Exit(1)

    # Load dataset
    console.print(f"\n[bold]Loading dataset from {dataset_path}...[/bold]")
    dataset = load_from_disk(str(dataset_path))
    console.print(f"  Train: {len(dataset['train']):,} examples")
    console.print(f"  Test: {len(dataset['test']):,} examples")

    # Determine repo name
    final_repo_name = repo_name or f"wordpress-translations-{locale}"
    if organization:
        repo_id = f"{organization}/{final_repo_name}"
    else:
        repo_id = f"{user_info['name']}/{final_repo_name}"

    console.print(f"\n[bold]Pushing to: {repo_id}[/bold]")
    console.print(f"Visibility: {'private' if private else 'public'}")

    # Push to Hub
    dataset.push_to_hub(repo_id, private=private)

    console.print(f"\n[bold green]✓ Dataset uploaded successfully![/bold green]")
    console.print(f"  View at: https://huggingface.co/datasets/{repo_id}")
    console.print()
    console.print("[bold]Next steps for AutoTrain:[/bold]")
    console.print("  1. Go to https://huggingface.co/spaces/autotrain-projects/autotrain-advanced")
    console.print("  2. Select 'LLM SFT' task")
    console.print(f"  3. Choose your dataset: {repo_id}")
    console.print("  4. Set text column: 'text'")
    console.print("  5. Choose base model: mistralai/Mistral-7B-Instruct-v0.2")
    console.print("  6. Start training!")
