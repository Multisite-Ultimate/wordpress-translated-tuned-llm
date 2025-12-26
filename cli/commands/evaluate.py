"""Evaluation commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("run")
def run_evaluation(
    model_path: Path = typer.Argument(..., help="Path to model or adapter"),
    locale: str = typer.Argument(..., help="Target locale"),
    dataset_path: Optional[Path] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Dataset path (default: data/datasets/{locale})",
    ),
    num_samples: Optional[int] = typer.Option(
        None,
        "--samples",
        "-n",
        help="Number of samples to evaluate (default: all)",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        "-b",
        help="Batch size for generation",
    ),
    output_dir: Path = typer.Option(
        "./logs",
        "--output",
        "-o",
        help="Output directory for reports",
    ),
    report_format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Report format (markdown, json, html)",
    ),
):
    """Evaluate model on test dataset."""
    from datasets import load_from_disk

    from src.wp_translation.evaluation import (
        ModelEvaluator,
        EvaluationReporter,
        TranslationMetrics,
    )
    from src.wp_translation.utils.logging import setup_logging
    from src.wp_translation.utils.paths import PathManager

    setup_logging(level="INFO")
    paths = PathManager()

    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        raise typer.Exit(1)

    # Load dataset
    if dataset_path is None:
        dataset_path = paths.get_locale_dataset_dir(locale)

    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        raise typer.Exit(1)

    dataset = load_from_disk(str(dataset_path))

    if "test" not in dataset:
        console.print("[red]No test split in dataset[/red]")
        raise typer.Exit(1)

    test_dataset = dataset["test"]

    console.print(f"\n[bold]Evaluating Model[/bold]\n")
    console.print(f"  Model: {model_path}")
    console.print(f"  Locale: {locale}")
    console.print(f"  Test samples: {len(test_dataset)}")
    if num_samples:
        console.print(f"  Evaluating: {num_samples} samples")
    console.print()

    # Run evaluation
    metrics = TranslationMetrics(use_gpu=True)
    evaluator = ModelEvaluator(
        model_path=model_path,
        source_lang="English",
        target_lang=locale,
        batch_size=batch_size,
    )

    try:
        result = evaluator.evaluate(
            test_dataset,
            num_samples=num_samples,
        )

        # Show results
        console.print("\n[bold]Evaluation Results[/bold]\n")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Score", justify="right")
        table.add_column("Assessment")

        assessments = metrics.assess_quality(result.metrics)

        if result.metrics.comet is not None:
            table.add_row(
                "COMET",
                f"{result.metrics.comet:.4f}",
                assessments.get("comet", "N/A"),
            )
        if result.metrics.bleu is not None:
            table.add_row(
                "BLEU",
                f"{result.metrics.bleu:.2f}",
                assessments.get("bleu", "N/A"),
            )
        if result.metrics.chrf is not None:
            table.add_row(
                "ChrF",
                f"{result.metrics.chrf:.4f}",
                assessments.get("chrf", "N/A"),
            )

        console.print(table)

        console.print(f"\n  Generation time: {result.generation_time_seconds:.2f}s")
        console.print(
            f"  Avg time per sample: {result.avg_generation_time_per_sample:.4f}s"
        )

        # Generate report
        reporter = EvaluationReporter(
            output_dir=output_dir,
            model_name=str(model_path.name),
            locale=locale,
        )

        report_path = reporter.generate_report(result, format=report_format)
        predictions_path = reporter.save_predictions(result)

        console.print(f"\n[bold green]Report saved: {report_path}[/bold green]")
        console.print(f"[green]Predictions saved: {predictions_path}[/green]")

    finally:
        evaluator.cleanup()


@app.command("translate")
def translate_single(
    model_path: Path = typer.Argument(..., help="Path to model or adapter"),
    text: str = typer.Argument(..., help="Text to translate"),
    source_lang: str = typer.Option("en", "--source", "-s", help="Source language"),
    target_lang: str = typer.Option("nl", "--target", "-t", help="Target language"),
):
    """Translate a single text for testing."""
    from src.wp_translation.inference import WordPressTranslator

    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Translating...[/bold]\n")

    translator = WordPressTranslator(
        model_path=model_path,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    try:
        translation = translator.translate(text)

        console.print(f"[blue]Source ({source_lang}):[/blue]")
        console.print(f"  {text}\n")
        console.print(f"[green]Translation ({target_lang}):[/green]")
        console.print(f"  {translation}\n")

    finally:
        translator.cleanup()


@app.command("compare")
def compare_models(
    model_paths: list[Path] = typer.Argument(..., help="Paths to models to compare"),
    locale: str = typer.Option(..., "--locale", "-l", help="Target locale"),
    num_samples: int = typer.Option(100, "--samples", "-n", help="Samples per model"),
    output_dir: Path = typer.Option("./logs", "--output", "-o", help="Output directory"),
):
    """Compare multiple models on the same test set."""
    from datasets import load_from_disk

    from src.wp_translation.evaluation import ModelEvaluator, EvaluationReporter
    from src.wp_translation.utils.paths import PathManager

    paths = PathManager()
    dataset_path = paths.get_locale_dataset_dir(locale)

    if not dataset_path.exists():
        console.print(f"[red]Dataset not found for locale: {locale}[/red]")
        raise typer.Exit(1)

    dataset = load_from_disk(str(dataset_path))
    test_dataset = dataset["test"]

    results = {}

    for model_path in model_paths:
        if not model_path.exists():
            console.print(f"[yellow]Skipping missing model: {model_path}[/yellow]")
            continue

        console.print(f"\n[bold]Evaluating: {model_path.name}[/bold]")

        evaluator = ModelEvaluator(
            model_path=model_path,
            source_lang="English",
            target_lang=locale,
        )

        try:
            result = evaluator.evaluate(test_dataset, num_samples=num_samples)
            results[model_path.name] = result

            if result.metrics.comet:
                console.print(f"  COMET: {result.metrics.comet:.4f}")
            if result.metrics.bleu:
                console.print(f"  BLEU: {result.metrics.bleu:.2f}")

        finally:
            evaluator.cleanup()

    if len(results) > 1:
        reporter = EvaluationReporter(output_dir=output_dir, locale=locale)
        comparison_path = reporter.compare_models(results)
        console.print(f"\n[bold green]Comparison report: {comparison_path}[/bold green]")
