"""Main CLI entry point for WordPress Translation LLM."""

import typer
from rich.console import Console

from cli.commands import download, parse, dataset, train, evaluate, serve

# Create main app
app = typer.Typer(
    name="wp-translate",
    help="WordPress Translation LLM Fine-tuning System",
    add_completion=False,
)

console = Console()

# Add command groups
app.add_typer(download.app, name="download", help="Download translation files")
app.add_typer(parse.app, name="parse", help="Parse PO files into translation pairs")
app.add_typer(dataset.app, name="dataset", help="Build training datasets")
app.add_typer(train.app, name="train", help="Fine-tune models")
app.add_typer(evaluate.app, name="evaluate", help="Evaluate model performance")
app.add_typer(serve.app, name="serve", help="Run inference server")


@app.command()
def version():
    """Show version information."""
    from src.wp_translation import __version__

    console.print(f"wp-translate version {__version__}")


@app.command()
def info():
    """Show system information and GPU status."""
    from src.wp_translation.utils.gpu_utils import get_gpu_info

    console.print("\n[bold]WordPress Translation LLM[/bold]\n")

    # GPU info
    gpus = get_gpu_info()
    if gpus:
        console.print("[bold]GPU Information:[/bold]")
        for gpu in gpus:
            console.print(
                f"  GPU {gpu.index}: {gpu.name} "
                f"({gpu.free_memory_gb:.1f}GB free / {gpu.total_memory_gb:.1f}GB total)"
            )
    else:
        console.print("[yellow]No GPUs detected[/yellow]")

    # Python and package info
    import sys
    console.print(f"\n[bold]Python:[/bold] {sys.version}")

    try:
        import torch
        console.print(f"[bold]PyTorch:[/bold] {torch.__version__}")
        console.print(f"[bold]CUDA Available:[/bold] {torch.cuda.is_available()}")
    except ImportError:
        console.print("[yellow]PyTorch not installed[/yellow]")

    try:
        import transformers
        console.print(f"[bold]Transformers:[/bold] {transformers.__version__}")
    except ImportError:
        console.print("[yellow]Transformers not installed[/yellow]")


@app.command()
def pipeline(
    locale: str = typer.Argument(..., help="Target locale (e.g., 'nl', 'de')"),
    model_name: str = typer.Option(
        "mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model to fine-tune",
    ),
    max_projects: int = typer.Option(
        None,
        help="Maximum projects to download (for testing)",
    ),
    skip_download: bool = typer.Option(
        False,
        help="Skip download step (use existing data)",
    ),
    skip_train: bool = typer.Option(
        False,
        help="Skip training step",
    ),
):
    """Run the full pipeline: download, parse, build dataset, train, evaluate."""
    import asyncio
    from pathlib import Path

    from src.wp_translation.utils.paths import PathManager
    from src.wp_translation.utils.logging import setup_logging

    setup_logging(level="INFO")
    paths = PathManager()
    paths.ensure_dirs()

    console.print(f"\n[bold]Running full pipeline for locale: {locale}[/bold]\n")

    # Step 1: Download
    if not skip_download:
        console.print("[bold]Step 1: Downloading translations...[/bold]")
        from src.wp_translation.downloader import (
            RateLimitedClient,
            ProjectRegistry,
            POFileFetcher,
        )

        async def run_download():
            async with RateLimitedClient() as client:
                registry = ProjectRegistry(client)
                fetcher = POFileFetcher(client, paths.raw_data_dir, registry)
                result = await fetcher.fetch_locale(
                    locale,
                    max_projects_per_type=max_projects,
                )
                return result

        result = asyncio.run(run_download())
        console.print(f"  Downloaded: {result.files_downloaded} files")
    else:
        console.print("[bold]Step 1: Skipping download[/bold]")

    # Step 2: Parse
    console.print("\n[bold]Step 2: Parsing PO files...[/bold]")
    from src.wp_translation.parser import PairExtractor

    extractor = PairExtractor()
    locale_dir = paths.get_locale_raw_dir(locale)
    pairs, stats = extractor.extract_from_directory(locale_dir)

    pairs_file = paths.get_pairs_file(locale)
    extractor.save_pairs(pairs, pairs_file)
    console.print(f"  Extracted: {stats.valid_pairs} pairs")

    # Step 3: Build dataset
    console.print("\n[bold]Step 3: Building dataset...[/bold]")
    from src.wp_translation.dataset import DatasetBuilder

    builder = DatasetBuilder(
        model_name=model_name,
        source_lang="en",
        target_lang=locale,
    )

    dataset_dir = paths.get_locale_dataset_dir(locale)
    dataset_dict, ds_stats, split_stats = builder.build_from_pairs(
        pairs,
        output_dir=dataset_dir,
    )
    console.print(f"  Train: {ds_stats.num_train_examples}, Test: {ds_stats.num_test_examples}")

    # Step 4: Train
    if not skip_train:
        console.print("\n[bold]Step 4: Fine-tuning model...[/bold]")
        from src.wp_translation.training import TrainingConfig, TranslationTrainer

        config = TrainingConfig(model_name=model_name)
        config.adapter_dir = str(paths.get_locale_adapter_dir(locale))
        config.output_dir = str(paths.checkpoints_dir / locale)

        trainer = TranslationTrainer(config)
        train_result = trainer.train(
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
        )
        console.print(f"  Final loss: {train_result.final_loss:.4f}")
        console.print(f"  Adapter saved: {train_result.adapter_path}")
    else:
        console.print("\n[bold]Step 4: Skipping training[/bold]")

    # Step 5: Evaluate
    console.print("\n[bold]Step 5: Evaluating model...[/bold]")
    from src.wp_translation.evaluation import ModelEvaluator, EvaluationReporter

    adapter_path = paths.get_locale_adapter_dir(locale)
    evaluator = ModelEvaluator(
        model_path=adapter_path,
        source_lang="English",
        target_lang=locale,
    )

    eval_result = evaluator.evaluate(
        dataset_dict["test"],
        num_samples=100,  # Evaluate on subset for speed
    )

    reporter = EvaluationReporter(
        output_dir=paths.logs_dir,
        model_name=model_name,
        locale=locale,
    )
    report_path = reporter.generate_report(eval_result)

    console.print(f"  COMET: {eval_result.metrics.comet:.4f}" if eval_result.metrics.comet else "")
    console.print(f"  BLEU: {eval_result.metrics.bleu:.2f}" if eval_result.metrics.bleu else "")
    console.print(f"  Report: {report_path}")

    console.print("\n[bold green]Pipeline complete![/bold green]\n")


if __name__ == "__main__":
    app()
