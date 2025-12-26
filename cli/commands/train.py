"""Training commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command("run")
def run_training(
    locale: str = typer.Argument(..., help="Locale code (e.g., 'nl', 'de')"),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Training config YAML file",
    ),
    model_name: str = typer.Option(
        "mistralai/Mistral-7B-Instruct-v0.2",
        "--model",
        "-m",
        help="Base model to fine-tune",
    ),
    epochs: int = typer.Option(
        3,
        "--epochs",
        "-e",
        help="Number of training epochs",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Per-device batch size",
    ),
    learning_rate: float = typer.Option(
        2e-4,
        "--lr",
        help="Learning rate",
    ),
    lora_r: int = typer.Option(
        64,
        "--lora-r",
        help="LoRA rank",
    ),
    resume_from: Optional[Path] = typer.Option(
        None,
        "--resume",
        help="Resume from checkpoint",
    ),
    dataset_dir: Optional[Path] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Dataset directory",
    ),
):
    """Run QLoRA fine-tuning."""
    from src.wp_translation.training import TrainingConfig, TranslationTrainer
    from src.wp_translation.utils.logging import setup_logging
    from src.wp_translation.utils.paths import PathManager
    from src.wp_translation.utils.gpu_utils import check_gpu_memory
    from datasets import load_from_disk

    setup_logging(level="INFO")
    paths = PathManager()

    # Check GPU
    ok, message = check_gpu_memory(required_gb=16.0, min_gpus=1)
    if not ok:
        console.print(f"[yellow]Warning: {message}[/yellow]")

    console.print(f"\n[bold]Starting training for locale: {locale}[/bold]\n")

    # Load config
    if config_file:
        config = TrainingConfig.from_yaml(config_file)
    else:
        config = TrainingConfig(
            model_name=model_name,
            num_epochs=epochs,
            per_device_batch_size=batch_size,
            learning_rate=learning_rate,
        )
        config.lora.r = lora_r

    # Set output paths
    config.adapter_dir = str(paths.get_locale_adapter_dir(locale))
    config.output_dir = str(paths.checkpoints_dir / locale)

    console.print(f"Model: {config.model_name}")
    console.print(f"Epochs: {config.num_epochs}")
    console.print(f"Batch size: {config.per_device_batch_size}")
    console.print(f"Learning rate: {config.learning_rate}")
    console.print(f"LoRA rank: {config.lora.r}")
    console.print(f"Output: {config.adapter_dir}")
    console.print()

    # Load dataset
    if dataset_dir is None:
        dataset_dir = paths.get_locale_dataset_dir(locale)

    if not dataset_dir.exists():
        console.print(f"[red]Dataset not found: {dataset_dir}[/red]")
        console.print("Run 'wp-translate dataset build' first.")
        raise typer.Exit(1)

    console.print(f"Loading dataset from: {dataset_dir}")
    dataset = load_from_disk(str(dataset_dir))

    # Train
    trainer = TranslationTrainer(config)

    try:
        result = trainer.train(
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            resume_from_checkpoint=str(resume_from) if resume_from else None,
        )

        console.print("\n[bold green]Training Complete![/bold green]")
        console.print(f"  Final loss: {result.final_loss:.4f}")
        console.print(f"  Total steps: {result.total_steps}")
        console.print(f"  Training time: {result.training_time_seconds/3600:.2f} hours")
        console.print(f"  Adapter saved: {result.adapter_path}")

    except Exception as e:
        console.print(f"\n[red]Training failed: {e}[/red]")
        raise typer.Exit(1)

    finally:
        trainer.cleanup()


@app.command("merge")
def merge_adapters(
    adapter_path: Path = typer.Argument(..., help="Path to LoRA adapter"),
    output_path: Path = typer.Argument(..., help="Output path for merged model"),
    base_model: Optional[str] = typer.Option(
        None,
        "--base-model",
        "-m",
        help="Base model (auto-detected if not specified)",
    ),
):
    """Merge LoRA adapters into base model."""
    import json
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not adapter_path.exists():
        console.print(f"[red]Adapter not found: {adapter_path}[/red]")
        raise typer.Exit(1)

    # Auto-detect base model
    if base_model is None:
        config_path = adapter_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path")

    if not base_model:
        console.print("[red]Could not determine base model. Specify with --base-model.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Merging Adapters[/bold]")
    console.print(f"  Base model: {base_model}")
    console.print(f"  Adapter: {adapter_path}")
    console.print(f"  Output: {output_path}")
    console.print()

    # Load and merge
    console.print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    console.print("Loading adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    console.print("Merging...")
    model = model.merge_and_unload()

    console.print("Saving merged model...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    console.print(f"\n[bold green]Merged model saved to: {output_path}[/bold green]")


@app.command("info")
def training_info(
    adapter_path: Path = typer.Argument(..., help="Path to adapter or checkpoint"),
):
    """Show information about a trained model/adapter."""
    import json

    if not adapter_path.exists():
        console.print(f"[red]Path not found: {adapter_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Model Info: {adapter_path}[/bold]\n")

    # Check for adapter config
    adapter_config = adapter_path / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config) as f:
            config = json.load(f)

        console.print("[bold]Adapter Configuration:[/bold]")
        console.print(f"  Base model: {config.get('base_model_name_or_path')}")
        console.print(f"  LoRA r: {config.get('r')}")
        console.print(f"  LoRA alpha: {config.get('lora_alpha')}")
        console.print(f"  Target modules: {config.get('target_modules')}")
        console.print(f"  Task type: {config.get('task_type')}")

    # Check for training metrics
    metrics_file = adapter_path.parent / "training_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

        console.print("\n[bold]Training Metrics:[/bold]")
        for key, value in metrics.items():
            if isinstance(value, float):
                console.print(f"  {key}: {value:.4f}")
            else:
                console.print(f"  {key}: {value}")

    # List files
    console.print("\n[bold]Files:[/bold]")
    for f in sorted(adapter_path.iterdir()):
        size = f.stat().st_size / 1e6
        console.print(f"  {f.name}: {size:.2f} MB")
