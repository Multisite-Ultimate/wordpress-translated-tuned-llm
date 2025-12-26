"""Server and inference commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command("start")
def start_server(
    model_path: Path = typer.Argument(..., help="Path to model or adapter"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    source_lang: str = typer.Option("en", "--source", "-s", help="Default source language"),
    target_lang: str = typer.Option("nl", "--target", "-t", help="Default target language"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the translation API server."""
    from src.wp_translation.inference.server import run_server

    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Starting Translation Server[/bold]\n")
    console.print(f"  Model: {model_path}")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  Languages: {source_lang} -> {target_lang}")
    console.print()
    console.print(f"API documentation will be available at http://{host}:{port}/docs")
    console.print()

    run_server(
        model_path=model_path,
        host=host,
        port=port,
        source_lang=source_lang,
        target_lang=target_lang,
        reload=reload,
    )


@app.command("export-gguf")
def export_gguf(
    model_path: Path = typer.Argument(..., help="Path to HuggingFace model"),
    output_path: Path = typer.Argument(..., help="Output GGUF file path"),
    quantization: str = typer.Option(
        "q4_k_m",
        "--quant",
        "-q",
        help="Quantization type (q4_k_m, q5_k_m, q8_0, f16)",
    ),
    llama_cpp_path: Optional[Path] = typer.Option(
        None,
        "--llama-cpp",
        help="Path to llama.cpp directory",
    ),
    merge_first: bool = typer.Option(
        False,
        "--merge",
        "-m",
        help="Merge LoRA adapters before export",
    ),
):
    """Export model to GGUF format for llama.cpp/Ollama."""
    from src.wp_translation.inference.export_gguf import (
        export_to_gguf,
        merge_lora_adapters,
    )

    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Exporting to GGUF[/bold]\n")
    console.print(f"  Model: {model_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Quantization: {quantization}")
    console.print()

    # Check if we need to merge first
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists() and merge_first:
        import tempfile
        import json

        with open(adapter_config) as f:
            config = json.load(f)
        base_model = config.get("base_model_name_or_path")

        console.print("Merging LoRA adapters first...")
        with tempfile.TemporaryDirectory() as temp_dir:
            merged_path = Path(temp_dir) / "merged"
            merge_lora_adapters(base_model, model_path, merged_path)
            model_path = merged_path

            console.print("Exporting to GGUF...")
            export_to_gguf(
                model_path=model_path,
                output_path=output_path,
                quantization=quantization,
                llama_cpp_path=llama_cpp_path,
            )
    else:
        export_to_gguf(
            model_path=model_path,
            output_path=output_path,
            quantization=quantization,
            llama_cpp_path=llama_cpp_path,
        )

    console.print(f"\n[bold green]GGUF exported: {output_path}[/bold green]")


@app.command("export-ollama")
def export_ollama(
    model_path: Path = typer.Argument(..., help="Path to HuggingFace model"),
    model_name: str = typer.Argument(..., help="Name for Ollama model"),
    output_dir: Path = typer.Option(
        "./models/ollama",
        "--output",
        "-o",
        help="Output directory",
    ),
    quantization: str = typer.Option(
        "q4_k_m",
        "--quant",
        "-q",
        help="Quantization type",
    ),
):
    """Export model for Ollama with Modelfile."""
    from src.wp_translation.inference.export_gguf import export_for_ollama

    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Exporting for Ollama[/bold]\n")
    console.print(f"  Model: {model_path}")
    console.print(f"  Name: {model_name}")
    console.print(f"  Quantization: {quantization}")
    console.print()

    result_dir = export_for_ollama(
        model_path=model_path,
        model_name=model_name,
        output_dir=output_dir,
        quantization=quantization,
    )

    console.print(f"\n[bold green]Ollama export complete![/bold green]")
    console.print(f"See {result_dir}/README.md for instructions.")


@app.command("translate-file")
def translate_file(
    model_path: Path = typer.Argument(..., help="Path to model"),
    input_file: Path = typer.Argument(..., help="Input PO file"),
    output_file: Path = typer.Argument(..., help="Output PO file"),
    source_lang: str = typer.Option("en", "--source", "-s"),
    target_lang: str = typer.Option("nl", "--target", "-t"),
    skip_translated: bool = typer.Option(
        True,
        "--skip-translated/--translate-all",
        help="Skip already translated entries",
    ),
):
    """Translate a PO file using the model."""
    from src.wp_translation.inference import WordPressTranslator

    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        raise typer.Exit(1)

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Translating PO File[/bold]\n")
    console.print(f"  Model: {model_path}")
    console.print(f"  Input: {input_file}")
    console.print(f"  Output: {output_file}")
    console.print()

    translator = WordPressTranslator(
        model_path=model_path,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    try:
        stats = translator.translate_po_file(
            input_file,
            output_file,
            skip_translated=skip_translated,
        )

        console.print("\n[bold]Translation Complete[/bold]")
        console.print(f"  Translated: {stats['translated']}")
        console.print(f"  Skipped: {stats['skipped']}")
        console.print(f"  Total entries: {stats['total']}")
        console.print(f"\n[green]Output saved: {output_file}[/green]")

    finally:
        translator.cleanup()
