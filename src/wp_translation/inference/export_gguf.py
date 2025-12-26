"""Export models to GGUF format for llama.cpp compatibility."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


def merge_lora_adapters(
    base_model_path: str,
    adapter_path: str | Path,
    output_path: str | Path,
    safe_serialization: bool = True,
) -> Path:
    """Merge LoRA adapters into base model.

    Args:
        base_model_path: Path or HuggingFace ID of base model
        adapter_path: Path to LoRA adapter weights
        output_path: Path to save merged model
        safe_serialization: Use safetensors format

    Returns:
        Path to merged model
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging adapter with base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")
    model.save_pretrained(
        output_path,
        safe_serialization=safe_serialization,
    )

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    logger.info("Merge complete!")
    return output_path


def export_to_gguf(
    model_path: str | Path,
    output_path: str | Path,
    quantization: str = "q4_k_m",
    llama_cpp_path: Optional[str | Path] = None,
) -> Path:
    """Export model to GGUF format.

    Requires llama.cpp to be installed and the convert script available.

    Args:
        model_path: Path to HuggingFace model
        output_path: Path for output GGUF file
        quantization: Quantization type (q4_k_m, q5_k_m, q8_0, f16, etc.)
        llama_cpp_path: Path to llama.cpp directory

    Returns:
        Path to GGUF file

    Raises:
        FileNotFoundError: If llama.cpp is not found
        RuntimeError: If conversion fails
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find llama.cpp
    if llama_cpp_path:
        llama_cpp_path = Path(llama_cpp_path)
    else:
        # Try common locations
        possible_paths = [
            Path.home() / "llama.cpp",
            Path("/opt/llama.cpp"),
            Path("./llama.cpp"),
        ]
        for path in possible_paths:
            if path.exists():
                llama_cpp_path = path
                break

    if llama_cpp_path is None or not llama_cpp_path.exists():
        raise FileNotFoundError(
            "llama.cpp not found. Please install it or specify llama_cpp_path. "
            "Install with: git clone https://github.com/ggerganov/llama.cpp"
        )

    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Try alternative name
        convert_script = llama_cpp_path / "convert-hf-to-gguf.py"
        if not convert_script.exists():
            raise FileNotFoundError(
                f"Convert script not found in {llama_cpp_path}. "
                "Make sure llama.cpp is up to date."
            )

    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Step 1: Convert to GGUF (F16)
        f16_path = temp_path / "model-f16.gguf"
        logger.info(f"Converting to GGUF F16: {model_path}")

        result = subprocess.run(
            [
                "python",
                str(convert_script),
                str(model_path),
                "--outfile",
                str(f16_path),
                "--outtype",
                "f16",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"GGUF conversion failed: {result.stderr}")

        logger.info("F16 conversion complete")

        # Step 2: Quantize if not F16
        if quantization.lower() == "f16":
            # Just copy the F16 file
            import shutil
            shutil.copy(f16_path, output_path)
        else:
            # Quantize
            quantize_bin = llama_cpp_path / "quantize"
            if not quantize_bin.exists():
                quantize_bin = llama_cpp_path / "build" / "bin" / "quantize"

            if not quantize_bin.exists():
                raise FileNotFoundError(
                    f"Quantize binary not found. Build llama.cpp first: "
                    f"cd {llama_cpp_path} && make"
                )

            logger.info(f"Quantizing to {quantization}...")

            result = subprocess.run(
                [
                    str(quantize_bin),
                    str(f16_path),
                    str(output_path),
                    quantization.upper(),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Quantization failed: {result.stderr}")

    logger.info(f"GGUF export complete: {output_path}")
    return output_path


def export_for_ollama(
    model_path: str | Path,
    model_name: str,
    output_dir: str | Path,
    quantization: str = "q4_k_m",
    system_prompt: Optional[str] = None,
) -> Path:
    """Export model for use with Ollama.

    Creates a Modelfile and GGUF for importing into Ollama.

    Args:
        model_path: Path to HuggingFace model
        model_name: Name for the Ollama model
        output_dir: Directory for output files
        quantization: Quantization type
        system_prompt: Custom system prompt

    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to GGUF
    gguf_path = output_dir / f"{model_name}.gguf"
    export_to_gguf(model_path, gguf_path, quantization)

    # Create Modelfile
    default_system = (
        "You are a professional translator specializing in WordPress content. "
        "Translate accurately while preserving placeholders and HTML tags."
    )

    modelfile_content = f"""FROM {gguf_path.name}

SYSTEM {system_prompt or default_system}

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 256
"""

    modelfile_path = output_dir / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    # Create import instructions
    instructions = f"""# Ollama Import Instructions

1. Navigate to this directory:
   cd {output_dir}

2. Create the Ollama model:
   ollama create {model_name} -f Modelfile

3. Test the model:
   ollama run {model_name} "Translate to Dutch: Hello, world!"

4. Use in your application:
   curl http://localhost:11434/api/generate -d '{{
     "model": "{model_name}",
     "prompt": "Translate to Dutch: Hello, world!"
   }}'
"""

    instructions_path = output_dir / "README.md"
    with open(instructions_path, "w") as f:
        f.write(instructions)

    logger.info(f"Ollama export complete. See {instructions_path} for instructions.")
    return output_dir


def get_recommended_quantization(
    target_vram_gb: float = 10.0,
    model_params_b: float = 7.0,
) -> str:
    """Get recommended quantization based on target VRAM.

    Args:
        target_vram_gb: Target VRAM in GB
        model_params_b: Model parameters in billions

    Returns:
        Recommended quantization type
    """
    # Rough estimates of VRAM usage per billion parameters
    # These are approximate and depend on context length
    vram_per_param = {
        "f16": 2.0,  # 2 bytes per param
        "q8_0": 1.0,  # 1 byte per param
        "q5_k_m": 0.65,  # ~0.65 bytes per param
        "q4_k_m": 0.5,  # ~0.5 bytes per param
        "q3_k_m": 0.4,  # ~0.4 bytes per param
        "q2_k": 0.3,  # ~0.3 bytes per param
    }

    # Calculate estimated VRAM for each quantization
    for quant, bytes_per_param in vram_per_param.items():
        estimated_vram = model_params_b * bytes_per_param
        if estimated_vram <= target_vram_gb * 0.8:  # 20% buffer
            return quant

    # If nothing fits, return lowest quantization
    return "q2_k"
