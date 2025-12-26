"""GPU utility functions."""

from dataclasses import dataclass
from typing import Optional

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    utilization_percent: float


def get_gpu_info() -> list[GPUInfo]:
    """Get information about available GPUs.

    Returns:
        List of GPUInfo objects for each available GPU
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA is not available")
            return []

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # Convert to GB

            # Get current memory usage
            torch.cuda.set_device(i)
            free_memory, total = torch.cuda.mem_get_info(i)
            free_memory_gb = free_memory / (1024**3)
            used_memory_gb = total_memory - free_memory_gb

            gpus.append(
                GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory_gb=round(total_memory, 2),
                    free_memory_gb=round(free_memory_gb, 2),
                    used_memory_gb=round(used_memory_gb, 2),
                    utilization_percent=round((used_memory_gb / total_memory) * 100, 1),
                )
            )

        return gpus

    except ImportError:
        logger.warning("PyTorch not installed, cannot get GPU info")
        return []
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return []


def check_gpu_memory(
    required_gb: float = 16.0,
    min_gpus: int = 1,
) -> tuple[bool, str]:
    """Check if sufficient GPU memory is available.

    Args:
        required_gb: Minimum required free memory in GB (total across all GPUs)
        min_gpus: Minimum number of GPUs required

    Returns:
        Tuple of (success, message)
    """
    gpus = get_gpu_info()

    if len(gpus) < min_gpus:
        return False, f"Need at least {min_gpus} GPUs, found {len(gpus)}"

    total_free = sum(gpu.free_memory_gb for gpu in gpus)

    if total_free < required_gb:
        return (
            False,
            f"Need {required_gb}GB free VRAM, have {total_free:.1f}GB across {len(gpus)} GPUs",
        )

    gpu_summary = ", ".join(
        f"GPU {g.index}: {g.name} ({g.free_memory_gb:.1f}GB free)" for g in gpus
    )
    return True, f"GPU check passed: {gpu_summary}"


def get_optimal_device_map(model_size_gb: float) -> Optional[dict]:
    """Get optimal device map for model loading.

    Args:
        model_size_gb: Estimated model size in GB

    Returns:
        Device map dict or None for auto placement
    """
    gpus = get_gpu_info()

    if not gpus:
        return {"": "cpu"}

    # For models that fit on a single GPU
    for gpu in gpus:
        if gpu.free_memory_gb >= model_size_gb * 1.2:  # 20% buffer
            return {"": f"cuda:{gpu.index}"}

    # For multi-GPU, use auto device map
    return "auto"


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cache cleared")
    except Exception as e:
        logger.warning(f"Could not clear GPU memory: {e}")
