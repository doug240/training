import torch
import logging
import json
from typing import Optional

logger = logging.getLogger(__name__)

def clear_cuda_memory() -> None:
    """
    Clear CUDA memory to prevent out-of-memory issues.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.debug("Cleared CUDA memory.")

def log_cuda_memory() -> None:
    """
    Log CUDA memory usage.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        logger.info(f"CUDA memory allocated: {allocated:.2f} MB")
        logger.info(f"CUDA memory reserved:  {reserved:.2f} MB")

def save_progress(progress: dict, filename: str = "progress.json") -> None:
    """
    Save training progress to a JSON file.

    Args:
        progress (dict): Dictionary holding progress info.
        filename (str): File path for saving progress.
    """
    with open(filename, "w") as f:
        json.dump(progress, f, indent=2)
    logger.info(f"Progress saved to {filename}")

def load_progress(filename: str = "progress.json") -> dict:
    """
    Load training progress from a JSON file.

    Args:
        filename (str): File path to load progress from.

    Returns:
        dict: Progress dictionary or empty if file missing.
    """
    try:
        with open(filename, "r") as f:
            progress = json.load(f)
        logger.info(f"Loaded progress from {filename}")
        return progress
    except FileNotFoundError:
        logger.info(f"No progress file found at {filename}, starting fresh.")
        return {}


    clear_cuda_memory()
    log_gpu_memory("after save & clear")
