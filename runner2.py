import os
import torch
from transformers import Trainer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from typing import Dict, Optional
import logging

from core import tokenizer, device, MODEL_NAME_OR_PATH, get_quantization_config
from runner3 import clear_cuda_memory, log_cuda_memory, save_progress, load_progress

logger = logging.getLogger(__name__)

def train_on_dataset(dataset_name: str, tokenized_dataset: dict, progress: Dict, model_path: Optional[str] = None) -> None:
    """
    Train the model on a given tokenized dataset.

    Args:
        dataset_name (str): Name of the dataset for logging and saving.
        tokenized_dataset (dict): Tokenized dataset inputs.
        progress (dict): Training progress dictionary to update.
        model_path (str, optional): Path or name of the pretrained model.
    """
    model_path = model_path or MODEL_NAME_OR_PATH
    clear_cuda_memory()
    log_cuda_memory()

    try:
        quant_config = get_quantization_config()
        logger.info(f"Loading model {model_path} with 8-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)

        trainer = Trainer(
            model=model,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            # Add any other Trainer args here as needed.
        )
        
        logger.info(f"Starting training on dataset: {dataset_name}")
        trainer.train()
        logger.info(f"Training complete on dataset: {dataset_name}")

        progress[dataset_name] = "trained"
        save_progress(progress)

        # Save the fine-tuned model checkpoint
        save_dir = os.path.join("models", dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Model checkpoint saved to {save_dir}")

    except Exception as e:
        logger.error(f"Training failed for dataset {dataset_name}: {e}", exc_info=True)
    finally:
        clear_cuda_memory()
