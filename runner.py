# runner.py

import os
import json
import logging
import torch
from datasets import load_from_disk, DatasetDict

from transformers import AutoTokenizer

# Constants and config
MODEL_NAME_OR_PATH = r"C:\Users\New User\.cache\huggingface\hub\models--bigcode--starcoder2-3b\snapshots\733247c55e3f73af49ce8e9c7949bf14af205928"
CHECKPOINTS_DIR = r"./checkpoints"
SAVE_DIR = r"./saved_models"
PROGRESS_FILE = os.path.normpath("training_progress.json")
MAX_EPOCHS = 5

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_save_dataset(dataset, path):
    os.makedirs(path, exist_ok=True)
    dataset.save_to_disk(path)
    logger.info(f"Dataset safely saved to: {path}")


def load_and_cache_tokenized(dataset_name, dataset, tokenizer):
    tokenized_path = os.path.join("tokenized_cache", dataset_name)
    if os.path.exists(tokenized_path):
        logger.info(f"Loading tokenized dataset from cache: {tokenized_path}")
        from datasets import load_from_disk
        return load_from_disk(tokenized_path)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length")

    if not isinstance(dataset, DatasetDict):
        raise ValueError("Expected a DatasetDict with 'train' and 'validation' splits.")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    safe_save_dataset(tokenized_datasets, tokenized_path)
    return tokenized_datasets


def find_cleaned_dataset_dir(base_path):
    dirs = sorted(
        (os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))),
        key=os.path.getmtime,
        reverse=True
    )
    if not dirs:
        raise FileNotFoundError(f"No cleaned dataset directory found inside {base_path}")
    return dirs[0]


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


def main():
    progress = load_progress()
    dataset_base = os.path.normpath(r"E:\cleaned_datasets")

    try:
        dataset_dir = find_cleaned_dataset_dir(dataset_base)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    try:
        dataset = load_from_disk(dataset_dir)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Import training function from runner2
    from runner2 import train_on_dataset

    train_on_dataset("cleaned_dataset", dataset, progress, MODEL_NAME_OR_PATH)


if __name__ == "__main__":
    main()
