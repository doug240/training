import os
import logging
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME_OR_PATH = "bigcode/starcoder"
DATASET_BASE_PATH = r"E:\cleaned_datasets\openassistant_oasst1"
TOKENIZED_CACHE_DIR = "tokenized_cache"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize tokenizer once globally
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

def get_quantization_config():
    """Returns quantization config for 8-bit loading."""
    return BitsAndBytesConfig(load_in_8bit=True)

def load_and_cache_tokenized(dataset_name: str):
    """
    Loads dataset from local Arrow/JSON files using HF datasets, tokenizes it with the tokenizer,
    caches the tokenized dataset to disk, and returns the tokenized dataset.

    Args:
        dataset_name (str): Name to use for caching tokenized dataset

    Returns:
        datasets.Dataset or datasets.DatasetDict: Tokenized dataset object
    """
    tokenized_cache_path = os.path.join(TOKENIZED_CACHE_DIR, dataset_name)

    # If tokenized dataset already cached on disk, load and return it from disk
    if os.path.exists(tokenized_cache_path):
        logger.info(f"Loading tokenized dataset from cache at {tokenized_cache_path}")
        return load_from_disk(tokenized_cache_path)  # <-- Correct way to load cached dataset

    try:
        logger.info(f"Loading raw dataset from {DATASET_BASE_PATH}")
        # Load dataset from local folder (Arrow/JSON) - returns DatasetDict or Dataset
        raw_dataset = load_dataset(DATASET_BASE_PATH)

        # Select 'train' split if DatasetDict
        if isinstance(raw_dataset, DatasetDict):
            dataset = raw_dataset['train']  # adjust if you want other splits
        else:
            dataset = raw_dataset

        logger.info("Tokenizing dataset...")

        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, max_length=512)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Make sure cache directory exists
        os.makedirs(tokenized_cache_path, exist_ok=True)

        logger.info(f"Saving tokenized dataset to {tokenized_cache_path}")
        tokenized_dataset.save_to_disk(tokenized_cache_path)

        return tokenized_dataset

    except Exception as e:
        logger.error(f"Failed to load and tokenize dataset from {DATASET_BASE_PATH}: {e}")
        raise RuntimeError(f"Failed to load and tokenize dataset from {DATASET_BASE_PATH}") from e

