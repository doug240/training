import os
import json
import logging
import subprocess
import shutil
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HF cache dirs setup (adjust paths as needed)
os.environ["HF_HOME"] = os.path.normpath(r"E:\HF_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.normpath(r"E:\HF_cache\datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.normpath(r"E:\HF_cache\models")
os.environ["HF_MODULES_CACHE"] = os.path.normpath(r"E:\HF_cache\modules")

# PyTorch CUDA memory config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

# Constants - adjust as needed
TOKENIZED_CACHE_DIR = os.path.normpath(r"E:\tokenized_cache")
CHECKPOINTS_DIR = os.path.normpath(r"E:\checkpoints")
CLEANED_DATA_DIR = os.path.normpath(r"E:\cleaned_datasets")
SAVE_DIR = os.path.normpath(r"E:\preprocessed")

for d in [TOKENIZED_CACHE_DIR, CHECKPOINTS_DIR, CLEANED_DATA_DIR, SAVE_DIR]:
    os.makedirs(d, exist_ok=True)

# Paths for model snapshots - adjust base_model_dir for your environment
base_model_dir = os.path.normpath(r"C:\Users\New User\.cache\huggingface\hub\models--bigcode--starcoder2-3b\snapshots")

def find_latest_snapshot_dir(base_dir):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base snapshots directory does not exist: {base_dir}")
    snapshots = [
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found inside {base_dir}")
    snapshots.sort(key=os.path.getctime, reverse=True)
    return snapshots[0]

local_model_path = find_latest_snapshot_dir(base_model_dir)

required_files = ["config.json", "tokenizer_config.json", "tokenizer.json"]
missing_files = [f for f in required_files if not os.path.isfile(os.path.join(local_model_path, f))]
if missing_files:
    raise FileNotFoundError(f"Missing model files in {local_model_path}: {missing_files}")

logger.info(f"Loading tokenizer and model from local path: {local_model_path}")

tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    local_files_only=True,
    quantization_config=quantization_config,
    device_map="auto",
    use_cache=False,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.warning(f"pad_token not set, using eos_token as pad_token: {tokenizer.pad_token}")

# 8-bit model is already on device, no need for .to(device)
# Disable gradient checkpointing when using 8-bit
# model.gradient_checkpointing_enable()  # removed/commented out

model.config.use_cache = False

if torch.cuda.is_available():
    torch.cuda.empty_cache()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def get_optimal_batch_size():
    try:
        output = subprocess.check_output("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits", shell=True)
        vram = int(output.decode().strip().split('\n')[0])
        logger.info(f"Detected GPU VRAM: {vram} MB")
        if vram >= 24576:
            return 8, 1
        elif vram >= 16384:
            return 6, 2
        elif vram >= 12288:
            return 4, 2
        else:
            return 2, 4
    except Exception as e:
        logger.warning(f"Failed to get VRAM info, using fallback batch sizes: {e}")
        return 2, 4

batch_size, gradient_accumulation_steps = get_optimal_batch_size()
logger.info(f"Using batch size {batch_size} and gradient accumulation steps {gradient_accumulation_steps}")

def build_text(example):
    if "text" in example and isinstance(example["text"], str):
        return example["text"]
    elif "question" in example and "context" in example:
        return f"{example['question']} {example['context']}"
    else:
        return " ".join(str(v) for v in example.values() if isinstance(v, str))

def clean_dataset(dataset_dict: DatasetDict) -> DatasetDict:
    cleaned = {}
    for split in dataset_dict.keys():
        dataset = dataset_dict[split]
        valid_examples = []

        for example in dataset:
            try:
                text = build_text(example)
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    valid_examples.append({"text": text})
            except Exception as e:
                logger.warning(f"Malformed sample skipped: {e}")

        cleaned_dataset = Dataset.from_list(valid_examples)
        cleaned[split] = cleaned_dataset

    return DatasetDict(cleaned)

def safe_save_dataset(dataset: DatasetDict, target_path: str):
    temp_path = target_path + "_temp"
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    if os.path.exists(target_path):
        backup_path = target_path + "_backup"
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.move(target_path, backup_path)

    try:
        dataset.save_to_disk(temp_path)
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.move(temp_path, target_path)

        backup_path = target_path + "_backup"
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        logger.info(f"Dataset saved successfully to {target_path}")
    except Exception as e:
        logger.error(f"Failed to save dataset to {target_path}: {e}")
        backup_path = target_path + "_backup"
        if os.path.exists(backup_path):
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.move(backup_path, target_path)
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        raise

def tokenize_and_cache_dataset(dataset: DatasetDict, dataset_name: str, tokenizer) -> DatasetDict:
    logger.info(f"Tokenizing and caching dataset: {dataset_name}")

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=2048,
        )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )
    tokenized_path = os.path.join(TOKENIZED_CACHE_DIR, dataset_name)
    os.makedirs(tokenized_path, exist_ok=True)
    tokenized_dataset.save_to_disk(tokenized_path)
    return tokenized_dataset

def load_and_cache_tokenized(dataset_name, dataset, tokenizer):
    tokenized_path = os.path.join(TOKENIZED_CACHE_DIR, dataset_name)
    if os.path.exists(tokenized_path):
        logger.info(f"Loading cached tokenized dataset for {dataset_name} from {tokenized_path}")
        return load_from_disk(tokenized_path)

    logger.warning(f"No cached tokenized dataset found for {dataset_name}. Proceeding to tokenize.")
    return tokenize_and_cache_dataset(dataset, dataset_name, tokenizer)

def find_cleaned_dataset_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Cleaned datasets base directory does not exist: {base_dir}")

    logger.info(f"Searching for cleaned dataset directory inside {base_dir}...")

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            for root, dirs, files in os.walk(full_path):
                if any(f.endswith(".arrow") for f in files):
                    logger.info(f"Found cleaned dataset directory: {full_path}")
                    return full_path
    raise FileNotFoundError(f"No valid cleaned dataset directory with .arrow files found in {base_dir}")
