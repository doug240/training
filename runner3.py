import os
import json
import logging
import torch

from runner2 import (
    clear_cuda_memory,
    log_gpu_memory,
    load_8bit_model,
    prepare_trainer,
    get_optimal_batch_size,
)

logger = logging.getLogger(__name__)

PROGRESS_FILE = "progress.json"
MAX_EPOCHS = 3  # or import from config

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

def train_on_dataset(dataset_name, tokenized_datasets, tokenizer, model_name_or_path, checkpoints_dir, save_dir):
    progress = load_progress()

    if "train" not in tokenized_datasets or "validation" not in tokenized_datasets:
        logger.warning(f"Dataset {dataset_name} missing train or validation split.")
        return

    epochs_done = progress.get(dataset_name, {}).get("epochs_completed", 0)
    if epochs_done >= MAX_EPOCHS:
        logger.info(f"Dataset {dataset_name} already trained. Skipping.")
        return

    clear_cuda_memory()
    log_gpu_memory("after clearing")

    batch_size, grad_accum_steps = get_optimal_batch_size()

    try:
        model = load_8bit_model(model_name_or_path)
    except Exception as e:
        logger.error(f"Failed to load model in 8-bit mode: {e}")
        return

    model.gradient_checkpointing_disable()
    model.config.use_cache = False

    output_dir = os.path.join(checkpoints_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    trainer = prepare_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        output_dir=output_dir,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_epochs=MAX_EPOCHS,
    )

    log_gpu_memory("before training")

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory during training.")
        clear_cuda_memory()
        return

    log_gpu_memory("after training")

    model_save_path = os.path.join(save_dir, dataset_name)
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Model saved to: {model_save_path}")

    progress[dataset_name] = {"epochs_completed": MAX_EPOCHS}
    save_progress(progress)

    clear_cuda_memory()
    log_gpu_memory("after save & clear")
