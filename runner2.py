import os
import gc
import logging
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
)

logger = logging.getLogger(__name__)

def clear_cuda_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

def log_gpu_memory(stage="", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"[{stage}] CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        logger.info(f"[{stage}] CUDA memory reserved:  {torch.cuda.memory_reserved() / 1e6:.2f} MB")
        logger.debug(torch.cuda.memory_summary(device=device.index, abbreviated=True))

def load_8bit_model(model_name_or_path):
    import bitsandbytes as bnb  # must be installed
    logger.info(f"Loading model {model_name_or_path} in 8-bit mode...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return model

def get_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)

def get_optimal_batch_size():
    # Adjust depending on GPU memory
    return 4, 2  # batch size, grad accumulation steps

def prepare_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir,
    batch_size,
    grad_accum_steps,
    max_epochs,
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=max_epochs,
        save_total_limit=3,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=False,
        push_to_hub=False,
        report_to="none",
    )

    data_collator = get_data_collator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    return trainer
