import logging
from core import load_and_cache_tokenized, MODEL_NAME_OR_PATH
from runner2 import train_on_dataset
from runner3 import load_progress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    dataset_name = "cleaned_dataset"
    progress = load_progress()

    if progress.get(dataset_name) == "trained":
        logger.info(f"Dataset '{dataset_name}' already trained. Skipping.")
        return

    try:
        tokenized_dataset = load_and_cache_tokenized(dataset_name)
        train_on_dataset(dataset_name, tokenized_dataset, progress, MODEL_NAME_OR_PATH)
    except Exception as e:
        logger.error(f"Failed to train on dataset '{dataset_name}': {e}", exc_info=True)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
