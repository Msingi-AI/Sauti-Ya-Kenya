import logging
from datasets import Audio, load_dataset

from .security import is_remote_code_trusted

logger = logging.getLogger(__name__)


def get_waxal_swahili(
    split: str = "train",
    streaming: bool = True,
    dataset_name: str = "google/WaxalNLP",
    config_name: str = "swa_tts",
):
    """Load the Swahili TTS subset from WAXAL."""
    logger.info("Loading WAXAL split=%s from %s/%s", split, dataset_name, config_name)

    try:
        trust_remote_code = is_remote_code_trusted(dataset_name)
        ds = load_dataset(
            dataset_name,
            config_name,
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds
    except Exception as exc:
        logger.error("Failed to load WAXAL: %s", exc)
        raise


def save_to_disk(ds, output_path: str):
    logger.info("Saving dataset to %s...", output_path)
    ds.save_to_disk(output_path)
    logger.info("Save complete.")
