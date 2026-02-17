"""Dataset preparation utilities for Sauti.

This module provides minimal helpers to prepare the WAXAL dataset (local or Hugging Face).
"""
import os
import logging

logger = logging.getLogger(__name__)


def prepare_waxal_dataset(hf_repo: str = None, local_path: str = None, out_dir: str = "data/processed"):
    """Prepare WAXAL data for training.

    Args:
        hf_repo: Optional Hugging Face dataset repo id (e.g. 'namespace/waxal').
        local_path: Optional local path to WAXAL archive or unpacked folder.
        out_dir: Destination directory for processed dataset.

    Returns:
        path to processed dataset directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    try:
        from datasets import load_dataset
    except Exception:
        logger.warning("datasets not installed; install requirements-sauti.txt to use dataset helpers")
        return out_dir

    if hf_repo:
        logger.info("Loading WAXAL from Hugging Face repo: %s", hf_repo)
        ds = load_dataset(hf_repo)
        try:
            from .data import download_and_filter_waxal  # local helper (below) if present
        except Exception:
            download_and_filter_waxal = None

        if download_and_filter_waxal:
            return download_and_filter_waxal(ds, out_dir=out_dir)

        # Fallback: save full dataset to disk
        ds.save_to_disk(out_dir)
        logger.info("Saved HF dataset to %s", out_dir)
        return out_dir

    if local_path:
        logger.info("Preparing WAXAL from local path: %s", local_path)
        # Implement local conversion steps (placeholder)
        # For now, we just copy or note the path
        return local_path

    logger.error("No hf_repo or local_path provided; nothing to prepare")
    return out_dir


def download_and_filter_waxal(ds, out_dir: str = "data/processed", language_terms=("swahili", "sw", "swa")):
    """Filter a loaded WAXAL dataset for Swahili-language examples and save to disk.

    Args:
        ds: a `datasets.Dataset` or `datasets.DatasetDict` returned by `load_dataset`.
        out_dir: destination directory to save the filtered dataset.
        language_terms: iterable of substrings to match language metadata.

    Returns:
        path to saved dataset directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Helper to test fields for swahili
    def _is_swahili(example):
        # Check common metadata fields
        candidates = [
            example.get(k) for k in ("language", "language_code", "lang", "locale", "language_short") if isinstance(example.get(k), str)
        ]
        for v in candidates:
            lv = v.lower()
            for t in language_terms:
                if t in lv:
                    return True

        # As a last resort, check text/transcript fields for Swahili tokens
        for key in ("text", "transcript", "utterance"):
            if key in example and isinstance(example[key], str):
                txt = example[key].lower()
                for t in (" karibu ", "asante", "habari", "salaam", "rafiki"):
                    if t.strip() in txt:
                        return True

        return False

    try:
        from datasets import Dataset, DatasetDict
    except Exception:
        logger.warning("datasets not available; cannot filter WAXAL programmatically")
        return out_dir

    if isinstance(ds, dict) or hasattr(ds, "items"):
        # DatasetDict-like
        filtered = {}
        for split, dset in ds.items():
            try:
                f = dset.filter(_is_swahili)
                filtered[split] = f
                logger.info("Filtered split %s -> %d examples", split, len(f))
            except Exception as e:
                logger.warning("Could not filter split %s: %s", split, e)
                filtered[split] = dset
        from datasets import DatasetDict
        out = DatasetDict(filtered)
        out.save_to_disk(out_dir)
        logger.info("Saved filtered WAXAL dataset to %s", out_dir)
        return out_dir

    if isinstance(ds, Dataset):
        f = ds.filter(_is_swahili)
        f.save_to_disk(out_dir)
        logger.info("Saved filtered WAXAL dataset to %s (count=%d)", out_dir, len(f))
        return out_dir

    logger.warning("Unknown dataset type; saving original to disk")
    try:
        ds.save_to_disk(out_dir)
    except Exception:
        logger.exception("Failed to save dataset to %s", out_dir)
    return out_dir
