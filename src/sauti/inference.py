"""Inference helpers for Sauti.

This module contains a very small `synthesize` function that demonstrates
how to load a saved model and produce waveform output. Replace the internals
with your chosen TTS model + vocoder combination.
"""
import os
import logging

logger = logging.getLogger(__name__)


def synthesize(text: str, model_dir: str, out_filepath: str = "out.wav"):
    """Placeholder synthesis function.

    Args:
        text: input text to synthesize.
        model_dir: path to the fine-tuned or distilled model.
        out_filepath: WAV file to write.

    Returns:
        path to written WAV file (string).
    """
    logger.info("Synthesizing text with model at %s", model_dir)
    # TODO: load model, run text->mel, run vocoder
    # This placeholder writes an empty file to indicate the output location.
    open(out_filepath, "wb").close()
    logger.info("Wrote placeholder output to %s", out_filepath)
    return out_filepath
