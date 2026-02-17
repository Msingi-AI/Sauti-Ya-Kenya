"""Sauti package
"""
__version__ = "0.1.0"

from .data import prepare_waxal_dataset
from .distill import Distiller
from .finetune import finetune_student
from .inference import synthesize

__all__ = [
    "__version__",
    "prepare_waxal_dataset",
    "Distiller",
    "finetune_student",
    "synthesize",
]
