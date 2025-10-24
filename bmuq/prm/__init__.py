"""
Process Reward Models (PRM) module for training and using PRMs in uncertainty quantification.

This module provides functionality for training BERT-like models as Process Reward Models
that can evaluate reasoning steps as positive, neutral, or negative.
"""

from .featurizer import PRMFeaturizer
from .train import PRMTrainer

__all__ = [
    "PRMFeaturizer",
    "PRMTrainer",
]
