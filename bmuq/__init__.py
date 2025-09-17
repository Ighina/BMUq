"""
BMUq - Bayesian Methods for Uncertainty Quantification in Large Language Models

A modular framework for implementing and evaluating uncertainty quantification methods
in LLM-based reasoning, with a focus on SelfCheck and tree search approaches.
"""

__version__ = "0.1.0"
__author__ = "BMUq Contributors"

from .core.data_structures import ReasoningStep, ReasoningPath
from .uncertainty.selfcheck import SelfCheck
from .search.tree_search import TreeSearchCoT
from .models.base import BaseLLM
from .models.openai_llm import OpenAILLM

# Conditionally import HuggingFace if available
try:
    from .models.huggingface_llm import HuggingFaceLLM
    _HUGGINGFACE_AVAILABLE = True
except ImportError:
    _HUGGINGFACE_AVAILABLE = False
    HuggingFaceLLM = None

__all__ = [
    "ReasoningStep",
    "ReasoningPath", 
    "SelfCheck",
    "TreeSearchCoT",
    "BaseLLM",
    "OpenAILLM"
]

if _HUGGINGFACE_AVAILABLE:
    __all__.append("HuggingFaceLLM")