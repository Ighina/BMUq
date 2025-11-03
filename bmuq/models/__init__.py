"""
LLM models and interfaces for uncertainty quantification.
"""

from .base import BaseLLM
from .openai_llm import OpenAILLM
from .mock_llm import MockLLM

try:
    from .huggingface_llm import HuggingFaceLLM, list_recommended_models, get_model_memory_requirements
    _HUGGINGFACE_AVAILABLE = True
except ImportError:
    _HUGGINGFACE_AVAILABLE = False
    HuggingFaceLLM = None
    list_recommended_models = None
    get_model_memory_requirements = None

try:
    from .ollama_llm import OllamaLLM, list_popular_ollama_models, get_ollama_model_requirements
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False
    OllamaLLM = None
    list_popular_ollama_models = None
    get_ollama_model_requirements = None

try:
    from .vllm_llm import VLLMLLM, get_vllm_supported_models, estimate_vllm_memory_requirements
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    VLLMLLM = None
    get_vllm_supported_models = None
    estimate_vllm_memory_requirements = None

__all__ = ["BaseLLM", "OpenAILLM", "MockLLM"]

if _HUGGINGFACE_AVAILABLE:
    __all__.extend(["HuggingFaceLLM", "list_recommended_models", "get_model_memory_requirements"])

if _OLLAMA_AVAILABLE:
    __all__.extend(["OllamaLLM", "list_popular_ollama_models", "get_ollama_model_requirements"])

if _VLLM_AVAILABLE:
    __all__.extend(["VLLMLLM", "get_vllm_supported_models", "estimate_vllm_memory_requirements"])