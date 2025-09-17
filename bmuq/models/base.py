"""
Base LLM interface for uncertainty quantification methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
from dataclasses import dataclass


@dataclass
class LLMUsageStats:
    """Statistics for LLM usage tracking."""
    total_requests: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class BaseLLM(ABC):
    """Base class for all LLM implementations."""

    def __init__(self, model: str, temperature: float = 0.7, max_retries: int = 3):
        """
        Initialize the LLM.

        Args:
            model: Model identifier
            temperature: Generation temperature (0.0-2.0)
            max_retries: Maximum number of retries for failed requests
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.usage_stats = LLMUsageStats()

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 150, temperature: Optional[float] = None) -> str:
        """
        Generate text response from the model.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str], max_tokens: int = 150, 
                      temperature: Optional[float] = None) -> List[str]:
        """
        Generate responses for multiple prompts efficiently.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Override default temperature

        Returns:
            List of generated text responses
        """
        pass

    def get_usage_stats(self) -> LLMUsageStats:
        """Get current usage statistics."""
        return self.usage_stats

    def reset_usage_stats(self):
        """Reset usage statistics to zero."""
        self.usage_stats = LLMUsageStats()

    @abstractmethod
    def estimate_cost(self) -> float:
        """Estimate cost in USD based on current usage."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "usage_stats": self.usage_stats
        }