"""
Mock LLM implementation for testing and development.
"""

from typing import Dict, List, Optional, Any
import random
import time

from .base import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM for demonstration and testing purposes."""

    def __init__(self, model: str = "mock-llm", temperature: float = 0.7, max_retries: int = 3,
                 response_delay: float = 0.1):
        """
        Initialize Mock LLM.

        Args:
            model: Mock model identifier
            temperature: Generation temperature (ignored in mock)
            max_retries: Maximum retries (ignored in mock)
            response_delay: Artificial delay to simulate API latency
        """
        super().__init__(model, temperature, max_retries)
        self.response_delay = response_delay
        
        # Predefined responses for different prompt patterns
        self.step_responses = {
            "extract_target": "The step aims to solve for x by isolating the variable.",
            "collect_info": "Step 2",
            "regenerate": "Solving for x: 2x + 3 = 7, so 2x = 4, therefore x = 2",
            "compare": "supports",
            "quadratic": "Using the quadratic formula: x = (-b ± √(b²-4ac)) / 2a",
            "factoring": "Factor the quadratic: (x - 2)(x - 3) = 0, so x = 2 or x = 3",
            "default": "Step: Let x = 5. Then we can substitute this value."
        }

        # More sophisticated response patterns
        self.math_responses = [
            "First, we need to isolate the variable on one side of the equation.",
            "Let's subtract 3 from both sides to get 2x = 4.",
            "Dividing both sides by 2, we get x = 2.",
            "We can verify our answer by substituting back into the original equation.",
            "Therefore, the solution is x = 2."
        ]

        self.reasoning_responses = [
            "This follows from the previous step by applying algebraic manipulation.",
            "We use the distributive property to expand the expression.",
            "By the transitive property, if A = B and B = C, then A = C.",
            "This is a direct application of the substitution method.",
            "We apply the quadratic formula to find the roots."
        ]

    def generate(self, prompt: str, max_tokens: int = 150, temperature: Optional[float] = None) -> str:
        """
        Mock generation with pattern-based responses.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate (used for token counting)
            temperature: Temperature (affects randomness in mock responses)

        Returns:
            Generated mock response
        """
        # Simulate API delay
        time.sleep(self.response_delay)

        # Update usage statistics
        self.usage_stats.total_requests += 1
        
        # Estimate token usage (rough approximation)
        prompt_tokens = len(prompt.split()) + 10  # Add some overhead
        
        response = self._generate_response(prompt)
        response_tokens = len(response.split())
        
        self.usage_stats.input_tokens += prompt_tokens
        self.usage_stats.output_tokens += response_tokens
        self.usage_stats.total_tokens += prompt_tokens + response_tokens

        return response

    def _generate_response(self, prompt: str) -> str:
        """Generate response based on prompt patterns."""
        prompt_lower = prompt.lower()
        
        # Pattern matching for different types of prompts
        if "extract target" in prompt_lower or "what specific action" in prompt_lower:
            return self.step_responses["extract_target"]
        elif "which previous steps" in prompt_lower or "information" in prompt_lower:
            return self.step_responses["collect_info"]
        elif "achieve the target" in prompt_lower or "regenerate" in prompt_lower:
            return self.step_responses["regenerate"]
        elif "compare" in prompt_lower:
            comparison_results = ["supports", "contradicts", "not_directly_related"]
            # Use temperature to affect randomness
            if hasattr(self, 'temperature') and self.temperature > 0.8:
                return random.choice(comparison_results)
            else:
                return self.step_responses["compare"]
        elif "quadratic" in prompt_lower:
            if "formula" in prompt_lower:
                return self.step_responses["quadratic"]
            elif "factor" in prompt_lower:
                return self.step_responses["factoring"]
        elif any(keyword in prompt_lower for keyword in ["solve", "equation", "math"]):
            return random.choice(self.math_responses)
        elif any(keyword in prompt_lower for keyword in ["reasoning", "step", "logic"]):
            return random.choice(self.reasoning_responses)
        else:
            return self.step_responses["default"]

    def batch_generate(self, prompts: List[str], max_tokens: int = 150, 
                      temperature: Optional[float] = None) -> List[str]:
        """Generate responses for multiple prompts."""
        return [self.generate(prompt, max_tokens, temperature) for prompt in prompts]

    def estimate_cost(self) -> float:
        """Mock cost estimation (returns 0 for mock LLM)."""
        return 0.0

    def add_custom_response(self, pattern: str, response: str):
        """Add custom response pattern for testing."""
        self.step_responses[pattern] = response

    def set_response_delay(self, delay: float):
        """Set artificial response delay."""
        self.response_delay = delay

    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        info = super().get_model_info()
        info.update({
            "provider": "Mock",
            "response_delay": self.response_delay,
            "available_patterns": list(self.step_responses.keys())
        })
        return info