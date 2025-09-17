"""
Ollama local LLM implementation for uncertainty quantification.
"""

import json
import time
import logging
from typing import List, Optional, Dict, Any, Union
import requests
from urllib.parse import urljoin

from .base import BaseLLM, LLMUsageStats

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """
    Ollama local LLM wrapper for running models locally.

    Supports running local models through Ollama API with:
    - Local model management
    - Streaming and non-streaming generation
    - Batch processing
    - Model information retrieval
    """

    def __init__(self,
                 model: str = "llama2",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 max_retries: int = 3,
                 timeout: int = 120,
                 system_prompt: Optional[str] = None):
        """
        Initialize Ollama LLM.

        Args:
            model: Ollama model name (e.g., 'llama2', 'mistral', 'codellama')
            base_url: Ollama server base URL
            temperature: Generation temperature (0.0-2.0)
            max_retries: Maximum retries for failed requests
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt for the model
        """
        super().__init__(model, temperature, max_retries)

        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.system_prompt = system_prompt or "You are a helpful assistant that provides clear, step-by-step mathematical reasoning."

        # Ollama API endpoints
        self.generate_url = urljoin(self.base_url + '/', 'api/generate')
        self.chat_url = urljoin(self.base_url + '/', 'api/chat')
        self.tags_url = urljoin(self.base_url + '/', 'api/tags')
        self.show_url = urljoin(self.base_url + '/', 'api/show')
        self.pull_url = urljoin(self.base_url + '/', 'api/pull')

        # Model information cache
        self._model_info_cache = None

        # Verify connection and model availability
        self._verify_connection()
        self._ensure_model_available()

    def _verify_connection(self):
        """Verify connection to Ollama server."""
        try:
            response = requests.get(
                self.tags_url,
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama server at {self.base_url}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to Ollama server at {self.base_url}. "
                f"Make sure Ollama is running. Error: {e}"
            )

    def _ensure_model_available(self):
        """Ensure the specified model is available, pull if necessary."""
        try:
            # Check if model exists
            available_models = self.list_available_models()
            model_names = [m['name'] for m in available_models]

            if self.model not in model_names:
                logger.info(f"Model {self.model} not found locally. Attempting to pull...")
                self._pull_model(self.model)
            else:
                logger.info(f"Model {self.model} is available")

        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")

    def _pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama library.

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model {model_name}...")

            response = requests.post(
                self.pull_url,
                json={"name": model_name},
                timeout=600,  # Longer timeout for model pulling
                stream=True
            )
            response.raise_for_status()

            # Stream the pull progress
            for line in response.iter_lines():
                if line:
                    try:
                        status = json.loads(line.decode('utf-8'))
                        if 'status' in status:
                            logger.info(f"Pull status: {status['status']}")
                    except json.JSONDecodeError:
                        pass

            logger.info(f"Successfully pulled model {model_name}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False

    def generate(self, prompt: str, max_tokens: int = 150, temperature: Optional[float] = None) -> str:
        """
        Generate text using Ollama API.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature

        Returns:
            Generated text response
        """
        temp = temperature if temperature is not None else self.temperature

        for attempt in range(self.max_retries):
            try:
                # Prepare the request payload
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "num_predict": max_tokens,
                        "top_p": 0.9,
                        "stop": []
                    }
                }

                # Add system prompt if using chat format
                if self.system_prompt:
                    payload["system"] = self.system_prompt

                start_time = time.time()

                # Make request to Ollama
                response = requests.post(
                    self.generate_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Parse response
                result = response.json()
                generated_text = result.get('response', '').strip()

                # Update usage statistics
                generation_time = time.time() - start_time
                self._update_usage_stats(prompt, generated_text, generation_time, result)

                return generated_text

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    return "Error: Request timed out after multiple attempts."

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return f"Error: Request failed after {self.max_retries} attempts: {e}"

            except Exception as e:
                logger.error(f"Unexpected error during generation: {e}")
                return f"Error: Unexpected error occurred: {e}"

            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    def batch_generate(self, prompts: List[str], max_tokens: int = 150,
                      temperature: Optional[float] = None) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Override default temperature

        Returns:
            List of generated text responses
        """
        results = []

        for i, prompt in enumerate(prompts):
            try:
                logger.info(f"Generating response {i+1}/{len(prompts)}")
                result = self.generate(prompt, max_tokens, temperature)
                results.append(result)

                # Small delay between requests to avoid overwhelming local server
                if i < len(prompts) - 1:
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error generating response for prompt {i+1}: {e}")
                results.append(f"Error: Failed to generate response: {e}")

        return results

    def _update_usage_stats(self, prompt: str, response: str, generation_time: float, ollama_response: Dict):
        """Update usage statistics based on Ollama response."""
        # Estimate token counts (Ollama doesn't always provide exact counts)
        prompt_tokens = len(prompt.split()) * 1.3  # Rough estimation
        response_tokens = len(response.split()) * 1.3

        # Update from Ollama response if available
        if 'eval_count' in ollama_response:
            response_tokens = ollama_response['eval_count']
        if 'prompt_eval_count' in ollama_response:
            prompt_tokens = ollama_response['prompt_eval_count']

        self.usage_stats.total_requests += 1
        self.usage_stats.input_tokens += int(prompt_tokens)
        self.usage_stats.output_tokens += int(response_tokens)
        self.usage_stats.total_tokens += int(prompt_tokens + response_tokens)

    def estimate_cost(self) -> float:
        """
        Estimate cost for Ollama models (free for local inference).

        Returns estimated compute cost based on token usage and generation time.
        """
        # Local models are free, but we can estimate compute cost
        # Based on approximate energy consumption and processing time

        # Rough estimate: $0.0001 per 1000 tokens for local compute
        estimated_cost = (self.usage_stats.total_tokens / 1000) * 0.0001
        self.usage_stats.estimated_cost_usd = estimated_cost

        return estimated_cost

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models in Ollama."""
        try:
            response = requests.get(self.tags_url, timeout=10)
            response.raise_for_status()
            return response.json().get('models', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_model_details(self) -> Dict[str, Any]:
        """Get detailed information about the current model."""
        if self._model_info_cache:
            return self._model_info_cache

        try:
            response = requests.post(
                self.show_url,
                json={"name": self.model},
                timeout=10
            )
            response.raise_for_status()
            self._model_info_cache = response.json()
            return self._model_info_cache

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model details: {e}")
            return {}

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().get_model_info()

        ollama_info = {
            "provider": "Ollama",
            "base_url": self.base_url,
            "timeout": self.timeout,
            "system_prompt": self.system_prompt,
        }

        # Add model details if available
        model_details = self.get_model_details()
        if model_details:
            ollama_info.update({
                "model_details": {
                    "family": model_details.get('details', {}).get('family', 'unknown'),
                    "format": model_details.get('details', {}).get('format', 'unknown'),
                    "size": model_details.get('size', 0),
                    "quantization": model_details.get('details', {}).get('quantization_level', 'unknown')
                }
            })

        # Add available models
        available_models = self.list_available_models()
        ollama_info["available_models"] = [m.get('name', 'unknown') for m in available_models]

        info.update(ollama_info)
        return info

    def health_check(self) -> Dict[str, Any]:
        """Check the health status of Ollama connection and model."""
        health_status = {
            "server_connected": False,
            "model_available": False,
            "server_url": self.base_url,
            "model_name": self.model,
            "timestamp": time.time()
        }

        try:
            # Check server connection
            response = requests.get(self.tags_url, timeout=5)
            response.raise_for_status()
            health_status["server_connected"] = True

            # Check model availability
            available_models = self.list_available_models()
            model_names = [m['name'] for m in available_models]
            health_status["model_available"] = self.model in model_names
            health_status["available_models"] = model_names

        except Exception as e:
            health_status["error"] = str(e)

        return health_status

    def stream_generate(self, prompt: str, max_tokens: int = 150,
                       temperature: Optional[float] = None):
        """
        Generate text with streaming response (generator).

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature

        Yields:
            Partial response chunks as they arrive
        """
        temp = temperature if temperature is not None else self.temperature

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temp,
                "num_predict": max_tokens,
                "top_p": 0.9,
            }
        }

        if self.system_prompt:
            payload["system"] = self.system_prompt

        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'response' in chunk:
                            text = chunk['response']
                            full_response += text
                            yield text

                        if chunk.get('done', False):
                            # Update usage stats when done
                            self._update_usage_stats(prompt, full_response, 0, chunk)
                            break

                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.RequestException as e:
            yield f"Error: Streaming failed: {e}"


# Utility functions for Ollama model management
def list_popular_ollama_models() -> Dict[str, Dict[str, Any]]:
    """List popular Ollama models for different use cases."""
    return {
        "code_models": {
            "codellama": {"size": "7B", "use_case": "Code generation and explanation"},
            "codellama:13b": {"size": "13B", "use_case": "Better code quality, more memory"},
            "deepseek-coder": {"size": "6.7B", "use_case": "Advanced code understanding"},
            "starcoder": {"size": "15.5B", "use_case": "Multi-language code generation"}
        },
        "chat_models": {
            "llama2": {"size": "7B", "use_case": "General conversation and reasoning"},
            "llama2:13b": {"size": "13B", "use_case": "Better reasoning, more memory"},
            "mistral": {"size": "7B", "use_case": "Fast, efficient general purpose"},
            "mixtral": {"size": "8x7B", "use_case": "High quality, mixture of experts"},
            "neural-chat": {"size": "7B", "use_case": "Optimized for conversations"}
        },
        "lightweight_models": {
            "tinyllama": {"size": "1.1B", "use_case": "Very fast, basic capabilities"},
            "orca-mini": {"size": "3B", "use_case": "Small but capable"},
            "phi": {"size": "2.7B", "use_case": "Microsoft's efficient model"}
        },
        "specialized_models": {
            "llava": {"size": "7B", "use_case": "Vision and language understanding"},
            "wizardmath": {"size": "7B", "use_case": "Mathematical problem solving"},
            "meditron": {"size": "7B", "use_case": "Medical and healthcare"},
        }
    }


def get_ollama_model_requirements(model_name: str) -> Dict[str, Union[str, float]]:
    """
    Get system requirements for Ollama models.

    Args:
        model_name: Ollama model name

    Returns:
        Dictionary with system requirements
    """
    # Common model size mappings
    size_mappings = {
        "1.1b": 1.1, "3b": 3, "6.7b": 6.7, "7b": 7,
        "13b": 13, "15.5b": 15.5, "34b": 34, "70b": 70
    }

    # Extract size from model name
    model_size = 7  # Default
    model_lower = model_name.lower()

    for size_str, size_val in size_mappings.items():
        if size_str in model_lower:
            model_size = size_val
            break

    # Estimate requirements based on model size
    ram_gb = max(8, model_size * 1.2)  # At least 8GB, scale with model
    disk_gb = model_size * 0.8  # Rough disk space estimate

    return {
        "model_size_b": model_size,
        "minimum_ram_gb": ram_gb,
        "recommended_ram_gb": ram_gb * 1.5,
        "disk_space_gb": disk_gb,
        "cpu_cores": 4 if model_size < 13 else 8,
        "gpu_recommended": model_size > 7,
        "gpu_memory_gb": max(6, model_size * 0.6) if model_size > 7 else None
    }