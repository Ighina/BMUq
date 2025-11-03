"""
vLLM implementation for high-performance inference with uncertainty quantification.
"""

import os
import logging
from typing import List, Optional, Dict, Any, Union
import torch

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

from .base import BaseLLM, LLMUsageStats

logger = logging.getLogger(__name__)


class VLLMLLM(BaseLLM):
    """
    vLLM implementation with high-performance inference and log probability support.

    Supports:
    - Fast batched inference with PagedAttention
    - GPU acceleration with optimal memory usage
    - Log probabilities (up to 100 top tokens per position)
    - Efficient KV cache management
    - Multiple sampling strategies
    """

    def __init__(self,
                 model: str = "meta-llama/Llama-2-7b-hf",
                 temperature: float = 0.7,
                 max_retries: int = 3,
                 max_model_len: Optional[int] = None,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 trust_remote_code: bool = False,
                 download_dir: Optional[str] = None,
                 top_logprobs: int = 5,
                 max_num_seqs: int = 256,
                 enforce_eager: bool = False,
                 dtype: str = "auto"):
        """
        Initialize vLLM model.

        Args:
            model: Model identifier (HuggingFace model name or local path)
            temperature: Generation temperature (0.0-2.0)
            max_retries: Maximum retries for failed requests
            max_model_len: Maximum model context length. If None, uses model's default
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            trust_remote_code: Whether to trust remote code for custom models
            download_dir: Directory to cache downloaded models
            top_logprobs: Number of top log probabilities to return (1-100)
            max_num_seqs: Maximum number of sequences to process in a batch
            enforce_eager: Whether to enforce eager execution (disable CUDA graphs)
            dtype: Data type for model weights ('auto', 'float16', 'bfloat16', 'float32')
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Please install it with: pip install vllm"
            )

        super().__init__(model, temperature, max_retries)

        self.model_name = model
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir or os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        self.top_logprobs = min(max(1, top_logprobs), 100)  # Clamp between 1 and 100
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.dtype = dtype

        # Initialize the model
        self.llm = None
        self._load_model()

        # Track generation statistics
        self.generation_stats = {
            'total_generations': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_sequences': 0,
            'gpu_memory_allocated': 0,
        }

    def _load_model(self):
        """Load the vLLM model with specified configuration."""
        try:
            logger.info(f"Loading vLLM model: {self.model_name}")

            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.warning("CUDA not available. vLLM requires GPU support.")
                raise RuntimeError("vLLM requires CUDA-capable GPU")

            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")

            # vLLM initialization parameters
            llm_kwargs = {
                'model': self.model_name,
                'tensor_parallel_size': self.tensor_parallel_size,
                'gpu_memory_utilization': self.gpu_memory_utilization,
                'trust_remote_code': self.trust_remote_code,
                'download_dir': self.download_dir,
                'max_num_seqs': self.max_num_seqs,
                'enforce_eager': self.enforce_eager,
                'dtype': self.dtype,
            }

            # Add max_model_len only if specified
            if self.max_model_len is not None:
                llm_kwargs['max_model_len'] = self.max_model_len

            # Initialize vLLM
            self.llm = LLM(**llm_kwargs)

            logger.info("vLLM model loaded successfully")

            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"CUDA memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        except Exception as e:
            logger.error(f"Failed to load vLLM model {self.model_name}: {e}")
            raise

    def _create_sampling_params(self, max_tokens: int = 150,
                                temperature: Optional[float] = None,
                                include_logprobs: bool = False) -> 'SamplingParams':
        """
        Create SamplingParams for vLLM generation.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature
            include_logprobs: Whether to include log probabilities

        Returns:
            SamplingParams object
        """
        temp = temperature if temperature is not None else self.temperature

        params = {
            'temperature': temp,
            'max_tokens': max_tokens,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
        }

        # Add logprobs if requested
        if include_logprobs:
            params['logprobs'] = self.top_logprobs

        return SamplingParams(**params)

    def generate(self, prompt: str, max_tokens: int = 150,
                temperature: Optional[float] = None,
                structured_output=None) -> str:
        """
        Generate text using vLLM.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature
            structured_output: A Pydantic class defining the format of the output (not implemented yet)

        Returns:
            Generated text response
        """
        if structured_output is not None:
            logger.warning("Structured output is not yet implemented for vLLM")

        if self.llm is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        try:
            # Format prompt
            formatted_prompt = f"You are a helpful assistant that provides clear, step-by-step mathematical reasoning.\n\nUser: {prompt}\n\nAssistant:"

            # Create sampling parameters
            sampling_params = self._create_sampling_params(max_tokens, temperature, include_logprobs=False)

            # Generate
            outputs = self.llm.generate([formatted_prompt], sampling_params)

            if not outputs:
                return "Error: No output generated"

            # Extract generated text
            output = outputs[0]
            generated_text = output.outputs[0].text.strip()

            # Update usage statistics
            prompt_tokens = len(output.prompt_token_ids)
            output_tokens = len(output.outputs[0].token_ids)

            self.usage_stats.total_requests += 1
            self.usage_stats.input_tokens += prompt_tokens
            self.usage_stats.output_tokens += output_tokens
            self.usage_stats.total_tokens += prompt_tokens + output_tokens

            self.generation_stats['total_generations'] += 1
            self.generation_stats['total_input_tokens'] += prompt_tokens
            self.generation_stats['total_output_tokens'] += output_tokens
            self.generation_stats['total_sequences'] += 1

            # Update GPU memory stats
            if torch.cuda.is_available():
                self.generation_stats['gpu_memory_allocated'] = torch.cuda.memory_allocated()

            return generated_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: Generation failed due to {type(e).__name__}: {str(e)}"

    def batch_generate(self, prompts: List[str], max_tokens: int = 150,
                      temperature: Optional[float] = None) -> List[str]:
        """
        Generate responses for multiple prompts efficiently using vLLM's batching.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Override default temperature

        Returns:
            List of generated text responses
        """
        if not prompts:
            return []

        if self.llm is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        try:
            # Format prompts
            formatted_prompts = [
                f"You are a helpful assistant that provides clear, step-by-step mathematical reasoning.\n\nUser: {prompt}\n\nAssistant:"
                for prompt in prompts
            ]

            # Create sampling parameters
            sampling_params = self._create_sampling_params(max_tokens, temperature, include_logprobs=False)

            # Generate in batch (vLLM handles this efficiently)
            outputs = self.llm.generate(formatted_prompts, sampling_params)

            # Extract results
            results = []
            for output in outputs:
                if output.outputs:
                    generated_text = output.outputs[0].text.strip()
                    results.append(generated_text)

                    # Update statistics
                    prompt_tokens = len(output.prompt_token_ids)
                    output_tokens = len(output.outputs[0].token_ids)

                    self.usage_stats.total_requests += 1
                    self.usage_stats.input_tokens += prompt_tokens
                    self.usage_stats.output_tokens += output_tokens
                    self.usage_stats.total_tokens += prompt_tokens + output_tokens
                else:
                    results.append("Error: No output generated")

            # Update generation stats
            self.generation_stats['total_generations'] += len(prompts)
            self.generation_stats['total_sequences'] += len(prompts)

            return results

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            # Fallback to individual generation
            return [self.generate(prompt, max_tokens, temperature) for prompt in prompts]

    def generate_with_log_probs(self, prompt: str, max_tokens: int = 150,
                                 temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate text and return log probabilities for each token.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature

        Returns:
            Dictionary containing:
                - 'text': Generated text
                - 'log_probs': List of log probabilities for each generated token
                - 'tokens': List of generated tokens (as strings)
                - 'top_log_probs': List of dicts with top-k tokens and their log probs
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        try:
            # Format prompt
            formatted_prompt = f"You are a helpful assistant that provides clear, step-by-step mathematical reasoning.\n\nUser: {prompt}\n\nAssistant:"

            # Create sampling parameters with logprobs enabled
            sampling_params = self._create_sampling_params(max_tokens, temperature, include_logprobs=True)

            # Generate
            outputs = self.llm.generate([formatted_prompt], sampling_params)

            if not outputs or not outputs[0].outputs:
                return {
                    'text': "Error: No output generated",
                    'log_probs': [],
                    'tokens': [],
                    'top_log_probs': []
                }

            output = outputs[0]
            generated_output = output.outputs[0]
            generated_text = generated_output.text.strip()

            # Extract log probabilities
            log_probs = []
            tokens = []
            top_log_probs = []

            if hasattr(generated_output, 'logprobs') and generated_output.logprobs:
                for token_logprobs in generated_output.logprobs:
                    if token_logprobs:
                        # Get the token with highest probability (the one that was generated)
                        # vLLM returns a dict of {token_id: Logprob} objects
                        sorted_logprobs = sorted(
                            token_logprobs.items(),
                            key=lambda x: x[1].logprob if hasattr(x[1], 'logprob') else x[1],
                            reverse=True
                        )

                        if sorted_logprobs:
                            # Get the selected token (highest probability)
                            top_token_id, top_logprob_obj = sorted_logprobs[0]

                            # Extract values based on object type
                            if hasattr(top_logprob_obj, 'logprob'):
                                token_log_prob = top_logprob_obj.logprob
                                token_text = top_logprob_obj.decoded_token if hasattr(top_logprob_obj, 'decoded_token') else str(top_token_id)
                            else:
                                token_log_prob = float(top_logprob_obj)
                                token_text = str(top_token_id)

                            log_probs.append(token_log_prob)
                            tokens.append(token_text)

                            # Get top-k tokens and their log probs
                            top_k_dict = {}
                            for token_id, logprob_obj in sorted_logprobs[:self.top_logprobs]:
                                if hasattr(logprob_obj, 'logprob'):
                                    lp = logprob_obj.logprob
                                    tok = logprob_obj.decoded_token if hasattr(logprob_obj, 'decoded_token') else str(token_id)
                                else:
                                    lp = float(logprob_obj)
                                    tok = str(token_id)
                                top_k_dict[tok] = lp

                            top_log_probs.append(top_k_dict)

            # Update usage statistics
            prompt_tokens = len(output.prompt_token_ids)
            output_tokens = len(generated_output.token_ids)

            self.usage_stats.total_requests += 1
            self.usage_stats.input_tokens += prompt_tokens
            self.usage_stats.output_tokens += output_tokens
            self.usage_stats.total_tokens += prompt_tokens + output_tokens

            self.generation_stats['total_generations'] += 1
            self.generation_stats['total_input_tokens'] += prompt_tokens
            self.generation_stats['total_output_tokens'] += output_tokens
            self.generation_stats['total_sequences'] += 1

            # Update GPU memory stats
            if torch.cuda.is_available():
                self.generation_stats['gpu_memory_allocated'] = torch.cuda.memory_allocated()

            return {
                'text': generated_text,
                'log_probs': log_probs,
                'tokens': tokens,
                'top_log_probs': top_log_probs
            }

        except Exception as e:
            logger.error(f"Generation with log probs failed: {e}")
            # Fallback to regular generation
            text = self.generate(prompt, max_tokens, temperature)
            return {
                'text': text,
                'log_probs': [],
                'tokens': [],
                'top_log_probs': []
            }

    def estimate_cost(self) -> float:
        """
        Estimate cost for vLLM models (typically free for local inference).

        Returns estimated compute cost based on token usage.
        """
        # For local inference, cost is primarily compute time
        # Similar to HuggingFace, estimate based on tokens processed

        # Rough cost estimation: $0.0005 per 1000 tokens for local compute
        estimated_cost = (self.usage_stats.total_tokens / 1000) * 0.0005
        self.usage_stats.estimated_cost_usd = estimated_cost

        return estimated_cost

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().get_model_info()

        vllm_info = {
            "provider": "vLLM",
            "model_name": self.model_name,
            "max_model_len": self.max_model_len,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "top_logprobs": self.top_logprobs,
            "max_num_seqs": self.max_num_seqs,
            "dtype": self.dtype,
            "generation_stats": self.generation_stats.copy()
        }

        # Add CUDA info if available
        if torch.cuda.is_available():
            vllm_info["cuda_info"] = {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            }

            if torch.cuda.device_count() > 0:
                vllm_info["cuda_info"]["memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

        info.update(vllm_info)
        return info

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {}

        if torch.cuda.is_available():
            memory_info = {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            }

            if torch.cuda.device_count() > 0:
                memory_info["total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
                memory_info["utilization"] = memory_info["allocated_gb"] / memory_info["total_gb"]

        return memory_info

    def clear_cuda_cache(self):
        """Clear CUDA cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            if torch.cuda.is_available():
                self.clear_cuda_cache()
        except:
            pass  # Ignore errors during cleanup


# Utility functions
def get_vllm_supported_models() -> Dict[str, Dict[str, Any]]:
    """List popular models that work well with vLLM."""
    return {
        "llama_models": {
            "meta-llama/Llama-2-7b-hf": {"params": "7B", "use_case": "General purpose, good balance"},
            "meta-llama/Llama-2-13b-hf": {"params": "13B", "use_case": "Better quality, more memory"},
            "meta-llama/Llama-2-70b-hf": {"params": "70B", "use_case": "Highest quality, requires multi-GPU"},
            "meta-llama/Meta-Llama-3-8B": {"params": "8B", "use_case": "Latest Llama 3 model"},
        },
        "mistral_models": {
            "mistralai/Mistral-7B-v0.1": {"params": "7B", "use_case": "Fast, efficient, good quality"},
            "mistralai/Mistral-7B-Instruct-v0.2": {"params": "7B", "use_case": "Instruction-tuned version"},
            "mistralai/Mixtral-8x7B-v0.1": {"params": "8x7B", "use_case": "Mixture of experts, high quality"},
        },
        "other_models": {
            "tiiuae/falcon-7b": {"params": "7B", "use_case": "Open source, permissive license"},
            "01-ai/Yi-6B": {"params": "6B", "use_case": "Bilingual (English/Chinese)"},
            "Qwen/Qwen-7B": {"params": "7B", "use_case": "Strong multilingual capabilities"},
        }
    }


def estimate_vllm_memory_requirements(model_name: str,
                                     dtype: str = "float16",
                                     tensor_parallel_size: int = 1) -> Dict[str, float]:
    """
    Estimate memory requirements for vLLM models.

    Args:
        model_name: Model identifier
        dtype: Data type ('float16', 'bfloat16', 'float32')
        tensor_parallel_size: Number of GPUs for tensor parallelism

    Returns:
        Dictionary with memory estimates in GB
    """
    # Extract parameter count from model name
    param_mappings = {
        "7b": 7e9, "8b": 8e9, "13b": 13e9, "34b": 34e9, "70b": 70e9,
        "6b": 6e9, "6.7b": 6.7e9
    }

    params = 7e9  # Default to 7B
    model_lower = model_name.lower()

    for size_str, size_val in param_mappings.items():
        if size_str in model_lower:
            params = size_val
            break

    # Bytes per parameter based on dtype
    dtype_bytes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int4": 0.5,
    }

    bytes_per_param = dtype_bytes.get(dtype, 2)

    # Calculate memory requirements
    model_memory = params * bytes_per_param / 1e9

    # vLLM has additional memory overhead for KV cache and PagedAttention
    kv_cache_overhead = model_memory * 0.3  # ~30% overhead for KV cache
    total_memory = model_memory + kv_cache_overhead

    # Divide by tensor parallel size if using multiple GPUs
    memory_per_gpu = total_memory / tensor_parallel_size

    return {
        "model_memory_gb": model_memory,
        "kv_cache_overhead_gb": kv_cache_overhead,
        "total_memory_gb": total_memory,
        "memory_per_gpu_gb": memory_per_gpu,
        "recommended_gpu_memory_gb": memory_per_gpu * 1.2,  # 20% safety margin
        "min_gpus_required": max(1, int(total_memory / 24)),  # Assuming 24GB GPUs
    }
