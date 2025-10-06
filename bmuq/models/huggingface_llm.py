"""
HuggingFace Transformers implementation with CUDA support for uncertainty quantification.
"""

import os
import torch
from typing import List, Optional, Dict, Any, Union
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    pipeline, BitsAndBytesConfig
)
import logging

from .base import BaseLLM, LLMUsageStats

logger = logging.getLogger(__name__)


class HuggingFaceLLM(BaseLLM):
    """
    HuggingFace Transformers LLM with CUDA support and memory optimization.
    
    Supports both causal LM (GPT-style) and seq2seq models (T5-style) with:
    - Automatic CUDA detection and usage
    - Memory optimization with quantization
    - Batch processing
    - Flexible model loading
    """

    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 device: Optional[str] = None,
                 temperature: float = 0.7,
                 max_retries: int = 3,
                 max_new_tokens: int = 150,
                 use_quantization: bool = False,
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False,
                 trust_remote_code: bool = False,
                 cache_dir: Optional[str] = None):
        """
        Initialize HuggingFace LLM.

        Args:
            model_name: HuggingFace model identifier or local path
            device: Device to use ('cuda', 'cpu', 'auto'). Auto-detects if None
            temperature: Generation temperature
            max_retries: Maximum retries for failed generations
            max_new_tokens: Maximum tokens to generate
            use_quantization: Whether to use quantization for memory efficiency
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision  
            trust_remote_code: Whether to trust remote code for custom models
            cache_dir: Directory to cache downloaded models
        """
        super().__init__(model_name, temperature, max_retries)
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.use_quantization = use_quantization
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir or os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        
        # Auto-detect device
        self.device = self._setup_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.model_type = None  # 'causal' or 'seq2seq'
        
        self._load_model()
        
        # Track generation statistics
        self.generation_stats = {
            'total_generations': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'cuda_memory_allocated': 0,
            'cuda_memory_cached': 0
        }

    def _setup_device(self, device: Optional[str]) -> str:
        """Setup and validate device configuration."""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                device = "cuda"
                # Log CUDA info
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = "cpu"
                logger.warning("CUDA not available, using CPU")
        
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        return device

    def _load_model(self):
        """Load model and tokenizer with appropriate configuration."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Setup quantization config if needed
            quantization_config = None
            if self.load_in_4bit or self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16 if self.load_in_4bit else None,
                    bnb_4bit_quant_type="nf4" if self.load_in_4bit else None,
                    bnb_4bit_use_double_quant=True if self.load_in_4bit else False,
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=self.trust_remote_code
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Try to determine model type and load accordingly
            model_config = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config_only=True,
                cache_dir=self.cache_dir,
                trust_remote_code=self.trust_remote_code
            ).config
            
            model_kwargs = {
                'cache_dir': self.cache_dir,
                'trust_remote_code': self.trust_remote_code,
                'torch_dtype': torch.float16 if self.device == "cuda" else torch.float32,
                'low_cpu_mem_usage': True,
            }
            
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
                model_kwargs['device_map'] = "auto"
            else:
                model_kwargs['device_map'] = self.device if self.device == "cuda" else None
            
            # Try causal LM first
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                self.model_type = 'causal'
                logger.info("Loaded as causal language model")
                
            except (ValueError, OSError):
                # Try seq2seq model
                logger.info("Trying to load as seq2seq model")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                self.model_type = 'seq2seq'
                logger.info("Loaded as seq2seq model")
            
            # Move to device if not using device_map
            if not quantization_config and self.device == "cuda":
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Create generation pipeline
            task = "text-generation" if self.model_type == 'causal' else "text2text-generation"
            self.pipeline = pipeline(
                task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" and not quantization_config else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("Model loaded successfully")
            
            # Log memory usage if CUDA
            if self.device == "cuda":
                self._log_cuda_memory()
                
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _log_cuda_memory(self):
        """Log CUDA memory usage."""
        if torch.cuda.is_available() and self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            logger.info(f"CUDA memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

    def generate(self, prompt: str, max_tokens: int = 150, temperature: Optional[float] = None, structured_output = None) -> str:
        """
        Generate text using the loaded model.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature
            structured_output: A Pydantic class defining the format of the output (not implemented yet)

        Returns:
            Generated text response
        """
        if self.structured_output is not None:
            print("WARNING: the structured output is not available in Huggingface yet!")

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        temp = temperature if temperature is not None else self.temperature
        max_tokens = min(max_tokens, self.max_new_tokens)
        
        try:
            # Prepare generation parameters
            generation_kwargs = {
                'max_new_tokens': max_tokens,
                'temperature': temp,
                'do_sample': temp > 0,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'return_full_text': False,  # Only return new tokens
            }
            
            # Add system prompt for reasoning tasks
            if self.model_type == 'causal':
                formatted_prompt = f"You are a helpful assistant that provides clear, step-by-step mathematical reasoning.\n\nUser: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = prompt
            
            # Generate using pipeline
            with torch.no_grad():
                outputs = self.pipeline(
                    formatted_prompt,
                    **generation_kwargs,
                    batch_size=1
                )
            
            # Extract generated text
            if isinstance(outputs, list) and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
            else:
                generated_text = str(outputs)
            
            # Clean up the output
            generated_text = generated_text.strip()
            
            # Update statistics
            input_tokens = len(self.tokenizer.encode(formatted_prompt))
            output_tokens = len(self.tokenizer.encode(generated_text))
            
            self.usage_stats.total_requests += 1
            self.usage_stats.input_tokens += input_tokens
            self.usage_stats.output_tokens += output_tokens
            self.usage_stats.total_tokens += input_tokens + output_tokens
            
            self.generation_stats['total_generations'] += 1
            self.generation_stats['total_input_tokens'] += input_tokens
            self.generation_stats['total_output_tokens'] += output_tokens
            
            # Update CUDA memory stats
            if self.device == "cuda" and torch.cuda.is_available():
                self.generation_stats['cuda_memory_allocated'] = torch.cuda.memory_allocated()
                self.generation_stats['cuda_memory_cached'] = torch.cuda.memory_reserved()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: Generation failed due to {type(e).__name__}: {str(e)}"

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
        if not prompts:
            return []
        
        temp = temperature if temperature is not None else self.temperature
        max_tokens = min(max_tokens, self.max_new_tokens)
        
        try:
            # Format prompts
            if self.model_type == 'causal':
                formatted_prompts = [
                    f"You are a helpful assistant that provides clear, step-by-step mathematical reasoning.\n\nUser: {prompt}\n\nAssistant:"
                    for prompt in prompts
                ]
            else:
                formatted_prompts = prompts
            
            generation_kwargs = {
                'max_new_tokens': max_tokens,
                'temperature': temp,
                'do_sample': temp > 0,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'return_full_text': False,
                'batch_size': min(len(prompts), 4)  # Limit batch size to avoid OOM
            }
            
            # Generate in batch
            with torch.no_grad():
                outputs = self.pipeline(
                    formatted_prompts,
                    **generation_kwargs
                )
            
            # Process outputs
            results = []
            for output in outputs:
                if isinstance(output, list) and len(output) > 0:
                    generated_text = output[0]['generated_text']
                else:
                    generated_text = str(output)
                results.append(generated_text.strip())
            
            # Update usage statistics
            for formatted_prompt, result in zip(formatted_prompts, results):
                input_tokens = len(self.tokenizer.encode(formatted_prompt))
                output_tokens = len(self.tokenizer.encode(result))
                
                self.usage_stats.total_requests += 1
                self.usage_stats.input_tokens += input_tokens
                self.usage_stats.output_tokens += output_tokens
                self.usage_stats.total_tokens += input_tokens + output_tokens
            
            return results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            # Fallback to individual generation
            return [self.generate(prompt, max_tokens, temperature) for prompt in prompts]

    def estimate_cost(self) -> float:
        """
        Estimate cost for HuggingFace models (typically free for local inference).
        
        Returns estimated compute cost based on token usage and model size.
        """
        # For local inference, cost is primarily compute time
        # We can estimate based on model parameters and tokens processed
        
        if hasattr(self.model, 'num_parameters'):
            model_params = self.model.num_parameters()
        else:
            # Rough estimation based on model name
            if 'large' in self.model_name.lower():
                model_params = 770e6  # ~770M parameters
            elif 'base' in self.model_name.lower():
                model_params = 220e6  # ~220M parameters
            else:
                model_params = 125e6  # ~125M parameters (small)
        
        # Rough cost estimation: $0.001 per 1B parameter-tokens
        param_token_operations = model_params * self.usage_stats.total_tokens
        estimated_cost = param_token_operations * 0.001 / 1e9
        
        return estimated_cost

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().get_model_info()
        
        model_info = {
            "provider": "HuggingFace",
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "use_quantization": self.use_quantization,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "generation_stats": self.generation_stats.copy()
        }
        
        # Add model-specific info if available
        if self.model is not None:
            if hasattr(self.model, 'config'):
                model_info["vocab_size"] = getattr(self.model.config, 'vocab_size', 'unknown')
                model_info["hidden_size"] = getattr(self.model.config, 'hidden_size', 'unknown')
                model_info["num_layers"] = getattr(self.model.config, 'num_hidden_layers', 
                                                  getattr(self.model.config, 'num_layers', 'unknown'))
            
            if hasattr(self.model, 'num_parameters'):
                model_info["num_parameters"] = self.model.num_parameters()
        
        # Add CUDA info if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            model_info["cuda_info"] = {
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        
        info.update(model_info)
        return info

    def clear_cuda_cache(self):
        """Clear CUDA cache to free memory."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {}
        
        if self.device == "cuda" and torch.cuda.is_available():
            memory_info = {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
            }
            memory_info["utilization"] = memory_info["allocated_gb"] / memory_info["total_gb"]
        
        return memory_info

    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                self.clear_cuda_cache()
        except:
            pass  # Ignore errors during cleanup


# Utility functions for model selection
def list_recommended_models() -> Dict[str, Dict[str, Any]]:
    """List recommended HuggingFace models for different use cases."""
    return {
        "small_models": {
            "microsoft/DialoGPT-small": {"params": "117M", "use_case": "Fast inference, limited capability"},
            "distilgpt2": {"params": "82M", "use_case": "Very fast, basic text generation"},
            "gpt2": {"params": "124M", "use_case": "Classic GPT-2, good balance"},
        },
        "medium_models": {
            "microsoft/DialoGPT-medium": {"params": "345M", "use_case": "Good quality conversations"},
            "gpt2-medium": {"params": "345M", "use_case": "Better text quality than small"},
            "facebook/opt-350m": {"params": "350M", "use_case": "Meta's OPT model, efficient"},
        },
        "large_models": {
            "microsoft/DialoGPT-large": {"params": "762M", "use_case": "High quality, needs more memory"},
            "gpt2-large": {"params": "762M", "use_case": "Large GPT-2, high quality"},
            "facebook/opt-1.3b": {"params": "1.3B", "use_case": "Very high quality, requires GPU"},
        },
        "instruction_tuned": {
            "microsoft/DialoGPT-medium": {"params": "345M", "use_case": "Conversational, good for reasoning"},
            "google/flan-t5-base": {"params": "250M", "use_case": "Instruction following, seq2seq"},
            "google/flan-t5-large": {"params": "780M", "use_case": "Better instruction following"},
        }
    }


def get_model_memory_requirements(model_name: str, precision: str = "fp16") -> Dict[str, float]:
    """
    Estimate memory requirements for a given model.
    
    Args:
        model_name: HuggingFace model name
        precision: Model precision ('fp32', 'fp16', '8bit', '4bit')
    
    Returns:
        Dictionary with memory estimates in GB
    """
    # Parameter counts for common models (approximate)
    param_counts = {
        "gpt2": 124e6,
        "gpt2-medium": 345e6, 
        "gpt2-large": 762e6,
        "gpt2-xl": 1.5e9,
        "microsoft/DialoGPT-small": 117e6,
        "microsoft/DialoGPT-medium": 345e6,
        "microsoft/DialoGPT-large": 762e6,
        "facebook/opt-350m": 350e6,
        "facebook/opt-1.3b": 1.3e9,
        "facebook/opt-2.7b": 2.7e9,
        "google/flan-t5-small": 80e6,
        "google/flan-t5-base": 250e6,
        "google/flan-t5-large": 780e6,
        "google/flan-t5-xl": 3e9,
    }
    
    # Get parameter count
    params = param_counts.get(model_name, 350e6)  # Default to medium size
    
    # Bytes per parameter based on precision
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "8bit": 1,
        "4bit": 0.5
    }
    
    bytes_per_param = precision_bytes.get(precision, 2)  # Default to fp16
    
    # Calculate memory requirements
    model_memory = params * bytes_per_param / 1e9  # Convert to GB
    
    # Add overhead for activations, gradients, etc.
    overhead_factor = 1.5
    total_memory = model_memory * overhead_factor
    
    return {
        "model_memory_gb": model_memory,
        "total_memory_gb": total_memory,
        "recommended_gpu_memory_gb": total_memory * 1.2  # 20% safety margin
    }