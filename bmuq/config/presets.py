"""
Preset configurations for common BMUq experiments.
"""

from typing import Dict, Any, List
from .settings import BMUqConfig, LLMConfig, UncertaintyConfig, SearchConfig, BenchmarkConfig


def get_preset_config(preset_name: str) -> BMUqConfig:
    """
    Get a preset configuration by name.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        BMUqConfig with preset values
        
    Raises:
        ValueError: If preset name is not found
    """
    presets = {
        "default": _get_default_preset,
        "selfcheck_tree": _get_selfcheck_tree_preset,
        "selfcheck_beam": _get_selfcheck_beam_preset,
        "baseline_comparison": _get_baseline_comparison_preset,
        "fast_development": _get_fast_development_preset,
        "thorough_evaluation": _get_thorough_evaluation_preset,
        "math_problems": _get_math_problems_preset,
        "reasoning_chains": _get_reasoning_chains_preset,
        "huggingface_local": _get_huggingface_local_preset,
        "huggingface_quantized": _get_huggingface_quantized_preset,
    }
    
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    
    return presets[preset_name]()


def list_available_presets() -> List[Dict[str, str]]:
    """
    List all available configuration presets.
    
    Returns:
        List of dictionaries with preset info
    """
    return [
        {
            "name": "default",
            "description": "Default BMUq configuration with SelfCheck and tree search"
        },
        {
            "name": "selfcheck_tree", 
            "description": "SelfCheck uncertainty with tree search (balanced performance)"
        },
        {
            "name": "selfcheck_beam",
            "description": "SelfCheck uncertainty with beam search (faster execution)"
        },
        {
            "name": "baseline_comparison",
            "description": "Configuration for comparing multiple uncertainty methods"
        },
        {
            "name": "fast_development",
            "description": "Fast settings for development and testing (mock LLM)"
        },
        {
            "name": "thorough_evaluation",
            "description": "Thorough evaluation with high beam width and depth"
        },
        {
            "name": "math_problems",
            "description": "Optimized for mathematical problem solving"
        },
        {
            "name": "reasoning_chains",
            "description": "Optimized for multi-step reasoning tasks"
        },
        {
            "name": "huggingface_local",
            "description": "Local HuggingFace model with CUDA support"
        },
        {
            "name": "huggingface_quantized",
            "description": "Memory-efficient quantized HuggingFace model"
        }
    ]


def _get_default_preset() -> BMUqConfig:
    """Default configuration."""
    return BMUqConfig(
        experiment_name="default_experiment",
        description="Default BMUq configuration"
    )


def _get_selfcheck_tree_preset() -> BMUqConfig:
    """SelfCheck with tree search - balanced performance."""
    return BMUqConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4-turbo-preview",
            temperature=0.7
        ),
        uncertainty=UncertaintyConfig(
            method="selfcheck",
            lambda_neg1=1.0,
            lambda_0=0.3
        ),
        search=SearchConfig(
            algorithm="tree_search",
            beam_width=3,
            max_depth=8,
            confidence_threshold=0.2,
            exploration_weight=1.0
        ),
        experiment_name="selfcheck_tree_search",
        description="SelfCheck uncertainty quantification with tree search"
    )


def _get_selfcheck_beam_preset() -> BMUqConfig:
    """SelfCheck with beam search - faster execution."""
    return BMUqConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4-turbo-preview",
            temperature=0.7
        ),
        uncertainty=UncertaintyConfig(
            method="selfcheck",
            lambda_neg1=1.0,
            lambda_0=0.3
        ),
        search=SearchConfig(
            algorithm="beam_search",
            beam_width=3,
            max_depth=6,
            diversity_penalty=0.1
        ),
        experiment_name="selfcheck_beam_search",
        description="SelfCheck uncertainty quantification with beam search"
    )


def _get_baseline_comparison_preset() -> BMUqConfig:
    """Configuration for comparing multiple uncertainty methods."""
    return BMUqConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4-turbo-preview",
            temperature=0.7
        ),
        uncertainty=UncertaintyConfig(
            method="entropy_based",  # Will be overridden for comparison
            num_samples=5,
            sampling_temperature=0.8
        ),
        search=SearchConfig(
            algorithm="beam_search",
            beam_width=3,
            max_depth=6
        ),
        benchmark=BenchmarkConfig(
            metrics=["accuracy", "confidence_correlation", "uncertainty_quality", "method_comparison"],
            save_intermediate=True
        ),
        experiment_name="baseline_comparison",
        description="Compare multiple uncertainty quantification methods"
    )


def _get_fast_development_preset() -> BMUqConfig:
    """Fast settings for development and testing."""
    return BMUqConfig(
        llm=LLMConfig(
            provider="mock",
            model="mock-llm",
            temperature=0.7
        ),
        uncertainty=UncertaintyConfig(
            method="random_baseline"
        ),
        search=SearchConfig(
            algorithm="beam_search",
            beam_width=2,
            max_depth=4
        ),
        benchmark=BenchmarkConfig(
            num_questions=10,
            verbose=True
        ),
        experiment_name="fast_development",
        description="Fast configuration for development and testing"
    )


def _get_thorough_evaluation_preset() -> BMUqConfig:
    """Thorough evaluation with high beam width and depth."""
    return BMUqConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4",  # More capable model
            temperature=0.5,  # Lower temperature for more consistent results
            max_tokens=200
        ),
        uncertainty=UncertaintyConfig(
            method="selfcheck",
            lambda_neg1=1.2,
            lambda_0=0.4
        ),
        search=SearchConfig(
            algorithm="tree_search",
            beam_width=5,
            max_depth=10,
            confidence_threshold=0.1,
            exploration_weight=0.8
        ),
        benchmark=BenchmarkConfig(
            metrics=["accuracy", "confidence_correlation", "uncertainty_quality", 
                    "path_diversity", "search_efficiency"],
            save_intermediate=True,
            verbose=True
        ),
        experiment_name="thorough_evaluation",
        description="Comprehensive evaluation with extensive search"
    )


def _get_math_problems_preset() -> BMUqConfig:
    """Optimized for mathematical problem solving."""
    return BMUqConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4-turbo-preview",
            temperature=0.3,  # Lower temperature for mathematical accuracy
            max_tokens=300,   # More tokens for detailed math
            extra_params={"top_p": 0.9}
        ),
        uncertainty=UncertaintyConfig(
            method="selfcheck",
            lambda_neg1=1.5,  # Higher penalty for contradictions in math
            lambda_0=0.2      # Lower penalty for uncertain steps
        ),
        search=SearchConfig(
            algorithm="tree_search",
            beam_width=4,
            max_depth=12,  # Allow longer reasoning chains
            confidence_threshold=0.3,
            completion_keywords=[
                'answer is', 'therefore', 'final answer', 'result is',
                'solution is', 'equals', 'x =', 'y =', 'the answer'
            ]
        ),
        benchmark=BenchmarkConfig(
            dataset="math",
            metrics=["accuracy", "mathematical_correctness", "solution_completeness"]
        ),
        experiment_name="math_problems",
        description="Configuration optimized for mathematical problem solving"
    )


def _get_reasoning_chains_preset() -> BMUqConfig:
    """Optimized for multi-step reasoning tasks."""
    return BMUqConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4-turbo-preview",
            temperature=0.6,
            max_tokens=200
        ),
        uncertainty=UncertaintyConfig(
            method="consistency_based",  # Good for logical reasoning
        ),
        search=SearchConfig(
            algorithm="tree_search",
            beam_width=4,
            max_depth=15,  # Allow very long reasoning chains
            confidence_threshold=0.2,
            exploration_weight=1.2,  # Encourage exploration of reasoning paths
        ),
        benchmark=BenchmarkConfig(
            metrics=["reasoning_quality", "chain_coherence", "logical_consistency"]
        ),
        experiment_name="reasoning_chains",
        description="Configuration for complex multi-step reasoning tasks"
    )


def create_custom_preset(base_preset: str, modifications: Dict[str, Any]) -> BMUqConfig:
    """
    Create a custom preset by modifying a base preset.
    
    Args:
        base_preset: Name of base preset to modify
        modifications: Dictionary of modifications to apply
        
    Returns:
        Modified BMUqConfig
        
    Example:
        config = create_custom_preset("selfcheck_tree", {
            "search.beam_width": 5,
            "llm.temperature": 0.5,
            "experiment_name": "custom_experiment"
        })
    """
    config = get_preset_config(base_preset)
    
    # Apply modifications using dot notation
    for key, value in modifications.items():
        parts = key.split('.')
        
        # Navigate to the target object
        target = config
        for part in parts[:-1]:
            target = getattr(target, part)
        
        # Set the final attribute
        setattr(target, parts[-1], value)
    
    return config


def _get_huggingface_local_preset() -> BMUqConfig:
    """HuggingFace local model with CUDA support."""
    return BMUqConfig(
        llm=LLMConfig(
            provider="huggingface",
            model="microsoft/DialoGPT-medium",  # Good balance of quality and memory usage
            temperature=0.7,
            max_tokens=200,
            extra_params={
                "device": "auto",  # Auto-detect CUDA
                "use_quantization": False,
                "load_in_8bit": False,
                "load_in_4bit": False,
                "trust_remote_code": False
            }
        ),
        uncertainty=UncertaintyConfig(
            method="selfcheck",
            lambda_neg1=1.0,
            lambda_0=0.3
        ),
        search=SearchConfig(
            algorithm="tree_search",
            beam_width=3,
            max_depth=8,
            confidence_threshold=0.2
        ),
        experiment_name="huggingface_local",
        description="Local HuggingFace model with CUDA acceleration"
    )


def _get_huggingface_quantized_preset() -> BMUqConfig:
    """Memory-efficient quantized HuggingFace model."""
    return BMUqConfig(
        llm=LLMConfig(
            provider="huggingface",
            model="microsoft/DialoGPT-large",  # Larger model made efficient with quantization
            temperature=0.7,
            max_tokens=200,
            extra_params={
                "device": "auto",
                "use_quantization": True,
                "load_in_8bit": True,  # Use 8-bit quantization for memory efficiency
                "load_in_4bit": False,
                "trust_remote_code": False
            }
        ),
        uncertainty=UncertaintyConfig(
            method="selfcheck",
            lambda_neg1=1.2,  # Slightly higher penalty for quantized models
            lambda_0=0.3
        ),
        search=SearchConfig(
            algorithm="beam_search",  # Beam search for efficiency
            beam_width=3,
            max_depth=6,
            diversity_penalty=0.1
        ),
        benchmark=BenchmarkConfig(
            metrics=["accuracy", "confidence_correlation", "uncertainty_quality"]
        ),
        experiment_name="huggingface_quantized",
        description="Memory-efficient quantized HuggingFace model for resource-constrained environments"
    )