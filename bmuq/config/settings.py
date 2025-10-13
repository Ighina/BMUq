"""
Configuration settings for BMUq experiments.
"""

import os
import json
import yaml
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for language model settings."""

    provider: str = "openai"  # openai, anthropic, huggingface, ollama, mock
    model: str = "gpt-4-turbo-preview"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 150
    max_retries: int = 3
    timeout: float = 30.0

    # Ollama-specific settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_system_prompt: Optional[str] = None

    # HuggingFace-specific settings
    hf_device: Optional[str] = None
    hf_use_quantization: bool = False
    hf_load_in_8bit: bool = False
    hf_load_in_4bit: bool = False

    # Provider-specific settings
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification methods."""

    method: str = (
        "selfcheck"  # selfcheck, entropy_based, consistency_based, random_baseline
    )
    answer_weight: bool = (
        False  # Weight uncertainty by answer identity (majority voting)
    )

    # SelfCheck specific settings
    lambda_neg1: float = 1.0
    lambda_0: float = 0.3

    # Semantic Entropy specific settings
    entailment_model: str = "cross-encoder/nli-roberta-base"
    strict_entailment: bool = False
    verbose: bool = False
    add_consistency: bool = False

    # Entropy-based specific settings
    num_samples: int = 5
    sampling_temperature: float = 0.8

    # Additional method parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchConfig:
    """Configuration for search algorithms."""

    algorithm: str = "tree_search"  # tree_search, beam_search
    beam_width: int = 3
    max_depth: int = 8

    # Tree search specific
    confidence_threshold: float = 0.1
    exploration_weight: float = 1.0

    # Beam search specific
    diversity_penalty: float = 0.1

    # Completion detection
    completion_keywords: List[str] = field(
        default_factory=lambda: [
            "answer is",
            "therefore",
            "final answer",
            "result is",
            "conclusion",
            "solution is",
            "the answer",
            "equals",
        ]
    )

    structured_format: bool = True  # Whether to use structured format detection
    structured_output: bool = False
    # Additional algorithm parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking and evaluation."""

    dataset: str = "custom"  # gsm8k, math, aqua, custom
    data_path: Optional[str] = None
    num_questions: Optional[int] = None
    shuffle: bool = True
    seed: int = 42

    # Evaluation metrics
    metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy",
            "confidence_correlation",
            "uncertainty_quality",
        ]
    )

    # Output settings
    output_dir: str = "results"
    save_intermediate: bool = True
    verbose: bool = True


@dataclass
class BMUqConfig:
    """Main configuration for BMUq experiments."""

    # Core components
    llm: LLMConfig = field(default_factory=LLMConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # Experiment metadata
    experiment_name: str = "default_experiment"
    description: str = ""
    version: str = "0.1.0"

    # System settings
    random_seed: int = 42
    num_workers: int = 1
    cache_enabled: bool = True
    log_level: str = "INFO"

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of validation error messages
        """
        issues = []

        # Validate LLM config
        if not self.llm.model:
            issues.append("LLM model must be specified")

        valid_providers = ["openai", "anthropic", "huggingface", "ollama", "mock"]
        if self.llm.provider not in valid_providers:
            issues.append(f"LLM provider must be one of: {valid_providers}")

        if self.llm.temperature < 0.0 or self.llm.temperature > 2.0:
            issues.append("LLM temperature must be between 0.0 and 2.0")

        # Validate uncertainty config
        valid_uncertainty_methods = [
            "selfcheck",
            "entropy_based",
            "consistency_based",
            "random_baseline",
            "semantic_entropy",
            "coherence_based",
            "relative_coherence_based",
        ]
        if self.uncertainty.method not in valid_uncertainty_methods:
            issues.append(
                f"Uncertainty method must be one of: {valid_uncertainty_methods}"
            )

        # Validate search config
        valid_search_algorithms = ["tree_search", "beam_search", "best_of_n"]
        if self.search.algorithm not in valid_search_algorithms:
            issues.append(f"Search algorithm must be one of: {valid_search_algorithms}")

        if self.search.beam_width < 1:
            issues.append("Beam width must be at least 1")

        if self.search.max_depth < 1:
            issues.append("Max depth must be at least 1")

        # Validate paths if specified
        if self.benchmark.data_path and not os.path.exists(self.benchmark.data_path):
            issues.append(
                f"Benchmark data path does not exist: {self.benchmark.data_path}"
            )

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BMUqConfig":
        """Create config from dictionary."""
        # Handle nested dataclasses
        if "llm" in data and isinstance(data["llm"], dict):
            data["llm"] = LLMConfig(**data["llm"])

        if "uncertainty" in data and isinstance(data["uncertainty"], dict):
            data["uncertainty"] = UncertaintyConfig(**data["uncertainty"])

        if "search" in data and isinstance(data["search"], dict):
            data["search"] = SearchConfig(**data["search"])

        if "benchmark" in data and isinstance(data["benchmark"], dict):
            data["benchmark"] = BenchmarkConfig(**data["benchmark"])

        return cls(**data)


def load_config(config_path: Union[str, Path]) -> BMUqConfig:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file (JSON or YAML)

    Returns:
        Loaded BMUqConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        config = BMUqConfig.from_dict(data)

        # Validate loaded config
        issues = config.validate()
        if issues:
            raise ValueError(
                f"Configuration validation failed:\n"
                + "\n".join(f"- {issue}" for issue in issues)
            )

        return config

    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ValueError(f"Failed to parse configuration file: {e}")


def save_config(
    config: BMUqConfig, config_path: Union[str, Path], format: str = "yaml"
) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        config_path: Path to save configuration file
        format: File format ('json' or 'yaml')

    Raises:
        ValueError: If format is invalid
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    with open(config_path, "w") as f:
        if format.lower() == "yaml":
            yaml.safe_dump(data, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")


def get_default_config() -> BMUqConfig:
    """Get default configuration."""
    return BMUqConfig()


def create_config_template(output_path: Union[str, Path], format: str = "yaml") -> None:
    """
    Create a configuration template file.

    Args:
        output_path: Path to save template
        format: File format ('json' or 'yaml')
    """
    default_config = get_default_config()
    save_config(default_config, output_path, format)

    print(f"Configuration template created at: {output_path}")
    print("Edit this file to customize your experiment settings.")


# Environment variable support
def load_config_from_env() -> BMUqConfig:
    """
    Load configuration from environment variables.

    Environment variables should be prefixed with 'BMUQ_' and use underscores
    to separate nested configuration levels. For example:
    - BMUQ_LLM_MODEL=gpt-4
    - BMUQ_SEARCH_BEAM_WIDTH=5
    """
    config = get_default_config()

    # Map environment variables to config fields
    env_mappings = {
        "BMUQ_LLM_MODEL": ("llm", "model"),
        "BMUQ_LLM_PROVIDER": ("llm", "provider"),
        "BMUQ_LLM_API_KEY": ("llm", "api_key"),
        "BMUQ_LLM_TEMPERATURE": ("llm", "temperature"),
        "BMUQ_LLM_OLLAMA_BASE_URL": ("llm", "ollama_base_url"),
        "BMUQ_LLM_OLLAMA_SYSTEM_PROMPT": ("llm", "ollama_system_prompt"),
        "BMUQ_LLM_HF_DEVICE": ("llm", "hf_device"),
        "BMUQ_LLM_HF_USE_QUANTIZATION": ("llm", "hf_use_quantization"),
        "BMUQ_LLM_HF_LOAD_IN_8BIT": ("llm", "hf_load_in_8bit"),
        "BMUQ_LLM_HF_LOAD_IN_4BIT": ("llm", "hf_load_in_4bit"),
        "BMUQ_UNCERTAINTY_METHOD": ("uncertainty", "method"),
        "BMUQ_SEARCH_ALGORITHM": ("search", "algorithm"),
        "BMUQ_SEARCH_BEAM_WIDTH": ("search", "beam_width"),
        "BMUQ_SEARCH_MAX_DEPTH": ("search", "max_depth"),
        "BMUQ_SEARCH_MAX_FORMAT": ("search", "structured_format"),
        "BMUQ_EXPERIMENT_NAME": ("experiment_name",),
        "BMUQ_RANDOM_SEED": ("random_seed",),
    }

    for env_var, path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Navigate to the correct config level
            current = config
            for level in path[:-1]:
                current = getattr(current, level)

            # Set the value with appropriate type conversion
            field_name = path[-1]
            current_value = getattr(current, field_name)

            if isinstance(current_value, bool):
                value = value.lower() in ("true", "1", "yes", "on")
            elif isinstance(current_value, int):
                value = int(value)
            elif isinstance(current_value, float):
                value = float(value)

            setattr(current, field_name, value)

    return config
