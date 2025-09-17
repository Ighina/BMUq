"""
Example of custom configuration and advanced BMUq usage.
"""

import os
from bmuq.config import BMUqConfig, LLMConfig, UncertaintyConfig, SearchConfig, BenchmarkConfig
from bmuq.config import save_config, load_config, create_custom_preset
from bmuq import BMUqBenchmark
from bmuq.benchmarks import load_dataset


def create_custom_configuration():
    """Create a custom configuration from scratch."""
    
    print("Creating Custom Configuration")
    print("=" * 40)
    
    # Create custom LLM configuration
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4-turbo-preview",
        temperature=0.5,  # Lower temperature for more consistent results
        max_tokens=200,   # Allow longer responses
        max_retries=5     # More retries for robustness
    )
    
    # Create custom uncertainty configuration
    uncertainty_config = UncertaintyConfig(
        method="selfcheck",
        lambda_neg1=1.2,  # Higher penalty for contradictions
        lambda_0=0.25     # Slightly lower penalty for uncertain steps
    )
    
    # Create custom search configuration
    search_config = SearchConfig(
        algorithm="tree_search",
        beam_width=4,           # Wider search
        max_depth=10,           # Deeper reasoning chains
        confidence_threshold=0.15,  # Lower threshold to explore more
        exploration_weight=1.2,     # Encourage exploration
        completion_keywords=[
            'answer is', 'therefore', 'final answer', 'result is',
            'solution is', 'equals', 'x =', 'y =', 'conclusion'
        ]
    )
    
    # Create custom benchmark configuration
    benchmark_config = BenchmarkConfig(
        dataset="gsm8k",
        num_questions=20,
        shuffle=True,
        seed=123,
        metrics=[
            "accuracy", "confidence_correlation", "uncertainty_quality",
            "path_diversity", "search_efficiency"
        ],
        output_dir="custom_results",
        save_intermediate=True,
        verbose=True
    )
    
    # Combine into main configuration
    config = BMUqConfig(
        llm=llm_config,
        uncertainty=uncertainty_config,
        search=search_config,
        benchmark=benchmark_config,
        experiment_name="custom_math_solver",
        description="Custom configuration for mathematical problem solving",
        random_seed=123
    )
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return None
    
    print("Custom configuration created successfully!")
    print(f"  LLM: {config.llm.provider} - {config.llm.model}")
    print(f"  Uncertainty: {config.uncertainty.method}")
    print(f"  Search: {config.search.algorithm} (beam_width={config.search.beam_width})")
    
    return config


def save_and_load_configuration():
    """Demonstrate saving and loading configurations."""
    
    print("\nSaving and Loading Configurations")
    print("=" * 40)
    
    # Create a configuration
    config = create_custom_configuration()
    if config is None:
        return
    
    # Save configuration to file
    config_path = "custom_config.yaml"
    save_config(config, config_path, format="yaml")
    print(f"Configuration saved to {config_path}")
    
    # Load configuration back
    loaded_config = load_config(config_path)
    print(f"Configuration loaded from {config_path}")
    
    # Verify they match
    assert loaded_config.experiment_name == config.experiment_name
    assert loaded_config.llm.model == config.llm.model
    print("Configuration integrity verified!")
    
    return loaded_config


def modify_preset_configuration():
    """Demonstrate modifying a preset configuration."""
    
    print("\nModifying Preset Configuration")
    print("=" * 40)
    
    # Start with a preset and modify it
    modifications = {
        "search.beam_width": 5,
        "search.max_depth": 12,
        "llm.temperature": 0.3,
        "uncertainty.lambda_neg1": 1.5,
        "experiment_name": "modified_selfcheck"
    }
    
    custom_config = create_custom_preset("selfcheck_tree", modifications)
    
    print("Modified preset configuration:")
    print(f"  Beam width: {custom_config.search.beam_width}")
    print(f"  Max depth: {custom_config.search.max_depth}")
    print(f"  Temperature: {custom_config.llm.temperature}")
    print(f"  Lambda_neg1: {custom_config.uncertainty.lambda_neg1}")
    
    return custom_config


def run_with_custom_config():
    """Run benchmark with custom configuration."""
    
    print("\nRunning Benchmark with Custom Configuration")
    print("=" * 40)
    
    # Create configuration
    config = create_custom_configuration()
    if config is None:
        return
    
    # Switch to mock LLM for this example
    config.llm.provider = "mock"
    config.llm.model = "mock-llm"
    config.benchmark.num_questions = 5  # Fewer questions for demo
    
    # Use built-in GSM8K sample data
    dataset = load_dataset("gsm8k")
    
    # Initialize and run benchmark
    benchmark = BMUqBenchmark(config)
    results = benchmark.run(
        dataset=dataset,
        num_questions=config.benchmark.num_questions,
        save_results=True
    )
    
    # Display detailed results
    print(f"\nDetailed Results:")
    print(f"  Experiment: {results.experiment_name}")
    print(f"  Runtime: {results.total_runtime_seconds:.2f} seconds")
    print(f"  Success rate: {results.success_rate:.3f}")
    
    # Show metrics
    print(f"\nMetrics:")
    for metric_name, metric_result in results.metrics.items():
        print(f"  {metric_name}: {metric_result.value:.3f}")
    
    # Show search statistics if available
    if "search_stats" in results.metadata:
        stats = results.metadata["search_stats"]
        print(f"\nSearch Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def environment_variable_config():
    """Demonstrate using environment variables for configuration."""
    
    print("\nEnvironment Variable Configuration")
    print("=" * 40)
    
    # Set some environment variables for demo
    os.environ["BMUQ_LLM_MODEL"] = "gpt-3.5-turbo"
    os.environ["BMUQ_SEARCH_BEAM_WIDTH"] = "2"
    os.environ["BMUQ_UNCERTAINTY_METHOD"] = "entropy_based"
    os.environ["BMUQ_EXPERIMENT_NAME"] = "env_config_test"
    
    # Load configuration from environment
    from bmuq.config import load_config_from_env
    config = load_config_from_env()
    
    print("Configuration loaded from environment variables:")
    print(f"  LLM Model: {config.llm.model}")
    print(f"  Beam Width: {config.search.beam_width}")
    print(f"  Uncertainty Method: {config.uncertainty.method}")
    print(f"  Experiment Name: {config.experiment_name}")
    
    # Clean up environment variables
    for var in ["BMUQ_LLM_MODEL", "BMUQ_SEARCH_BEAM_WIDTH", 
                "BMUQ_UNCERTAINTY_METHOD", "BMUQ_EXPERIMENT_NAME"]:
        if var in os.environ:
            del os.environ[var]


def configuration_comparison():
    """Compare different configurations side by side."""
    
    print("\nConfiguration Comparison")
    print("=" * 40)
    
    from bmuq.config import get_preset_config, list_available_presets
    
    # List all available presets
    presets = list_available_presets()
    print("Available presets:")
    for preset in presets[:3]:  # Show first 3
        print(f"  {preset['name']}: {preset['description']}")
    
    # Compare two specific presets
    config1 = get_preset_config("selfcheck_tree")
    config2 = get_preset_config("selfcheck_beam")
    
    print(f"\nComparison: Tree Search vs Beam Search")
    print(f"{'Setting':<25} {'Tree Search':<15} {'Beam Search'}")
    print("-" * 55)
    print(f"{'Algorithm':<25} {config1.search.algorithm:<15} {config2.search.algorithm}")
    print(f"{'Beam Width':<25} {config1.search.beam_width:<15} {config2.search.beam_width}")
    print(f"{'Max Depth':<25} {config1.search.max_depth:<15} {config2.search.max_depth}")
    print(f"{'Exploration Weight':<25} {getattr(config1.search, 'exploration_weight', 'N/A'):<15} {getattr(config2.search, 'diversity_penalty', 'N/A')}")


if __name__ == "__main__":
    # Run all examples
    create_custom_configuration()
    save_and_load_configuration()
    modify_preset_configuration()
    environment_variable_config()
    configuration_comparison()
    run_with_custom_config()
    
    print("\n" + "="*60)
    print("Configuration Examples Completed!")
    print("\nKey takeaways:")
    print("1. Configurations can be created programmatically or loaded from files")
    print("2. Preset configurations provide good starting points")
    print("3. Environment variables allow easy experimentation")
    print("4. Always validate configurations before use")
    print("5. Save configurations with results for reproducibility")