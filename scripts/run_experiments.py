#!/usr/bin/env python3
"""
Script for running BMUq experiments from command line.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

# Add the parent directory to path to import bmuq
sys.path.insert(0, str(Path(__file__).parent.parent))

from bmuq.config import load_config, get_preset_config, list_available_presets
from bmuq.benchmarks import load_dataset, BMUqBenchmark
from bmuq.benchmarks.datasets import list_available_datasets


def run_single_experiment(config_path: Optional[str] = None, 
                         preset: Optional[str] = None,
                         dataset: str = "gsm8k",
                         dataset_path: Optional[str] = None,
                         num_questions: Optional[int] = None,
                         output_dir: str = "results",
                         verbose: bool = True) -> None:
    """Run a single BMUq experiment."""
    
    # Load configuration
    if config_path:
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
    elif preset:
        print(f"Using preset configuration: {preset}")
        config = get_preset_config(preset)
    else:
        print("Using default configuration")
        config = get_preset_config("default")
    
    # Override settings from command line
    if num_questions:
        config.benchmark.num_questions = num_questions
    config.benchmark.output_dir = output_dir
    config.benchmark.verbose = verbose
    
    print(f"\nExperiment Configuration:")
    print(f"  Name: {config.experiment_name}")
    print(f"  LLM: {config.llm.provider} - {config.llm.model}")
    print(f"  Uncertainty: {config.uncertainty.method}")
    print(f"  Search: {config.search.algorithm}")
    print(f"  Dataset: {dataset}")
    
    # Load dataset
    try:
        dataset_obj = load_dataset(dataset, dataset_path)
        print(f"  Questions: {len(dataset_obj)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Run benchmark
    try:
        benchmark = BMUqBenchmark(config)
        results = benchmark.run(
            dataset=dataset_obj,
            num_questions=config.benchmark.num_questions,
            save_results=True,
            output_dir=output_dir
        )
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {output_dir}/")
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()


def run_comparison_experiment(methods: List[str],
                            dataset: str = "gsm8k", 
                            dataset_path: Optional[str] = None,
                            num_questions: Optional[int] = None,
                            output_dir: str = "results",
                            base_preset: str = "baseline_comparison") -> None:
    """Run comparison across multiple uncertainty methods."""
    
    print(f"Running comparison experiment")
    print(f"Methods: {', '.join(methods)}")
    
    # Load base configuration
    config = get_preset_config(base_preset)
    config.experiment_name = "method_comparison"
    
    if num_questions:
        config.benchmark.num_questions = num_questions
    config.benchmark.output_dir = output_dir
    
    # Load dataset
    try:
        dataset_obj = load_dataset(dataset, dataset_path)
        print(f"Dataset: {dataset} ({len(dataset_obj)} questions)")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Run comparison
    try:
        benchmark = BMUqBenchmark(config)
        comparison_results = benchmark.run_comparison(
            uncertainty_methods=methods,
            dataset=dataset_obj,
            num_questions=config.benchmark.num_questions
        )
        
        print(f"\nComparison Results:")
        print(f"{'Method':<20} {'Success Rate':<12} {'Avg Confidence':<15}")
        print("-" * 50)
        
        for method, result in comparison_results.items():
            print(f"{method:<20} {result.success_rate:<12.3f} {result.average_confidence:<15.3f}")
        
        print(f"\nDetailed results saved to: {output_dir}/")
        
    except Exception as e:
        print(f"Error running comparison: {e}")
        import traceback
        traceback.print_exc()


def list_available_options():
    """List available presets, datasets, and methods."""
    
    print("Available Presets:")
    presets = list_available_presets()
    for preset in presets:
        print(f"  {preset['name']:<20} - {preset['description']}")
    
    print(f"\nAvailable Datasets:")
    datasets = list_available_datasets()
    for dataset in datasets:
        print(f"  {dataset['name']:<20} - {dataset['description']}")
    
    print(f"\nAvailable Uncertainty Methods:")
    methods = ["selfcheck", "entropy_based", "consistency_based", "random_baseline"]
    for method in methods:
        print(f"  {method}")
    
    print(f"\nAvailable Search Algorithms:")
    algorithms = ["tree_search", "beam_search"]
    for algorithm in algorithms:
        print(f"  {algorithm}")


def main():
    """Main entry point for experiment script."""
    
    parser = argparse.ArgumentParser(
        description="Run BMUq uncertainty quantification experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with preset configuration
  python scripts/run_experiments.py --preset selfcheck_tree

  # Run with custom config file
  python scripts/run_experiments.py --config my_config.yaml

  # Compare multiple methods
  python scripts/run_experiments.py --compare selfcheck entropy_based consistency_based

  # Run on custom dataset
  python scripts/run_experiments.py --dataset custom --dataset-path data/my_questions.json

  # List available options
  python scripts/run_experiments.py --list
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config", "-c", type=str, help="Path to configuration file")
    config_group.add_argument("--preset", "-p", type=str, help="Preset configuration name")
    
    # Dataset options
    parser.add_argument("--dataset", "-d", type=str, default="gsm8k", 
                       help="Dataset name (default: gsm8k)")
    parser.add_argument("--dataset-path", type=str, 
                       help="Path to custom dataset file")
    parser.add_argument("--num-questions", "-n", type=int, 
                       help="Number of questions to evaluate")
    
    # Experiment options
    parser.add_argument("--compare", nargs="+", metavar="METHOD",
                       help="Compare multiple uncertainty methods")
    parser.add_argument("--output-dir", "-o", type=str, default="results",
                       help="Output directory for results (default: results)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode (minimal output)")
    
    # Information
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available presets, datasets, and methods")
    
    args = parser.parse_args()
    
    # Handle list option
    if args.list:
        list_available_options()
        return
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    # Run comparison if requested
    if args.compare:
        run_comparison_experiment(
            methods=args.compare,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            num_questions=args.num_questions,
            output_dir=args.output_dir
        )
    else:
        # Run single experiment
        run_single_experiment(
            config_path=args.config,
            preset=args.preset,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            num_questions=args.num_questions,
            output_dir=args.output_dir,
            verbose=verbose
        )


if __name__ == "__main__":
    main()