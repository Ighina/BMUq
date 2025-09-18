"""
Basic usage example for BMUq - SelfCheck with Tree Search
"""

import os
from bmuq.config import BMUqConfig, get_preset_config
from bmuq.benchmarks import BMUqBenchmark
from bmuq.benchmarks import create_dataset_from_questions

def main():
    """Demonstrate basic BMUq usage."""
    
    # Example questions for demonstration
    questions = [
        "Solve for x: 2x + 3 = 7. Show your work step by step.",
        "A rectangle has length 8 and width 5. What is its area?",
        "If John has 15 apples and gives away 6, how many does he have left?",
        "Find the value of y when y = 3x + 2 and x = 4.",
        "Calculate: (5 + 3) × 2 - 4"
    ]
    
    # Optional ground truth answers
    answers = ["2", "40", "9", "14", "12"]
    
    print("BMUq Basic Usage Example")
    print("=" * 50)
    
    # Option 1: Use preset configuration
    print("\n1. Using preset configuration...")
    config = get_preset_config("selfcheck_tree")
    config.experiment_name = "basic_example"
    
    # For this example, we'll use mock LLM to avoid requiring API keys
    config.llm.provider = "mock"
    config.llm.model = "mock-llm"
    
    print(f"Configuration: {config.uncertainty.method} + {config.search.algorithm}")
    
    # Create custom dataset
    dataset = create_dataset_from_questions(questions, answers, name="basic_examples")
    print(f"Dataset: {len(dataset)} questions")
    
    # Initialize benchmark
    benchmark = BMUqBenchmark(config)
    
    # Run evaluation
    print("\n2. Running evaluation...")
    results = benchmark.run(
        dataset=dataset,
        num_questions=len(questions),
        save_results=True,
        output_dir="results"
    )
    
    # Display results
    print(f"\n3. Results Summary:")
    print(f"   Success rate: {results.success_rate:.3f}")
    print(f"   Average confidence: {results.average_confidence:.3f}")
    print(f"   Average path length: {results.average_path_length:.1f} steps")
    
    # Show individual question results
    print(f"\n4. Individual Results:")
    for i, result in enumerate(results.question_results[:3]):  # Show first 3
        if result.get("success", False):
            print(f"   Question {i+1}: {result.get('correct', False)} (confidence: {result.get('confidence', 0):.3f})")
            print(f"   Predicted: {result.get('predicted_answer', 'N/A')}")
            print(f"   Ground truth: {result.get('ground_truth', 'N/A')}")
            print()
    
    print("Example completed! Check the 'results' directory for detailed output.")


def advanced_example():
    """Demonstrate advanced configuration and comparison."""
    
    print("\nAdvanced Example: Method Comparison")
    print("=" * 50)
    
    # Create base configuration
    config = get_preset_config("baseline_comparison")
    config.llm.provider = "mock"  # Use mock for demo
    config.experiment_name = "method_comparison"
    
    # Create simple dataset
    questions = [
        "What is 5 + 3?",
        "Solve: x + 2 = 5",
        "Find 10% of 50"
    ]
    answers = ["8", "3", "5"]
    
    dataset = create_dataset_from_questions(questions, answers)
    
    # Initialize benchmark
    benchmark = BMUqBenchmark(config)
    
    # Compare multiple uncertainty quantification methods
    methods_to_compare = ["selfcheck", "entropy_based", "consistency_based", "random_baseline"]
    
    print("Running comparison across methods:")
    for method in methods_to_compare:
        print(f"  - {method}")
    
    comparison_results = benchmark.run_comparison(
        uncertainty_methods=methods_to_compare,
        dataset=dataset,
        num_questions=len(questions)
    )
    
    # Compare results
    print(f"\nComparison Results:")
    print(f"{'Method':<20} {'Success Rate':<12} {'Avg Confidence':<15} {'Avg Path Length'}")
    print("-" * 60)
    
    for method, result in comparison_results.items():
        print(f"{method:<20} {result.success_rate:<12.3f} {result.average_confidence:<15.3f} {result.average_path_length:.1f}")


if __name__ == "__main__":
    # Set up basic example
    main()
    
    # Run advanced example  
    advanced_example()
    
    print("\n" + "="*50)
    print("Next Steps:")
    print("- Try with real LLM by setting OPENAI_API_KEY environment variable")
    print("- Experiment with different configurations using bmuq.config.get_preset_config()")
    print("- Load your own datasets using bmuq.benchmarks.load_dataset()")
    print("- See examples/custom_config.py for configuration customization")