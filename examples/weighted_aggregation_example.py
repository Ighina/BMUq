"""
Example demonstrating how to use the weighted answer aggregation system.

This example shows how to:
1. Create reasoning paths with different answers
2. Apply uncertainty methods to evaluate them
3. Use weighted aggregation to combine confidence scores for identical answers
4. Get the most confident answer based on multiple supporting paths
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmuq.core.data_structures import ReasoningStep, ReasoningPath, UncertaintyScore
from bmuq.uncertainty.base_methods import RandomBaselineUQ
from bmuq.uncertainty.coherence_uq import CoherenceBasedUQ
from bmuq.uncertainty.weighted_aggregation import (
    MathAnswerExtractor,
    WeightedAnswerAggregator,
    WeightedUncertaintyMethod,
    create_math_weighted_method
)
from bmuq.uncertainty.adapters import add_weighted_aggregation


def create_sample_reasoning_paths():
    """Create sample reasoning paths for demonstration."""

    # Path 1: Leading to answer "24" (high confidence expected)
    path1 = ReasoningPath([
        ReasoningStep(1, "Let's solve: What is 6 × 4?"),
        ReasoningStep(2, "6 × 4 = 24"),
        ReasoningStep(3, "Therefore, the answer is 24.")
    ], path_id="path_1")

    # Path 2: Also leading to answer "24" (medium confidence expected)
    path2 = ReasoningPath([
        ReasoningStep(1, "I need to calculate 6 × 4"),
        ReasoningStep(2, "6 × 4 = 6 + 6 + 6 + 6 = 12 + 12 = 24"),
        ReasoningStep(3, "The final answer is 24.")
    ], path_id="path_2")

    # Path 3: Leading to answer "25" (wrong answer, lower confidence expected)
    path3 = ReasoningPath([
        ReasoningStep(1, "Computing 6 × 4..."),
        ReasoningStep(2, "6 × 4 = 6 + 6 + 6 + 6 + 1 = 25"),  # Deliberate error
        ReasoningStep(3, "So the answer is 25.")
    ], path_id="path_3")

    # Path 4: Another path leading to "24" (high confidence expected)
    path4 = ReasoningPath([
        ReasoningStep(1, "To find 6 × 4, I'll use the multiplication table"),
        ReasoningStep(2, "From the multiplication table: 6 × 4 = 24"),
        ReasoningStep(3, "Answer: 24")
    ], path_id="path_4")

    return [path1, path2, path3, path4]


def create_complex_math_paths():
    """Create more complex math reasoning paths."""

    # Path 1: Solving quadratic equation, answer x = 2
    path1 = ReasoningPath([
        ReasoningStep(1, "Solve x² - 4x + 4 = 0"),
        ReasoningStep(2, "This is a perfect square: (x - 2)² = 0"),
        ReasoningStep(3, "Therefore x - 2 = 0"),
        ReasoningStep(4, "So x = 2")
    ], path_id="quad_1")

    # Path 2: Same equation, different method, answer x = 2
    path2 = ReasoningPath([
        ReasoningStep(1, "Solve x² - 4x + 4 = 0 using quadratic formula"),
        ReasoningStep(2, "a = 1, b = -4, c = 4"),
        ReasoningStep(3, "x = (4 ± √(16 - 16)) / 2 = (4 ± 0) / 2"),
        ReasoningStep(4, "x = 4/2 = 2")
    ], path_id="quad_2")

    # Path 3: Wrong calculation, answer x = 1
    path3 = ReasoningPath([
        ReasoningStep(1, "Solve x² - 4x + 4 = 0"),
        ReasoningStep(2, "Using quadratic formula with a=1, b=-4, c=4"),
        ReasoningStep(3, "x = (4 ± √(16 - 8)) / 2"),  # Wrong calculation
        ReasoningStep(4, "x = (4 ± √8) / 2 ≈ 1")  # Wrong result
    ], path_id="quad_3")

    return [path1, path2, path3]


def demonstrate_basic_aggregation():
    """Demonstrate basic weighted answer aggregation."""

    print("="*60)
    print("BASIC WEIGHTED ANSWER AGGREGATION DEMO")
    print("="*60)

    # Create sample data
    question = "What is 6 × 4?"
    paths = create_sample_reasoning_paths()

    print(f"Question: {question}")
    print(f"Number of reasoning paths: {len(paths)}")
    print()

    # Create uncertainty method (using random for demo)
    base_method = RandomBaselineUQ(seed=42)  # Fixed seed for reproducible results

    # Add weighted aggregation
    weighted_method = add_weighted_aggregation(base_method, problem_type="math")

    # Evaluate all paths
    candidates = weighted_method.evaluate_paths_with_aggregation(question, paths)

    print("RESULTS:")
    print("-" * 40)
    for i, candidate in enumerate(candidates, 1):
        print(f"{i}. {candidate}")
        print(f"   Supported by {candidate.support_count} path(s)")
        print(f"   Individual confidences: {[f'{c:.3f}' for c in candidate.individual_confidences]}")
        print()

    # Get the best answer
    best_answer = weighted_method.get_best_answer(question, paths)
    if best_answer:
        print(f"BEST ANSWER: {best_answer.answer}")
        print(f"CONFIDENCE: {best_answer.aggregated_confidence:.3f}")
        print(f"SUPPORT: {best_answer.support_count} reasoning path(s)")
    else:
        print("No valid answer found.")


def demonstrate_coherence_aggregation():
    """Demonstrate aggregation with coherence-based uncertainty method."""

    print("\n" + "="*60)
    print("COHERENCE-BASED WEIGHTED AGGREGATION DEMO")
    print("="*60)

    # Create sample data
    question = "Solve x² - 4x + 4 = 0"
    paths = create_complex_math_paths()

    print(f"Question: {question}")
    print(f"Number of reasoning paths: {len(paths)}")
    print()

    # Create coherence-based uncertainty method
    base_method = CoherenceBasedUQ(
        model_name="all-MiniLM-L6-v2",
        coherence_method="mean_cosine_similarity"
    )

    # Add weighted aggregation
    weighted_method = add_weighted_aggregation(
        base_method,
        problem_type="math",
        confidence_aggregation="sum"  # Sum confidences for same answers
    )

    # Evaluate all paths
    candidates = weighted_method.evaluate_paths_with_aggregation(question, paths)

    print("RESULTS WITH COHERENCE-BASED UNCERTAINTY:")
    print("-" * 50)
    for i, candidate in enumerate(candidates, 1):
        stats = candidate.metadata.get('confidence_stats', {})
        print(f"{i}. Answer: {candidate.answer}")
        print(f"   Aggregated confidence: {candidate.aggregated_confidence:.3f}")
        print(f"   Supported by: {candidate.support_count} path(s)")
        print(f"   Confidence stats: mean={stats.get('mean', 0):.3f}, "
              f"min={stats.get('min', 0):.3f}, max={stats.get('max', 0):.3f}")
        print()

    # Compare with non-aggregated evaluation
    print("COMPARISON: Individual path evaluations vs. Aggregated:")
    print("-" * 55)
    for i, path in enumerate(paths, 1):
        individual_confidence = base_method.evaluate_path(question, path)
        print(f"Path {i}: Individual confidence = {individual_confidence:.3f}")

        # Extract answer for this path
        extractor = MathAnswerExtractor()
        answer = extractor.extract_answer(path)
        print(f"         Answer = {answer}")

    print()

    # Show how aggregation helps
    best_answer = weighted_method.get_best_answer(question, paths)
    if best_answer and best_answer.support_count > 1:
        print(f"AGGREGATION BENEFIT:")
        print(f"Answer '{best_answer.answer}' is supported by {best_answer.support_count} paths")
        print(f"Combined confidence: {best_answer.aggregated_confidence:.3f}")
        print(f"Individual confidences: {[f'{c:.3f}' for c in best_answer.individual_confidences]}")
        print(f"This is higher than any individual path confidence!")


def demonstrate_different_aggregation_methods():
    """Demonstrate different confidence aggregation methods."""

    print("\n" + "="*60)
    print("DIFFERENT AGGREGATION METHODS COMPARISON")
    print("="*60)

    question = "What is 6 × 4?"
    paths = create_sample_reasoning_paths()
    base_method = RandomBaselineUQ(seed=42)

    aggregation_methods = ["sum", "mean", "weighted_mean", "max"]

    for method in aggregation_methods:
        print(f"\nAggregation method: {method.upper()}")
        print("-" * 30)

        weighted_method = add_weighted_aggregation(
            base_method,
            problem_type="math",
            confidence_aggregation=method
        )

        candidates = weighted_method.evaluate_paths_with_aggregation(question, paths)

        for candidate in candidates:
            print(f"Answer: {candidate.answer}, "
                  f"Confidence: {candidate.aggregated_confidence:.3f}, "
                  f"Paths: {candidate.support_count}")


def main():
    """Run all demonstration examples."""

    print("WEIGHTED ANSWER AGGREGATION SYSTEM DEMO")
    print("This demo shows how to use confidence scores to weight final answers")
    print("when multiple reasoning paths lead to the same or different conclusions.\n")

    try:
        # Basic demo with random uncertainty method
        demonstrate_basic_aggregation()

        # More realistic demo with coherence-based method
        demonstrate_coherence_aggregation()

        # Compare different aggregation strategies
        demonstrate_different_aggregation_methods()

        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey takeaways:")
        print("1. Multiple paths leading to the same answer get combined confidence")
        print("2. The system automatically extracts and normalizes answers")
        print("3. Different aggregation methods (sum, mean, etc.) can be used")
        print("4. Works with any existing uncertainty quantification method")
        print("5. Provides detailed metadata about the aggregation process")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()