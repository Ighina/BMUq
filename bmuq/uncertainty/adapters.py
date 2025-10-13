"""
Adapters for integrating weighted answer aggregation with existing uncertainty methods.

This module provides convenient adapters and utilities to easily plug the weighted
answer aggregation system on top of existing uncertainty methods without requiring
changes to the underlying implementations.
"""

from typing import List, Optional, Dict, Any, Union
import logging

from ..core.data_structures import ReasoningStep, ReasoningPath, UncertaintyScore
from ..core.interfaces import UncertaintyMethod
from .weighted_aggregation import (
    WeightedUncertaintyMethod,
    WeightedAnswerAggregator,
    MathAnswerExtractor,
    GenericAnswerExtractor,
    AnswerCandidate
)
from .base_methods import EntropyBasedUQ, ConsistencyBasedUQ, RandomBaselineUQ
from .coherence_uq import CoherenceBasedUQ, RelativeCoherenceBasedUQ

# Try to import SelfCheck-related methods if available
try:
    from .selfcheck import SelfCheckUQ  # Assuming this exists
except ImportError:
    SelfCheckUQ = None

try:
    from .uq_methods import SequentialConsistencyBasedUQ, SemanticEntropyBasedUQ
except ImportError:
    SequentialConsistencyBasedUQ = None
    SemanticEntropyBasedUQ = None

logger = logging.getLogger(__name__)


class WeightedAnswerAdapter:
    """
    Universal adapter for adding weighted answer aggregation to any uncertainty method.

    This adapter automatically detects the type of problem (math vs. general) and
    applies appropriate answer extraction and aggregation strategies.
    """

    def __init__(
        self,
        base_method: UncertaintyMethod,
        problem_type: str = "auto",
        confidence_aggregation: str = "sum",
        custom_answer_indicators: Optional[List[str]] = None,
        min_confidence_threshold: float = 0.1
    ):
        """
        Initialize the weighted answer adapter.

        Args:
            base_method: The base uncertainty method to wrap
            problem_type: Type of problem ("math", "general", or "auto" for auto-detection)
            confidence_aggregation: How to aggregate confidences ("sum", "mean", "weighted_mean", "max")
            custom_answer_indicators: Custom phrases that indicate answers (for general problems)
            min_confidence_threshold: Minimum confidence to consider a reasoning path
        """
        self.base_method = base_method
        self.problem_type = problem_type
        self.confidence_aggregation = confidence_aggregation
        self.custom_answer_indicators = custom_answer_indicators
        self.min_confidence_threshold = min_confidence_threshold

        # Initialize the weighted uncertainty method
        self._initialize_weighted_method()

    def _initialize_weighted_method(self):
        """Initialize the appropriate weighted uncertainty method based on problem type."""
        if self.problem_type == "math":
            extractor = MathAnswerExtractor()
        elif self.problem_type == "general":
            extractor = GenericAnswerExtractor(self.custom_answer_indicators)
        else:  # auto-detect
            # Default to math extractor, but we'll try both during evaluation
            extractor = MathAnswerExtractor()

        aggregator = WeightedAnswerAggregator(
            answer_extractor=extractor,
            confidence_aggregation=self.confidence_aggregation,
            min_confidence_threshold=self.min_confidence_threshold
        )

        self.weighted_method = WeightedUncertaintyMethod(
            base_uncertainty_method=self.base_method,
            answer_extractor=extractor,
            aggregator=aggregator
        )

        # Keep a backup generic extractor for auto-detection
        if self.problem_type == "auto":
            self.generic_extractor = GenericAnswerExtractor(self.custom_answer_indicators)

    def evaluate_paths_with_aggregation(
        self,
        question: str,
        reasoning_paths: List[ReasoningPath]
    ) -> List[AnswerCandidate]:
        """
        Evaluate multiple reasoning paths and return weighted answer candidates.

        Args:
            question: The original question
            reasoning_paths: List of reasoning paths to evaluate

        Returns:
            List of AnswerCandidate objects ranked by aggregated confidence
        """
        if self.problem_type == "auto":
            # Try math extraction first
            candidates = self.weighted_method.evaluate_multiple_paths(question, reasoning_paths)

            # If we didn't get good answers, try generic extraction
            if not candidates or all(not candidate.answer.strip() for candidate in candidates):
                logger.info("Math extraction failed, trying generic extraction")
                self.weighted_method.answer_extractor = self.generic_extractor
                self.weighted_method.aggregator.answer_extractor = self.generic_extractor
                candidates = self.weighted_method.evaluate_multiple_paths(question, reasoning_paths)
                # Reset to math extractor for next time
                self.weighted_method.answer_extractor = MathAnswerExtractor()
                self.weighted_method.aggregator.answer_extractor = self.weighted_method.answer_extractor

        else:
            candidates = self.weighted_method.evaluate_multiple_paths(question, reasoning_paths)

        return candidates

    def get_best_answer(
        self,
        question: str,
        reasoning_paths: List[ReasoningPath]
    ) -> Optional[AnswerCandidate]:
        """
        Get the best weighted answer from multiple reasoning paths.

        Args:
            question: The original question
            reasoning_paths: List of reasoning paths to evaluate

        Returns:
            Best answer candidate with aggregated confidence, or None
        """
        candidates = self.evaluate_paths_with_aggregation(question, reasoning_paths)
        return candidates[0] if candidates else None

    def evaluate_single_path(
        self,
        question: str,
        reasoning_path: ReasoningPath
    ) -> float:
        """
        Evaluate a single reasoning path using the base method.

        Args:
            question: The original question
            reasoning_path: The reasoning path to evaluate

        Returns:
            Confidence score for the path
        """
        return self.base_method.evaluate_path(question, reasoning_path)

    def get_method_info(self) -> Dict[str, Any]:
        """Get information about this adapter and the underlying method."""
        info = self.base_method.get_method_info()
        info.update({
            "adapter_type": "WeightedAnswerAdapter",
            "problem_type": self.problem_type,
            "confidence_aggregation": self.confidence_aggregation,
            "weighted_aggregation": True
        })
        return info


def add_weighted_aggregation(
    uncertainty_method: UncertaintyMethod,
    problem_type: str = "auto",
    confidence_aggregation: str = "sum",
    **kwargs
) -> WeightedAnswerAdapter:
    """
    Convenience function to add weighted answer aggregation to any uncertainty method.

    Args:
        uncertainty_method: The uncertainty method to enhance
        problem_type: Type of problem ("math", "general", or "auto")
        confidence_aggregation: How to aggregate confidences
        **kwargs: Additional arguments for WeightedAnswerAdapter

    Returns:
        Enhanced uncertainty method with weighted answer aggregation
    """
    return WeightedAnswerAdapter(
        base_method=uncertainty_method,
        problem_type=problem_type,
        confidence_aggregation=confidence_aggregation,
        **kwargs
    )


class BulkEvaluator:
    """
    Utility class for bulk evaluation of multiple questions with weighted aggregation.

    This class is useful when you want to evaluate many questions and compare
    the performance of different uncertainty methods with and without weighted aggregation.
    """

    def __init__(self, uncertainty_methods: List[UncertaintyMethod]):
        """
        Initialize bulk evaluator.

        Args:
            uncertainty_methods: List of uncertainty methods to evaluate
        """
        self.base_methods = uncertainty_methods
        self.weighted_adapters = [
            add_weighted_aggregation(method, problem_type="auto")
            for method in uncertainty_methods
        ]

    def evaluate_question_batch(
        self,
        questions: List[str],
        reasoning_paths_per_question: List[List[ReasoningPath]],
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of questions with multiple reasoning paths each.

        Args:
            questions: List of questions to evaluate
            reasoning_paths_per_question: List of reasoning path lists (one per question)
            return_details: Whether to return detailed results for each question

        Returns:
            Dictionary containing evaluation results
        """
        if len(questions) != len(reasoning_paths_per_question):
            raise ValueError("Number of questions must match number of reasoning path lists")

        results = {
            "base_results": {},
            "weighted_results": {},
            "summary": {},
        }

        if return_details:
            results["detailed_results"] = []

        # Evaluate each question
        for q_idx, (question, paths) in enumerate(zip(questions, reasoning_paths_per_question)):
            question_results = {
                "question": question,
                "path_count": len(paths),
                "base_method_results": {},
                "weighted_method_results": {}
            }

            # Evaluate with base methods
            for method in self.base_methods:
                method_name = method.name
                path_confidences = [method.evaluate_path(question, path) for path in paths]
                avg_confidence = sum(path_confidences) / len(path_confidences) if path_confidences else 0.0

                question_results["base_method_results"][method_name] = {
                    "individual_confidences": path_confidences,
                    "average_confidence": avg_confidence
                }

                if method_name not in results["base_results"]:
                    results["base_results"][method_name] = []
                results["base_results"][method_name].append(avg_confidence)

            # Evaluate with weighted adapters
            for adapter in self.weighted_adapters:
                method_name = adapter.base_method.name + "_weighted"
                candidates = adapter.evaluate_paths_with_aggregation(question, paths)

                best_confidence = candidates[0].aggregated_confidence if candidates else 0.0
                answer_diversity = len(candidates)

                question_results["weighted_method_results"][method_name] = {
                    "best_confidence": best_confidence,
                    "answer_diversity": answer_diversity,
                    "candidates": [str(candidate) for candidate in candidates]
                }

                if method_name not in results["weighted_results"]:
                    results["weighted_results"][method_name] = []
                results["weighted_results"][method_name].append(best_confidence)

            if return_details:
                results["detailed_results"].append(question_results)

        # Compute summary statistics
        for method_name, confidences in results["base_results"].items():
            results["summary"][method_name + "_base"] = {
                "mean_confidence": sum(confidences) / len(confidences),
                "min_confidence": min(confidences),
                "max_confidence": max(confidences)
            }

        for method_name, confidences in results["weighted_results"].items():
            results["summary"][method_name] = {
                "mean_confidence": sum(confidences) / len(confidences),
                "min_confidence": min(confidences),
                "max_confidence": max(confidences)
            }

        return results


# Pre-configured adapters for common methods
def create_entropy_with_aggregation(**kwargs) -> WeightedAnswerAdapter:
    """Create EntropyBasedUQ with weighted answer aggregation."""
    try:
        from ..models.base import BaseLLM  # This might need adjustment based on your LLM setup
        # You'll need to provide an LLM instance
        # base_method = EntropyBasedUQ(llm=your_llm_instance)
        raise NotImplementedError("Please provide an LLM instance to create EntropyBasedUQ")
    except ImportError:
        raise ImportError("Required dependencies not available for EntropyBasedUQ")


def create_consistency_with_aggregation(**kwargs) -> WeightedAnswerAdapter:
    """Create ConsistencyBasedUQ with weighted answer aggregation."""
    try:
        from ..models.base import BaseLLM
        # base_method = ConsistencyBasedUQ(llm=your_llm_instance)
        raise NotImplementedError("Please provide an LLM instance to create ConsistencyBasedUQ")
    except ImportError:
        raise ImportError("Required dependencies not available for ConsistencyBasedUQ")


def create_coherence_with_aggregation(
    model_name: str = "all-MiniLM-L6-v2",
    coherence_method: str = "mean_cosine_similarity",
    **kwargs
) -> WeightedAnswerAdapter:
    """Create CoherenceBasedUQ with weighted answer aggregation."""
    base_method = CoherenceBasedUQ(
        model_name=model_name,
        coherence_method=coherence_method
    )
    return add_weighted_aggregation(base_method, **kwargs)


def create_relative_coherence_with_aggregation(
    model_name: str = "all-MiniLM-L6-v2",
    coherence_method: str = "arp_pair",
    **kwargs
) -> WeightedAnswerAdapter:
    """Create RelativeCoherenceBasedUQ with weighted answer aggregation."""
    base_method = RelativeCoherenceBasedUQ(
        model_name=model_name,
        coherence_method=coherence_method
    )
    return add_weighted_aggregation(base_method, **kwargs)


def create_random_with_aggregation(**kwargs) -> WeightedAnswerAdapter:
    """Create RandomBaselineUQ with weighted answer aggregation (for testing)."""
    base_method = RandomBaselineUQ()
    return add_weighted_aggregation(base_method, **kwargs)