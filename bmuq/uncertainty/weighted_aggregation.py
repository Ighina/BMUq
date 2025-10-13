"""
Weighted answer aggregation system for uncertainty quantification methods.

This module provides classes to aggregate confidence scores for identical answers
across different reasoning paths, allowing uncertainty methods to weight final
answers based on their accumulated confidence.
"""

import re
from typing import List, Dict, Optional, Any, Callable, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.data_structures import ReasoningStep, ReasoningPath, UncertaintyScore
from ..core.interfaces import UncertaintyMethod


@dataclass
class AnswerCandidate:
    """Represents a candidate answer with its associated reasoning paths and confidence."""
    answer: str
    paths: List[ReasoningPath]
    individual_confidences: List[float]
    aggregated_confidence: float
    metadata: Dict[str, Any]

    @property
    def support_count(self) -> int:
        """Number of reasoning paths supporting this answer."""
        return len(self.paths)

    def __str__(self) -> str:
        return f"Answer: {self.answer} (confidence: {self.aggregated_confidence:.3f}, paths: {self.support_count})"


class AnswerExtractor(ABC):
    """Abstract base class for extracting answers from reasoning paths."""

    @abstractmethod
    def extract_answer(self, reasoning_path: ReasoningPath) -> Optional[str]:
        """
        Extract the final answer from a reasoning path.

        Args:
            reasoning_path: The reasoning path to extract answer from

        Returns:
            Extracted answer string, or None if no answer found
        """
        pass

    @abstractmethod
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize an answer for comparison (e.g., remove formatting).

        Args:
            answer: Raw answer string

        Returns:
            Normalized answer string
        """
        pass


class MathAnswerExtractor(AnswerExtractor):
    """Answer extractor specifically designed for mathematical reasoning problems."""

    def __init__(self):
        # Common patterns for mathematical answers
        self.answer_patterns = [
            r'(?:the answer is|answer:|final answer:?\s*)\s*([^\n.]+)',
            r'(?:therefore|thus|so),?\s*([^.]+)\s*(?:is the answer|is our answer)',
            r'(?:=\s*)?(\d+(?:\.\d+)?)\s*(?:$|\n|\.)',
            r'(?:x\s*=\s*)?(-?\d+(?:\.\d+)?)\s*(?:$|\n|\.)',
        ]

    def extract_answer(self, reasoning_path: ReasoningPath) -> Optional[str]:
        """Extract mathematical answer from the final steps of reasoning path."""
        if not reasoning_path.steps:
            return None

        # Check the last few steps for answers
        steps_to_check = reasoning_path.steps[-3:] if len(reasoning_path.steps) >= 3 else reasoning_path.steps

        for step in reversed(steps_to_check):
            content = step.content.lower().strip()

            # Try each pattern
            for pattern in self.answer_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    raw_answer = match.group(1).strip()
                    return self.normalize_answer(raw_answer)

        # Fallback: look for standalone numbers in the last step
        last_step = reasoning_path.steps[-1].content
        numbers = re.findall(r'-?\d+(?:\.\d+)?', last_step)
        if numbers:
            return self.normalize_answer(numbers[-1])

        return None

    def normalize_answer(self, answer: str) -> str:
        """Normalize mathematical answer for comparison."""
        # Remove common prefixes/suffixes
        answer = re.sub(r'^(the\s+)?answer\s+is\s+', '', answer.lower()).strip()
        answer = re.sub(r'\.$', '', answer)

        # Try to convert to number format
        try:
            # Handle fractions
            if '/' in answer:
                parts = answer.split('/')
                if len(parts) == 2:
                    num, denom = float(parts[0].strip()), float(parts[1].strip())
                    if denom != 0:
                        return str(num / denom)

            # Convert to float and back to remove unnecessary decimals
            float_val = float(answer)
            if float_val.is_integer():
                return str(int(float_val))
            else:
                return str(float_val)
        except ValueError:
            # Return as-is if not a number
            return answer.strip()


class GenericAnswerExtractor(AnswerExtractor):
    """Generic answer extractor for various types of questions."""

    def __init__(self, answer_indicators: Optional[List[str]] = None):
        """
        Initialize generic extractor.

        Args:
            answer_indicators: List of phrases that typically precede answers
        """
        self.answer_indicators = answer_indicators or [
            "the answer is", "answer:", "final answer:", "therefore", "thus", "so", "in conclusion"
        ]

    def extract_answer(self, reasoning_path: ReasoningPath) -> Optional[str]:
        """Extract answer using generic patterns."""
        if not reasoning_path.steps:
            return None

        # Check the last few steps
        steps_to_check = reasoning_path.steps[-2:] if len(reasoning_path.steps) >= 2 else reasoning_path.steps

        for step in reversed(steps_to_check):
            content = step.content.strip()
            content_lower = content.lower()

            # Look for answer indicators
            for indicator in self.answer_indicators:
                if indicator in content_lower:
                    # Extract text after the indicator
                    idx = content_lower.find(indicator)
                    after_indicator = content[idx + len(indicator):].strip()

                    # Clean up the extracted answer
                    answer = re.split(r'[.!?\n]', after_indicator)[0].strip()
                    if answer:
                        return self.normalize_answer(answer)

        # Fallback: return the last step content (truncated)
        last_content = reasoning_path.steps[-1].content.strip()
        if len(last_content) > 100:
            last_content = last_content[:100] + "..."
        return self.normalize_answer(last_content)

    def normalize_answer(self, answer: str) -> str:
        """Basic normalization for generic answers."""
        # Remove common prefixes
        for prefix in ["is ", ":", "="]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        return answer.strip()


class WeightedAnswerAggregator:
    """
    Aggregates answers from multiple reasoning paths with weighted confidence scores.

    This class takes multiple reasoning paths that may lead to different answers,
    extracts the answers, groups them by similarity, and computes weighted
    confidence scores based on the uncertainty scores of supporting paths.
    """

    def __init__(
        self,
        answer_extractor: AnswerExtractor,
        confidence_aggregation: str = "sum",
        min_confidence_threshold: float = 0.1,
        answer_similarity_threshold: float = 0.9
    ):
        """
        Initialize weighted answer aggregator.

        Args:
            answer_extractor: Method to extract answers from reasoning paths
            confidence_aggregation: How to aggregate confidences ("sum", "mean", "weighted_mean", "max")
            min_confidence_threshold: Minimum confidence to consider a path
            answer_similarity_threshold: Threshold for considering answers as identical
        """
        self.answer_extractor = answer_extractor
        self.confidence_aggregation = confidence_aggregation
        self.min_confidence_threshold = min_confidence_threshold
        self.answer_similarity_threshold = answer_similarity_threshold

        # Validate aggregation method
        valid_methods = {"sum", "mean", "weighted_mean", "max"}
        if confidence_aggregation not in valid_methods:
            raise ValueError(f"Invalid aggregation method: {confidence_aggregation}. Must be one of {valid_methods}")

    def aggregate_answers(
        self,
        reasoning_paths: List[ReasoningPath],
        uncertainty_method: str = "selfcheck"
    ) -> List[AnswerCandidate]:
        """
        Aggregate answers from multiple reasoning paths.

        Args:
            reasoning_paths: List of reasoning paths to aggregate
            uncertainty_method: Name of uncertainty method to use for confidence scores

        Returns:
            List of AnswerCandidate objects sorted by aggregated confidence
        """
        if not reasoning_paths:
            return []

        # Extract answers and confidences from paths
        answer_groups = defaultdict(list)

        for path in reasoning_paths:
            # Extract answer
            answer = self.answer_extractor.extract_answer(path)
            if not answer:
                continue

            # Get path confidence
            confidence = self._get_path_confidence(path, uncertainty_method)
            if confidence < self.min_confidence_threshold:
                continue

            # Group by normalized answer
            normalized_answer = self.answer_extractor.normalize_answer(answer)
            answer_groups[normalized_answer].append((path, confidence, answer))

        # Create answer candidates
        candidates = []
        for normalized_answer, path_data in answer_groups.items():
            paths = [data[0] for data in path_data]
            confidences = [data[1] for data in path_data]
            original_answers = [data[2] for data in path_data]

            # Aggregate confidence
            aggregated_confidence = self._aggregate_confidences(confidences, paths)

            # Use the most common original answer format
            display_answer = max(set(original_answers), key=original_answers.count)

            candidate = AnswerCandidate(
                answer=display_answer,
                paths=paths,
                individual_confidences=confidences,
                aggregated_confidence=aggregated_confidence,
                metadata={
                    "normalized_answer": normalized_answer,
                    "uncertainty_method": uncertainty_method,
                    "aggregation_method": self.confidence_aggregation,
                    "path_count": len(paths),
                    "confidence_stats": {
                        "mean": sum(confidences) / len(confidences),
                        "min": min(confidences),
                        "max": max(confidences),
                        "std": self._calculate_std(confidences) if len(confidences) > 1 else 0.0
                    }
                }
            )
            candidates.append(candidate)

        # Sort by aggregated confidence (descending)
        candidates.sort(key=lambda x: x.aggregated_confidence, reverse=True)
        return candidates

    def get_best_answer(
        self,
        reasoning_paths: List[ReasoningPath],
        uncertainty_method: str = "selfcheck"
    ) -> Optional[AnswerCandidate]:
        """
        Get the best answer (highest aggregated confidence) from reasoning paths.

        Args:
            reasoning_paths: List of reasoning paths
            uncertainty_method: Name of uncertainty method to use

        Returns:
            Best answer candidate, or None if no valid answers found
        """
        candidates = self.aggregate_answers(reasoning_paths, uncertainty_method)
        return candidates[0] if candidates else None

    def _get_path_confidence(self, path: ReasoningPath, uncertainty_method: str) -> float:
        """Extract confidence score from a reasoning path."""
        if hasattr(path, 'total_confidence') and path.total_confidence > 0:
            return path.total_confidence

        # Fallback: average step confidences for the specified method
        confidences = []
        for step in path.steps:
            if uncertainty_method in step.uncertainty_scores:
                confidences.append(step.uncertainty_scores[uncertainty_method].value)
            elif step.uncertainty_scores:
                # Use any available uncertainty score
                confidences.append(list(step.uncertainty_scores.values())[0].value)

        return sum(confidences) / len(confidences) if confidences else 0.5

    def _aggregate_confidences(self, confidences: List[float], paths: List[ReasoningPath]) -> float:
        """Aggregate confidence scores using the specified method."""
        if not confidences:
            return 0.0

        if self.confidence_aggregation == "sum":
            return min(1.0, sum(confidences))  # Cap at 1.0

        elif self.confidence_aggregation == "mean":
            return sum(confidences) / len(confidences)

        elif self.confidence_aggregation == "weighted_mean":
            # Weight by path length (longer paths get higher weight)
            weights = [len(path.steps) for path in paths]
            total_weight = sum(weights)
            if total_weight == 0:
                return sum(confidences) / len(confidences)
            weighted_sum = sum(c * w for c, w in zip(confidences, weights))
            return weighted_sum / total_weight

        elif self.confidence_aggregation == "max":
            return max(confidences)

        else:
            return sum(confidences) / len(confidences)  # Default to mean

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


class WeightedUncertaintyMethod:
    """
    Wrapper that combines any uncertainty method with weighted answer aggregation.

    This class allows you to plug weighted answer aggregation on top of existing
    uncertainty methods, providing a unified interface that returns aggregated
    confidence scores for final answers.
    """

    def __init__(
        self,
        base_uncertainty_method: UncertaintyMethod,
        answer_extractor: AnswerExtractor,
        aggregator: Optional[WeightedAnswerAggregator] = None
    ):
        """
        Initialize weighted uncertainty method.

        Args:
            base_uncertainty_method: The underlying uncertainty method
            answer_extractor: Method to extract answers from reasoning paths
            aggregator: Custom aggregator (uses default if None)
        """
        self.base_method = base_uncertainty_method
        self.answer_extractor = answer_extractor
        self.aggregator = aggregator or WeightedAnswerAggregator(
            answer_extractor=answer_extractor,
            confidence_aggregation="sum"
        )

    def evaluate_multiple_paths(
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
            List of AnswerCandidate objects ranked by confidence
        """
        # First, evaluate each path with the base uncertainty method
        for path in reasoning_paths:
            path_confidence = self.base_method.evaluate_path(question, path)
            path.total_confidence = path_confidence

            # Also evaluate individual steps if needed
            for i, step in enumerate(path.steps):
                if self.base_method.name not in step.uncertainty_scores:
                    context_steps = path.steps[:i]
                    step_score = self.base_method.evaluate_step(question, context_steps, step)
                    step.uncertainty_scores[self.base_method.name] = step_score

        # Aggregate answers using the aggregator
        return self.aggregator.aggregate_answers(reasoning_paths, self.base_method.name)

    def get_best_weighted_answer(
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
            Best answer candidate with aggregated confidence
        """
        candidates = self.evaluate_multiple_paths(question, reasoning_paths)
        return candidates[0] if candidates else None


# Factory functions for common use cases
def create_math_weighted_method(
    base_uncertainty_method: UncertaintyMethod,
    confidence_aggregation: str = "sum"
) -> WeightedUncertaintyMethod:
    """
    Create a weighted uncertainty method optimized for mathematical problems.

    Args:
        base_uncertainty_method: The underlying uncertainty method
        confidence_aggregation: How to aggregate confidences

    Returns:
        Configured WeightedUncertaintyMethod for math problems
    """
    extractor = MathAnswerExtractor()
    aggregator = WeightedAnswerAggregator(
        answer_extractor=extractor,
        confidence_aggregation=confidence_aggregation
    )
    return WeightedUncertaintyMethod(base_uncertainty_method, extractor, aggregator)


def create_generic_weighted_method(
    base_uncertainty_method: UncertaintyMethod,
    answer_indicators: Optional[List[str]] = None,
    confidence_aggregation: str = "sum"
) -> WeightedUncertaintyMethod:
    """
    Create a weighted uncertainty method for general questions.

    Args:
        base_uncertainty_method: The underlying uncertainty method
        answer_indicators: Custom answer indicators
        confidence_aggregation: How to aggregate confidences

    Returns:
        Configured WeightedUncertaintyMethod for general questions
    """
    extractor = GenericAnswerExtractor(answer_indicators)
    aggregator = WeightedAnswerAggregator(
        answer_extractor=extractor,
        confidence_aggregation=confidence_aggregation
    )
    return WeightedUncertaintyMethod(base_uncertainty_method, extractor, aggregator)