"""
Evaluator for individual question results.
"""

import re
from typing import Dict, Any, List, Optional
from ..core.data_structures import ReasoningPath
from ..core.interfaces import UncertaintyMethod, SearchAlgorithm


class Evaluator:
    """Evaluates reasoning paths against ground truth answers."""

    def __init__(self):
        self.answer_extractors = {
            "numeric": self._extract_numeric_answer,
            "text": self._extract_text_answer,
            "equation": self._extract_equation_answer,
        }

    def evaluate_question(
        self,
        question: Dict[str, Any],
        predicted_path: ReasoningPath,
        uncertainty_method: UncertaintyMethod,
        search_algorithm: SearchAlgorithm,
        structured_output: bool,
    ) -> Dict[str, Any]:
        """
        Evaluate a single question result.

        Args:
            question: Question data with ground truth
            predicted_path: Predicted reasoning path
            uncertainty_method: Uncertainty method used
            search_algorithm: Search algorithm used
            structured_output: If using the standard Math structured output

        Returns:
            Dictionary containing evaluation results
        """
        # Extract predicted answer from reasoning path
        if search_algorithm.name == "best_of_n_cot":
            # in this structured output case, path and answers were packed in a tuple
            predicted_path, predicted_answer = predicted_path
        else:
            predicted_answer = self._extract_answer_from_path(predicted_path)

        # Get and parse ground truth answer
        raw_ground_truth = question.get("answer", "")
        ground_truth_answer = self._extract_ground_truth_answer(raw_ground_truth)

        # Check correctness
        is_correct = self._check_correctness(predicted_answer, ground_truth_answer)

        # Get confidence score
        confidence = predicted_path.total_confidence

        # Analyze reasoning path
        path_analysis = self._analyze_reasoning_path(predicted_path)

        # Create result dictionary
        result = {
            "question_id": question.get("id", -1),
            "question": question["question"],
            "ground_truth": ground_truth_answer,  # Use the parsed answer
            "predicted_answer": predicted_answer,
            "correct": is_correct,
            "confidence": confidence,
            "path_length": len(predicted_path.steps),
            "reasoning_path": [step.content for step in predicted_path.steps],
            "uncertainty_method": uncertainty_method.name,
            "search_algorithm": search_algorithm.name,
            "success": True,
            **path_analysis,
        }

        return result

    def _extract_final_answer_key(self, text: str) -> Optional[str]:
        """
        Extracts the answer from a 'FINAL_ANSWER:' key. This is the most reliable method.
        """
        # Regex to find "FINAL_ANSWER:", followed by optional symbols and the answer
        match = re.search(r"final_answer\s*[:\s]*\$?\s*(.*)", text, re.IGNORECASE)
        if match:
            # Return the captured group, stripped of whitespace
            return match.group(1).strip()
        return None

    def _extract_ground_truth_answer(self, text: str) -> str:
        """
        Extracts the final answer from the ground truth string, marked by '####'.
        """
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
        if match:
            return match.group(1)
        # Fallback to the last numerical value if #### is not found
        numbers = re.findall(r"(-?[\d,]+\.?\d+)", text)
        if numbers:
            return numbers[-1]
        return text  # Fallback to the whole string if no pattern matches

    def _extract_answer_from_path(self, path: ReasoningPath) -> str:
        """Extract final answer from reasoning path."""
        if not path.steps:
            return ""

        # Iterate backwards through the steps to find the answer
        for step in reversed(path.steps):
            # **CORRECTED LOGIC**: Prioritize the explicit FINAL_ANSWER key
            final_answer = self._extract_final_answer_key(step.content)
            if final_answer:
                return final_answer

            # Fallback to other extraction methods if key is not found
            for extractor in self.answer_extractors.values():
                answer = extractor(step.content)
                if answer:
                    return answer

        # If no specific answer found, return content of last step as a last resort
        return path.steps[-1].content.strip()

    def _extract_numeric_answer(self, text: str) -> Optional[str]:
        """Extract numeric answer from text."""
        # Look for patterns like "answer is 5", "x = 5", "= 5"
        patterns = [
            r"(?:the final answer is|the answer is|answer is|result is|solution is|equals?)\s*[:\s]*\$?\s*([+-]?[\d,]*\.?\d+)",
            r"([a-z])\s*=\s*\$?\s*([+-]?[\d,]*\.?\d+)",
            r"=\s*\$?\s*([+-]?[\d,]*\.?\d+)",
            r"([+-]?[\d,]*\.?\d+)\s*(?:dollars?|cents?|\$)",
            r"\$\s*([+-]?[\d,]*\.?\d+)",
        ]

        text_lower = text.lower()

        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Return the last match (most likely to be the final answer)
                match = matches[-1]
                if isinstance(match, tuple):
                    # Return the numeric part
                    return match[-1]
                return match

        # Look for standalone numbers in the last line as a weaker signal
        last_line = text.strip().split("\n")[-1]
        numeric_matches = re.findall(r"([+-]?[\d,]*\.?\d+)", last_line)
        if len(numeric_matches) == 1:  # Only if there's one unambiguous number
            return numeric_matches[0]

        return None

    def _extract_text_answer(self, text: str) -> Optional[str]:
        """Extract text-based answer."""
        patterns = [
            r"(?:answer is|result is|solution is|conclusion)\s*:?\s*(.+?)(?:\.|$)",
            r"(?:therefore|thus|so)\s*,?\s*(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_equation_answer(self, text: str) -> Optional[str]:
        """Extract mathematical equation as answer."""
        equation_patterns = [
            r"([a-z]\s*=\s*[^=]+)",
            r"(\d+\s*[+\-*/]\s*\d+\s*=\s*\d+)",
        ]

        for pattern in equation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].strip()

        return None

    def _check_correctness(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        if not predicted or not ground_truth:
            return False

        predicted_norm = self._normalize_answer(predicted)
        ground_truth_norm = self._normalize_answer(ground_truth)

        if not predicted_norm or not ground_truth_norm:
            return False

        # Exact match
        if predicted_norm == ground_truth_norm:
            return True

        # Numeric comparison with tolerance
        try:
            pred_num = float(predicted_norm)
            gt_num = float(ground_truth_norm)
            # Check for relative and absolute tolerance
            return abs(pred_num - gt_num) < 1e-5
        except (ValueError, TypeError):
            # If conversion to float fails, proceed to string comparison
            pass

        return False

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer string for comparison."""
        if not isinstance(answer, str):
            return ""

        normalized = answer.strip().lower()

        # Remove common punctuation, including commas in numbers
        normalized = re.sub(r"[.,;:!?]", "", normalized)

        # Remove currency symbols
        normalized = re.sub(r"[\$€£¥]", "", normalized)

        # Remove units (rough heuristic)
        normalized = re.sub(
            r"\s*(dollars?|cents?|meters?|feet|inches?|hours?|minutes?|seconds?)",
            "",
            normalized,
        )

        # Handle trailing text like "per year" or ".00"
        normalized = re.sub(r"\s*per\s*\w+", "", normalized)
        normalized = re.sub(r"\.0+$", "", normalized)

        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    # The analysis methods below remain unchanged as they were not the source of the issue.
    def _analyze_reasoning_path(self, path: ReasoningPath) -> Dict[str, Any]:
        """Analyze the reasoning path for additional metrics."""
        if not path.steps:
            return {}

        analysis = {}

        step_confidences = [step.confidence for step in path.steps]
        analysis["confidence_stats"] = {
            "mean": sum(step_confidences) / len(step_confidences),
            "min": min(step_confidences),
            "max": max(step_confidences),
            "std": (
                self._calculate_std(step_confidences)
                if len(step_confidences) > 1
                else 0.0
            ),
        }

        analysis["coherence_score"] = self._calculate_coherence(path)
        analysis["mathematical_content"] = self._analyze_mathematical_content(path)
        analysis["dependency_stats"] = self._analyze_dependencies(path)

        return analysis

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def _calculate_coherence(self, path: ReasoningPath) -> float:
        """Calculate reasoning coherence score."""
        if len(path.steps) <= 1:
            return 1.0

        coherence_score = 0.0
        total_pairs = len(path.steps) - 1

        for i in range(len(path.steps) - 1):
            current_step = path.steps[i]
            next_step = path.steps[i + 1]
            coherence_score += self._calculate_step_similarity(
                current_step.content, next_step.content
            )

        return coherence_score / total_pairs if total_pairs > 0 else 0.0

    def _calculate_step_similarity(self, step1: str, step2: str) -> float:
        """Calculate similarity between two steps."""
        words1 = set(re.findall(r"\w+", step1.lower()))
        words2 = set(re.findall(r"\w+", step2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _analyze_mathematical_content(self, path: ReasoningPath) -> Dict[str, Any]:
        """Analyze mathematical content in reasoning path."""
        all_content = " ".join(step.content for step in path.steps)

        operations = {
            "addition": len(re.findall(r"\+", all_content)),
            "subtraction": len(re.findall(r"-", all_content)),
            "multiplication": len(re.findall(r"\*|×", all_content)),
            "division": len(re.findall(r"/", all_content)),
            "equals": len(re.findall(r"=", all_content)),
        }

        equations = len(re.findall(r"[a-z]\s*=\s*[^=]+", all_content, re.IGNORECASE))
        variables = len(set(re.findall(r"\b[a-z]\b", all_content.lower())))

        return {
            "operations": operations,
            "num_equations": equations,
            "num_variables": variables,
            "total_math_operations": sum(operations.values()),
        }

    def _analyze_dependencies(self, path: ReasoningPath) -> Dict[str, Any]:
        """Analyze step dependencies."""
        if not path.steps:
            return {}

        steps_with_deps = sum(1 for step in path.steps if step.dependencies)
        steps_without_deps = len(path.steps) - steps_with_deps

        total_deps = sum(len(step.dependencies) for step in path.steps)
        avg_deps = total_deps / len(path.steps)

        return {
            "steps_with_dependencies": steps_with_deps,
            "steps_without_dependencies": steps_without_deps,
            "average_dependencies_per_step": avg_deps,
            "total_dependencies": total_deps,
        }
