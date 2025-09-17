"""
Baseline uncertainty quantification methods for comparison with SelfCheck.
"""

import math
import re
from typing import List, Dict, Optional, Any
from collections import Counter
from ..core.interfaces import UncertaintyMethod
from ..core.data_structures import ReasoningStep, ReasoningPath, UncertaintyScore
from ..models.base import BaseLLM


class EntropyBasedUQ(UncertaintyMethod):
    """
    Entropy-based uncertainty quantification using multiple generations.
    
    Generates multiple completions for each step and measures uncertainty
    based on the diversity of responses.
    """

    def __init__(self, llm: BaseLLM, num_samples: int = 5, temperature: float = 0.8):
        """
        Initialize entropy-based UQ method.
        
        Args:
            llm: Language model for generation
            num_samples: Number of samples to generate for entropy calculation
            temperature: Temperature for diverse generation
        """
        super().__init__("entropy_based")
        self.llm = llm
        self.num_samples = num_samples
        self.temperature = temperature

    def evaluate_step(self, question: str, reasoning_path: List[ReasoningStep], 
                     step_to_evaluate: ReasoningStep) -> UncertaintyScore:
        """
        Evaluate step uncertainty using response diversity.
        
        Args:
            question: Original question
            reasoning_path: Context steps
            step_to_evaluate: Step to evaluate
            
        Returns:
            UncertaintyScore with entropy-based confidence
        """
        try:
            # Generate multiple alternative completions
            alternatives = self._generate_alternatives(question, reasoning_path, step_to_evaluate)
            
            # Calculate entropy based on response diversity
            entropy = self._calculate_response_entropy(alternatives)
            
            # Convert entropy to confidence (higher entropy = lower confidence)
            confidence = self._entropy_to_confidence(entropy)
            
            metadata = {
                "alternatives": alternatives,
                "entropy": entropy,
                "num_samples": len(alternatives),
                "temperature": self.temperature
            }
            
            return UncertaintyScore(
                value=confidence,
                method=self.name,
                metadata=metadata
            )
            
        except Exception as e:
            return UncertaintyScore(
                value=0.5,
                method=self.name,
                metadata={"error": str(e)}
            )

    def evaluate_path(self, question: str, reasoning_path: ReasoningPath) -> float:
        """
        Evaluate path uncertainty by averaging step entropies.
        
        Args:
            question: Original question
            reasoning_path: Complete reasoning path
            
        Returns:
            Average confidence across all steps
        """
        if not reasoning_path.steps:
            return 0.0
        
        confidences = []
        for i, step in enumerate(reasoning_path.steps):
            if self.name in step.uncertainty_scores:
                confidences.append(step.uncertainty_scores[self.name].value)
            else:
                # Evaluate step if not already done
                context = reasoning_path.steps[:i]
                score = self.evaluate_step(question, context, step)
                step.uncertainty_scores[self.name] = score
                confidences.append(score.value)
        
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _generate_alternatives(self, question: str, reasoning_path: List[ReasoningStep],
                             target_step: ReasoningStep) -> List[str]:
        """Generate alternative step completions."""
        context_text = "\n".join([f"Step {s.step_id}: {s.content}" for s in reasoning_path])
        
        prompt = f"""Problem: {question}

Current progress:
{context_text}

Continue with the next logical step. Focus on making progress toward solving the problem.

Next step:"""

        alternatives = []
        for _ in range(self.num_samples):
            try:
                response = self.llm.generate(prompt, max_tokens=150, temperature=self.temperature)
                cleaned_response = self._clean_response(response)
                if cleaned_response and cleaned_response not in alternatives:
                    alternatives.append(cleaned_response)
            except Exception:
                continue
        
        return alternatives

    def _clean_response(self, response: str) -> str:
        """Clean LLM response to extract step content."""
        cleaned = response.strip()
        if cleaned.lower().startswith("step"):
            colon_idx = cleaned.find(":")
            if colon_idx != -1:
                cleaned = cleaned[colon_idx + 1:].strip()
        return cleaned

    def _calculate_response_entropy(self, responses: List[str]) -> float:
        """Calculate entropy based on response similarity."""
        if len(responses) <= 1:
            return 0.0
        
        # Tokenize responses into word sets
        response_signatures = []
        for response in responses:
            words = set(word.lower() for word in re.findall(r'\w+', response))
            response_signatures.append(frozenset(words))
        
        # Count unique signatures
        signature_counts = Counter(response_signatures)
        
        # Calculate entropy
        total_responses = len(responses)
        entropy = 0.0
        for count in signature_counts.values():
            probability = count / total_responses
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(total_responses)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _entropy_to_confidence(self, entropy: float) -> float:
        """Convert entropy to confidence score."""
        # High entropy (diverse responses) = low confidence
        # Low entropy (similar responses) = high confidence
        return max(0.1, 1.0 - entropy)


class ConsistencyBasedUQ(UncertaintyMethod):
    """
    Consistency-based uncertainty quantification.
    
    Evaluates uncertainty by checking consistency with previous steps
    and mathematical principles.
    """

    def __init__(self, llm: BaseLLM):
        """
        Initialize consistency-based UQ method.
        
        Args:
            llm: Language model for consistency checking
        """
        super().__init__("consistency_based")
        self.llm = llm

    def evaluate_step(self, question: str, reasoning_path: List[ReasoningStep],
                     step_to_evaluate: ReasoningStep) -> UncertaintyScore:
        """
        Evaluate step uncertainty based on consistency checks.
        
        Args:
            question: Original question
            reasoning_path: Context steps
            step_to_evaluate: Step to evaluate
            
        Returns:
            UncertaintyScore with consistency-based confidence
        """
        try:
            consistency_scores = []
            
            # Check mathematical consistency
            math_consistency = self._check_mathematical_consistency(
                question, reasoning_path, step_to_evaluate
            )
            consistency_scores.append(("mathematical", math_consistency))
            
            # Check logical consistency with previous steps
            if reasoning_path:
                logical_consistency = self._check_logical_consistency(
                    reasoning_path, step_to_evaluate
                )
                consistency_scores.append(("logical", logical_consistency))
            
            # Check goal alignment
            goal_alignment = self._check_goal_alignment(question, step_to_evaluate)
            consistency_scores.append(("goal_alignment", goal_alignment))
            
            # Aggregate scores
            overall_score = sum(score for _, score in consistency_scores) / len(consistency_scores)
            
            metadata = {
                "consistency_checks": dict(consistency_scores),
                "overall_consistency": overall_score
            }
            
            return UncertaintyScore(
                value=overall_score,
                method=self.name,
                metadata=metadata
            )
            
        except Exception as e:
            return UncertaintyScore(
                value=0.5,
                method=self.name,
                metadata={"error": str(e)}
            )

    def evaluate_path(self, question: str, reasoning_path: ReasoningPath) -> float:
        """Evaluate path consistency by checking overall coherence."""
        if not reasoning_path.steps:
            return 0.0
        
        # Check individual step consistency
        step_scores = []
        for i, step in enumerate(reasoning_path.steps):
            if self.name in step.uncertainty_scores:
                step_scores.append(step.uncertainty_scores[self.name].value)
            else:
                context = reasoning_path.steps[:i]
                score = self.evaluate_step(question, context, step)
                step.uncertainty_scores[self.name] = score
                step_scores.append(score.value)
        
        # Check overall path coherence
        path_coherence = self._check_path_coherence(reasoning_path)
        
        # Combine step-level and path-level scores
        avg_step_score = sum(step_scores) / len(step_scores)
        combined_score = (avg_step_score * 0.7) + (path_coherence * 0.3)
        
        return combined_score

    def _check_mathematical_consistency(self, question: str, reasoning_path: List[ReasoningStep],
                                      step: ReasoningStep) -> float:
        """Check mathematical consistency of the step."""
        context_text = "\n".join([f"Step {s.step_id}: {s.content}" for s in reasoning_path])
        
        prompt = f"""Problem: {question}

Previous steps:
{context_text}

Current step: {step.content}

Is this step mathematically sound and consistent with the previous work? 
Rate the mathematical consistency from 0.0 (completely inconsistent) to 1.0 (perfectly consistent).
Consider:
- Are calculations correct?
- Are mathematical operations valid?
- Does it follow logically from previous steps?

Provide only a numerical score between 0.0 and 1.0:"""

        try:
            response = self.llm.generate(prompt, max_tokens=50)
            # Extract numerical score
            score_match = re.search(r'(\d*\.?\d+)', response)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))
            else:
                return 0.5
        except Exception:
            return 0.5

    def _check_logical_consistency(self, reasoning_path: List[ReasoningStep], 
                                 step: ReasoningStep) -> float:
        """Check logical consistency with previous steps."""
        if not reasoning_path:
            return 1.0
        
        # Simple heuristic: check for contradictory statements
        step_content = step.content.lower()
        
        consistency_score = 1.0
        for prev_step in reasoning_path[-3:]:  # Check last 3 steps
            prev_content = prev_step.content.lower()
            
            # Look for contradictory patterns
            if self._has_contradiction(step_content, prev_content):
                consistency_score -= 0.3
        
        return max(0.1, consistency_score)

    def _check_goal_alignment(self, question: str, step: ReasoningStep) -> float:
        """Check if step is aligned with solving the original question."""
        question_lower = question.lower()
        step_content = step.content.lower()
        
        # Extract key terms from question
        question_terms = set(re.findall(r'\w+', question_lower))
        step_terms = set(re.findall(r'\w+', step_content))
        
        # Check overlap
        overlap = len(question_terms.intersection(step_terms))
        total_terms = len(question_terms.union(step_terms))
        
        if total_terms == 0:
            return 0.5
        
        similarity = overlap / total_terms
        
        # Boost score if step contains progress indicators
        progress_indicators = ['solve', 'find', 'calculate', 'answer', 'result', 'therefore']
        if any(indicator in step_content for indicator in progress_indicators):
            similarity += 0.2
        
        return min(1.0, similarity)

    def _has_contradiction(self, content1: str, content2: str) -> bool:
        """Check for contradictory statements between two step contents."""
        # Simple pattern matching for contradictions
        contradiction_patterns = [
            (r'(\w+)\s*=\s*(\d+)', r'\1\s*=\s*(?!\2)\d+'),  # Different values for same variable
            (r'not\s+', r'(?<!not\s)'),  # Negation vs affirmation
        ]
        
        for pattern1, pattern2 in contradiction_patterns:
            matches1 = re.findall(pattern1, content1)
            if matches1:
                if re.search(pattern2, content2):
                    return True
        
        return False

    def _check_path_coherence(self, reasoning_path: ReasoningPath) -> float:
        """Check overall coherence of the reasoning path."""
        if len(reasoning_path.steps) <= 1:
            return 1.0
        
        coherence_score = 1.0
        
        # Check that steps build on each other
        for i in range(1, len(reasoning_path.steps)):
            current_step = reasoning_path.steps[i]
            prev_steps = reasoning_path.steps[:i]
            
            # Simple heuristic: each step should relate to at least one previous step
            has_connection = False
            for prev_step in prev_steps[-2:]:  # Check last 2 steps
                if self._steps_are_connected(prev_step, current_step):
                    has_connection = True
                    break
            
            if not has_connection:
                coherence_score -= 0.2
        
        return max(0.1, coherence_score)

    def _steps_are_connected(self, step1: ReasoningStep, step2: ReasoningStep) -> bool:
        """Check if two steps are logically connected."""
        content1 = step1.content.lower()
        content2 = step2.content.lower()
        
        # Look for shared mathematical terms or concepts
        math_terms = ['equation', 'solve', 'substitute', 'calculate', 'value', 'result']
        
        shared_math_terms = sum(1 for term in math_terms 
                               if term in content1 and term in content2)
        
        if shared_math_terms > 0:
            return True
        
        # Check for variable continuity (e.g., "x = 5" followed by step using x)
        variables1 = set(re.findall(r'\b[a-z]\b', content1))
        variables2 = set(re.findall(r'\b[a-z]\b', content2))
        
        return len(variables1.intersection(variables2)) > 0


class RandomBaselineUQ(UncertaintyMethod):
    """
    Random baseline for uncertainty quantification.
    
    Provides random confidence scores for comparison purposes.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random baseline.
        
        Args:
            seed: Random seed for reproducibility
        """
        super().__init__("random_baseline")
        import random
        if seed is not None:
            random.seed(seed)
        self.random = random

    def evaluate_step(self, question: str, reasoning_path: List[ReasoningStep],
                     step_to_evaluate: ReasoningStep) -> UncertaintyScore:
        """Return random uncertainty score."""
        confidence = self.random.uniform(0.1, 0.9)
        
        return UncertaintyScore(
            value=confidence,
            method=self.name,
            metadata={"note": "Random baseline for comparison"}
        )

    def evaluate_path(self, question: str, reasoning_path: ReasoningPath) -> float:
        """Return random path confidence."""
        return self.random.uniform(0.1, 0.9)