"""
Base search algorithm implementation.
"""

from typing import List, Dict, Optional, Any
from ..core.interfaces import SearchAlgorithm, UncertaintyMethod
from ..core.data_structures import ReasoningStep, ReasoningPath
from ..models.base import BaseLLM


class BaseSearchAlgorithm(SearchAlgorithm):
    """
    Base implementation for search algorithms with common functionality.
    """

    def __init__(self, name: str, llm: BaseLLM, uncertainty_method: UncertaintyMethod,
                 max_depth: int = 8, completion_keywords: Optional[List[str]] = None,
                 structured_format: bool = True):
        """
        Initialize base search algorithm.

        Args:
            name: Algorithm name
            llm: Language model for generation
            uncertainty_method: Uncertainty quantification method
            max_depth: Maximum reasoning chain depth
            completion_keywords: Keywords that indicate solution completion (legacy)
            structured_format: Use structured FINAL_ANSWER format for completion detection
        """
        super().__init__(name, uncertainty_method)
        self.llm = llm
        self.max_depth = max_depth
        self.structured_format = structured_format

        # Default completion detection keywords (fallback)
        self.completion_keywords = completion_keywords or [
            'answer is', 'therefore', 'final answer', 'result is',
            'conclusion', 'solution is', 'the answer', 'equals'
        ]

    def generate_next_steps(self, question: str, current_path: ReasoningPath, 
                           num_candidates: int = 3) -> List[ReasoningStep]:
        """
        Generate candidate next steps for the current reasoning path.
        
        Args:
            question: Original question
            current_path: Current reasoning path
            num_candidates: Number of candidates to generate
            
        Returns:
            List of candidate next steps
        """
        steps_text = self._format_steps(current_path.steps)
        
        prompt = self._build_generation_prompt(question, steps_text, num_candidates)
        
        candidates = []
        for i in range(num_candidates):
            try:
                response = self.llm.generate(prompt, max_tokens=200)
                step_content = self._clean_step_response(response)
                
                next_step_id = len(current_path.steps)
                step = ReasoningStep(
                    step_id=next_step_id,
                    content=step_content,
                    dependencies=[],
                    metadata={"generation_attempt": i + 1}
                )
                candidates.append(step)
                
            except Exception as e:
                print(f"Error generating step candidate {i+1}: {e}")
                continue
        
        return candidates

    def is_solution_complete(self, question: str, path: ReasoningPath) -> bool:
        """
        Check if reasoning path provides a complete solution.

        Args:
            question: Original question
            path: Reasoning path to check

        Returns:
            True if solution appears complete
        """
        if not path.steps:
            return False

        if self.structured_format:
            # Check for structured FINAL_ANSWER format
            final_answer = self.extract_final_answer(path)
            return final_answer is not None
        else:
            # Fallback to legacy keyword-based detection
            return self._legacy_completion_check(path)

    def extract_final_answer(self, path: ReasoningPath) -> Optional[str]:
        """
        Extract final answer from structured format in reasoning path.

        Args:
            path: Reasoning path to extract answer from

        Returns:
            Final answer string if found, None otherwise
        """
        if not path.steps:
            return None

        # Check last few steps for FINAL_ANSWER marker
        steps_to_check = path.steps[-3:] if len(path.steps) >= 3 else path.steps

        import re
        final_answer_pattern = r'FINAL_ANSWER:\s*(.+?)(?:\n|$)'

        for step in reversed(steps_to_check):
            match = re.search(final_answer_pattern, step.content.strip(), re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _legacy_completion_check(self, path: ReasoningPath) -> bool:
        """Legacy keyword-based completion detection."""
        # Check last few steps for completion indicators
        last_steps = path.steps[-2:] if len(path.steps) >= 2 else path.steps

        for step in last_steps:
            step_content = step.content.lower()
            if any(keyword in step_content for keyword in self.completion_keywords):
                return True

        # Additional heuristics for mathematical problems
        if self._has_numerical_answer(path):
            return True

        return False

    def _format_steps(self, steps: List[ReasoningStep]) -> str:
        """Format reasoning steps for prompt inclusion."""
        if not steps:
            return "No previous steps."
        return "\n".join([f"Step {s.step_id}: {s.content}" for s in steps])

    def _build_generation_prompt(self, question: str, steps_text: str, num_candidates: int) -> str:
        """Build prompt for generating next reasoning steps."""
        if self.structured_format:
            return f"""Problem: {question}

Current solution so far:
{steps_text}

Continue solving this problem with the next logical step. Provide a clear, specific step that builds on the previous work.

Focus on:
1. Making concrete progress toward the solution
2. Using proper mathematical reasoning
3. Being specific with calculations or logical deductions
4. Building directly on previous steps

When you reach the final answer, format it as: FINAL_ANSWER: [your answer]

Next step:"""
        else:
            return f"""Problem: {question}

Current solution so far:
{steps_text}

Continue solving this problem with the next logical step. Provide a clear, specific step that builds on the previous work.

Focus on:
1. Making concrete progress toward the solution
2. Using proper mathematical reasoning
3. Being specific with calculations or logical deductions
4. Building directly on previous steps

Next step:"""

    def _clean_step_response(self, response: str) -> str:
        """Clean and format LLM response into step content."""
        # Remove step numbering if present
        cleaned = response.strip()
        if cleaned.lower().startswith("step"):
            # Remove "Step X:" pattern
            colon_idx = cleaned.find(":")
            if colon_idx != -1:
                cleaned = cleaned[colon_idx + 1:].strip()
        
        # Remove extra whitespace and ensure proper capitalization
        cleaned = " ".join(cleaned.split())
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]
            
        return cleaned

    def _has_numerical_answer(self, path: ReasoningPath) -> bool:
        """Check if path contains numerical answer indicating completion."""
        if not path.steps:
            return False
            
        last_step = path.steps[-1].content
        
        # Look for patterns like "x = 5", "answer = 10", etc.
        import re
        patterns = [
            r'x\s*=\s*-?\d+\.?\d*',
            r'answer\s*=\s*-?\d+\.?\d*', 
            r'result\s*=\s*-?\d+\.?\d*',
            r'solution\s*=\s*-?\d+\.?\d*'
        ]
        
        for pattern in patterns:
            if re.search(pattern, last_step.lower()):
                return True
                
        return False

    def evaluate_path_quality(self, question: str, path: ReasoningPath) -> Dict[str, float]:
        """
        Evaluate overall path quality using multiple metrics.
        
        Returns:
            Dictionary of quality metrics
        """
        # Evaluate uncertainty/confidence
        confidence = self.uncertainty_method.evaluate_path(question, path)
        
        # Evaluate completeness
        completeness = 1.0 if self.is_solution_complete(question, path) else 0.5
        
        # Length penalty (prefer shorter, more direct solutions)
        length_penalty = max(0.1, 1.0 - (len(path.steps) - 3) * 0.1)
        
        # Dependency coherence (steps should build on each other)
        coherence = self._evaluate_step_coherence(path)
        
        return {
            "confidence": confidence,
            "completeness": completeness,
            "length_penalty": length_penalty,
            "coherence": coherence,
            "overall": (confidence * 0.4 + completeness * 0.3 + 
                       length_penalty * 0.1 + coherence * 0.2)
        }

    def _evaluate_step_coherence(self, path: ReasoningPath) -> float:
        """Evaluate how well steps build on each other."""
        if len(path.steps) <= 1:
            return 1.0
            
        # Simple heuristic: check that steps have reasonable dependencies
        coherent_steps = 0
        total_steps = len(path.steps)
        
        for i, step in enumerate(path.steps):
            if i == 0:
                # First step should have no dependencies
                if not step.dependencies:
                    coherent_steps += 1
            else:
                # Later steps should reference earlier steps
                if step.dependencies and all(dep < step.step_id for dep in step.dependencies):
                    coherent_steps += 1
                elif not step.dependencies and i < total_steps - 1:
                    # Middle steps without dependencies are suspicious
                    continue
                else:
                    coherent_steps += 0.5  # Partial credit
        
        return coherent_steps / total_steps if total_steps > 0 else 0.0

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get detailed algorithm information."""
        info = super().get_algorithm_info()
        info.update({
            "max_depth": self.max_depth,
            "structured_format": self.structured_format,
            "completion_keywords": self.completion_keywords if not self.structured_format else "N/A (using structured format)",
            "llm_model": self.llm.model if hasattr(self.llm, 'model') else "unknown"
        })
        return info