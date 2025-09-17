"""
SelfCheck uncertainty quantification method implementation.

Based on the paper: "SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning"
"""

import re
from typing import List, Dict, Optional, Any
from ..core.interfaces import UncertaintyMethod
from ..core.data_structures import ReasoningStep, ReasoningPath, UncertaintyScore, ComparisonResult
from ..models.base import BaseLLM


class SelfCheck(UncertaintyMethod):
    """
    Implements the SelfCheck verification method from the paper.
    
    SelfCheck performs 4-stage verification:
    1. Extract the target of the current step
    2. Collect information that the current step depends on
    3. Regenerate the step independently 
    4. Compare original and regenerated steps
    """

    def __init__(self, llm: BaseLLM, lambda_neg1: float = 1.0, lambda_0: float = 0.3):
        """
        Initialize SelfCheck method.
        
        Args:
            llm: Language model for verification
            lambda_neg1: Weight for contradictory evidence in integration function
            lambda_0: Weight for uncertain evidence in integration function
        """
        super().__init__("selfcheck")
        self.llm = llm
        self.lambda_neg1 = lambda_neg1
        self.lambda_0 = lambda_0

    def evaluate_step(self, question: str, reasoning_path: List[ReasoningStep], 
                     step_to_evaluate: ReasoningStep) -> UncertaintyScore:
        """
        Perform complete SelfCheck evaluation on a single step.
        
        Args:
            question: Original question being solved
            reasoning_path: Reasoning path context (steps before the one being evaluated)
            step_to_evaluate: Step to evaluate uncertainty for
            
        Returns:
            UncertaintyScore with SelfCheck confidence and metadata
        """
        try:
            # Stage 1: Extract target
            target = self.extract_target(question, reasoning_path, step_to_evaluate)
            
            # Stage 2: Collect information
            dep_indices = self.collect_information(question, reasoning_path, step_to_evaluate)
            dependencies = [s for s in reasoning_path if s.step_id in dep_indices]
            
            # Stage 3: Regenerate step
            regenerated = self.regenerate_step(question, target, dependencies)
            
            # Stage 4: Compare results
            comparison = self.compare_results(step_to_evaluate.content, regenerated)
            
            # Convert comparison to confidence score
            confidence = self._comparison_to_confidence(comparison)
            
            # Create metadata
            metadata = {
                "target": target,
                "dependencies": dep_indices,
                "regenerated_step": regenerated,
                "comparison": comparison.value,
                "stage_1_target": target,
                "stage_2_dependencies": dep_indices,
                "stage_3_regeneration": regenerated,
                "stage_4_comparison": comparison.value
            }
            
            return UncertaintyScore(
                value=confidence,
                method=self.name,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error in SelfCheck evaluation: {e}")
            return UncertaintyScore(
                value=0.5,  # Neutral confidence on error
                method=self.name,
                metadata={"error": str(e)}
            )

    def evaluate_path(self, question: str, reasoning_path: ReasoningPath) -> float:
        """
        Evaluate uncertainty for entire reasoning path using SelfCheck integration.
        
        Args:
            question: Original question
            reasoning_path: Complete reasoning path
            
        Returns:
            Integrated confidence score for the path
        """
        if not reasoning_path.steps:
            return 0.0
            
        # Evaluate each step if not already done
        for i, step in enumerate(reasoning_path.steps):
            if self.name not in step.uncertainty_scores:
                context = reasoning_path.steps[:i]  # Previous steps as context
                score = self.evaluate_step(question, context, step)
                step.uncertainty_scores[self.name] = score
        
        # Apply SelfCheck integration function
        return self._integrate_step_confidences(reasoning_path)

    def extract_target(self, question: str, steps: List[ReasoningStep], current_step: ReasoningStep) -> str:
        """
        Stage 1: Extract the target/goal of the current step.
        
        Args:
            question: Original question
            steps: Previous reasoning steps
            current_step: Step to extract target for
            
        Returns:
            Description of what the step aims to achieve
        """
        steps_text = "\n".join([f"Step {s.step_id}: {s.content}" for s in steps])

        prompt = f"""The following is a part of the solution to the problem: {question}

{steps_text}

What specific action does the step "Step {current_step.step_id}: {current_step.content}" take?
Please give a brief answer using a single sentence and do not copy the steps."""

        return self.llm.generate(prompt)

    def collect_information(self, question: str, steps: List[ReasoningStep], current_step: ReasoningStep) -> List[int]:
        """
        Stage 2: Collect information that the current step depends on.
        
        Args:
            question: Original question
            steps: Previous reasoning steps  
            current_step: Step to find dependencies for
            
        Returns:
            List of step IDs that the current step depends on
        """
        # Split question into information chunks for better context
        question_parts = [s.strip() for s in question.split('.') if s.strip()]
        info_text = "\n".join([f"Information {i}: {part}" for i, part in enumerate(question_parts)])

        steps_text = "\n".join([f"Step {s.step_id}: {s.content}" for s in steps])

        prompt = f"""This is a math question: {question}

The following is information extracted from the question:
{info_text}

The following are the first few steps in a solution to the problem:
{steps_text}

Which previous steps or information does the next step "Step {current_step.step_id}: {current_step.content}" directly follow from?
Please identify the step numbers."""

        response = self.llm.generate(prompt)

        # Extract step numbers using regex
        step_matches = re.findall(r'Step (\d+)', response)
        return [int(match) for match in step_matches]

    def regenerate_step(self, question: str, target: str, dependencies: List[ReasoningStep]) -> str:
        """
        Stage 3: Regenerate the step independently based on target and dependencies.
        
        Args:
            question: Original question
            target: Target/goal of the step
            dependencies: Steps that this step depends on
            
        Returns:
            Independently regenerated step content
        """
        deps_text = "\n".join([f"Step {s.step_id}: {s.content}" for s in dependencies])

        prompt = f"""We are in the process of solving a math problem: {question}

The following are some previous steps:
{deps_text}

The target for the next step is: {target}

Please try to achieve the target with the information from previous steps.
Provide a clear, step-by-step solution."""

        return self.llm.generate(prompt)

    def compare_results(self, original_step: str, regenerated_step: str) -> ComparisonResult:
        """
        Stage 4: Compare original and regenerated steps.
        
        Args:
            original_step: Original step content
            regenerated_step: Independently regenerated step content
            
        Returns:
            ComparisonResult indicating relationship between steps
        """
        prompt = f"""The following are 2 solutions to a math problem:

Solution 1 (Regenerated): {regenerated_step}
Solution 2 (Original): {original_step}

Compare the key points from both solutions step by step and then check whether Solution 1 'supports', 'contradicts' or 'is not directly related to' the conclusion in Solution 2. 

Pay special attention to:
1. Numerical values and calculations
2. Logical reasoning steps
3. Mathematical operations
4. Final conclusions

Respond with exactly one of: supports, contradicts, not_directly_related"""

        response = self.llm.generate(prompt).lower().strip()

        if 'supports' in response:
            return ComparisonResult.SUPPORTS
        elif 'contradicts' in response:
            return ComparisonResult.CONTRADICTS
        else:
            return ComparisonResult.NOT_DIRECTLY_RELATED

    def _comparison_to_confidence(self, comparison: ComparisonResult) -> float:
        """Convert comparison result to confidence score."""
        mapping = {
            ComparisonResult.SUPPORTS: 0.9,
            ComparisonResult.CONTRADICTS: 0.1,
            ComparisonResult.NOT_DIRECTLY_RELATED: 0.5
        }
        return mapping[comparison]

    def _integrate_step_confidences(self, reasoning_path: ReasoningPath) -> float:
        """
        Integrate step-level confidences into path-level confidence using SelfCheck formula.
        
        Uses the integration function from the SelfCheck paper (Equation 1):
        score = -λ₋₁ * failed_checks - λ₀ * uncertain_checks
        confidence = 2 * sigmoid(score)
        """
        if not reasoning_path.steps:
            return 0.0

        # Convert step confidences to check results (-1, 0, 1)
        check_results = []
        for step in reasoning_path.steps:
            if self.name in step.uncertainty_scores:
                confidence = step.uncertainty_scores[self.name].value
                if confidence >= 0.8:
                    check_results.append(1)  # support
                elif confidence <= 0.3:
                    check_results.append(-1)  # contradict  
                else:
                    check_results.append(0)  # not directly related
            else:
                check_results.append(0)  # Default to uncertain

        # Apply SelfCheck integration function
        failed_checks = sum(1 for r in check_results if r == -1)
        uncertain_checks = sum(1 for r in check_results if r == 0)

        score = -self.lambda_neg1 * failed_checks - self.lambda_0 * uncertain_checks
        confidence = 2 * (1 / (1 + 1/abs(score) if score != 0 else 1))  # 2 * sigmoid, handle division by zero
        
        return max(0.0, min(1.0, confidence))  # Ensure confidence is in [0, 1]

    def batch_evaluate_steps(self, question: str, reasoning_path: List[ReasoningStep],
                            steps_to_evaluate: List[ReasoningStep]) -> List[UncertaintyScore]:
        """
        Evaluate multiple steps in batch for efficiency.
        
        Note: Current implementation is sequential. This could be optimized
        with batch LLM calls or parallel processing.
        """
        results = []
        for step in steps_to_evaluate:
            # Find the context for this step (all previous steps)
            context = [s for s in reasoning_path if s.step_id < step.step_id]
            result = self.evaluate_step(question, context, step)
            results.append(result)
        return results

    def get_method_info(self) -> Dict[str, Any]:
        """Get information about SelfCheck method configuration."""
        info = super().get_method_info()
        info.update({
            "lambda_neg1": self.lambda_neg1,
            "lambda_0": self.lambda_0,
            "stages": [
                "Extract target",
                "Collect information", 
                "Regenerate step",
                "Compare results"
            ],
            "llm_model": self.llm.model if hasattr(self.llm, 'model') else "unknown"
        })
        return info