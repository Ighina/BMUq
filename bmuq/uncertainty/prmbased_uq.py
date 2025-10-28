"""
Process Reward Model (PRM) based uncertainty quantification.

This module implements uncertainty quantification using Process Reward Models
that score reasoning steps as positive, neutral, or negative. The PRM provides
fine-grained uncertainty estimates by evaluating each step in a reasoning chain.
"""

import torch
from typing import List, Optional, Dict, Any, Literal
import numpy as np

from ..core.interfaces import UncertaintyMethod
from ..core.data_structures import ReasoningStep, ReasoningPath, UncertaintyScore
from ..prm.inference import InferenceBertForTokenClassificationWithEmbeddings


class PRMBasedUQ(UncertaintyMethod):
    """
    PRM-based uncertainty quantification using a trained Process Reward Model.

    This method uses a BERT-based model fine-tuned to classify reasoning steps
    into three categories: negative, neutral, and positive. The uncertainty score
    is derived from the model's probability distribution over these classes.

    The PRM model accepts lists of strings representing reasoning steps and outputs
    probability distributions for each step. These probabilities are then converted
    into uncertainty scores using configurable scoring strategies.
    """

    def __init__(
        self,
        pretrained_model_path: str,
        featurizer_model: str = "sentence-transformers/all-mpnet-base-v2",
        scoring_method: Literal["validity", "redundancy", "positive_only"] = "validity",
        device: Optional[str] = None,
        batch_size: int = 1,
    ):
        """
        Initialize PRM-based uncertainty quantification.

        Args:
            pretrained_model_path: Path to the pretrained PRM model directory
            featurizer_model: Name or path of the sentence transformer model used for embeddings
            scoring_method: Method for computing uncertainty scores from PRM outputs:
                - "validity": Score = p_neutral + p_positive (default)
                - "redundancy": Score = p_neutral (measures step redundancy)
                - "positive_only": Score = p_positive (only positive evidence)
            device: Device to run the model on ('cpu', 'cuda', 'mps'). If None, auto-detect.
            batch_size: Batch size for processing (currently only supports 1)
        """
        super().__init__("prm_based")

        self.pretrained_model_path = pretrained_model_path
        self.featurizer_model = featurizer_model
        self.scoring_method = scoring_method
        self.batch_size = batch_size

        # Initialize the PRM model
        self.model = InferenceBertForTokenClassificationWithEmbeddings(
            pretrained_model=pretrained_model_path,
            featurizer_model=featurizer_model,
        )

        # Set device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()

    def _compute_score_from_probs(self, probs: List[float]) -> float:
        """
        Convert PRM probability distribution to uncertainty score.

        Args:
            probs: List of 3 probabilities [p_negative, p_neutral, p_positive]

        Returns:
            Uncertainty score between 0 and 1
        """
        p_negative, p_neutral, p_positive = probs

        if self.scoring_method == "validity":
            # Validity score: probability that step is neutral or positive
            score = p_neutral + p_positive
        elif self.scoring_method == "redundancy":
            # Redundancy score: probability that step is neutral
            score = p_neutral
        elif self.scoring_method == "positive_only":
            # Positive-only score: probability that step is positive
            score = p_positive
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")

        return float(score)

    def _score_steps(
        self, question: str, steps: List[str]
    ) -> List[float]:
        """
        Score multiple reasoning steps using the PRM model.

        Args:
            question: Original question
            steps: List of reasoning step contents

        Returns:
            List of uncertainty scores for each step
        """
        # Prepare input: [question, step1, step2, ...]
        input_texts = [question] + steps

        try:
            # Get probabilities from PRM model
            # Model returns shape: [batch_size, num_steps, num_classes]
            probs = self.model(inputs_text=input_texts)

            # probs is a list of shape [batch_size][num_tokens][num_classes]
            # We expect batch_size=1 for now
            if isinstance(probs, list) and len(probs) > 0:
                step_probs = probs[0]  # Get first batch
            else:
                step_probs = probs

            # Convert probabilities to scores
            # Skip the first token (corresponds to the question)
            step_scores = []
            for i in range(1, len(step_probs)):  # Start from 1 to skip question
                score = self._compute_score_from_probs(step_probs[i])
                step_scores.append(score)

            return step_scores

        except Exception as e:
            # Return neutral scores on error
            return [0.5] * len(steps)

    def evaluate_step(
        self,
        question: str,
        reasoning_path: List[ReasoningStep],
        step_to_evaluate: ReasoningStep,
    ) -> UncertaintyScore:
        """
        Evaluate uncertainty for a single reasoning step using the PRM model.

        The PRM evaluates the step in the context of the full reasoning chain
        up to that point, providing a score based on the learned patterns of
        valid reasoning.

        Args:
            question: Original question being solved
            reasoning_path: Context steps before the step to evaluate
            step_to_evaluate: The step to evaluate

        Returns:
            UncertaintyScore with PRM-based confidence
        """
        try:
            # Collect all steps up to and including the one to evaluate
            all_steps = reasoning_path + [step_to_evaluate]
            step_contents = [step.content for step in all_steps]

            # Score all steps
            scores = self._score_steps(question, step_contents)

            # Get score for the target step (last one)
            target_score = scores[-1] if scores else 0.5

            # Get PRM probabilities for metadata
            input_texts = [question] + step_contents
            probs = self.model(inputs_text=input_texts)

            if isinstance(probs, list) and len(probs) > 0:
                step_probs = probs[0]
                target_probs = step_probs[-1] if len(step_probs) > len(step_contents) - 1 else [0.33, 0.33, 0.34]
            else:
                target_probs = [0.33, 0.33, 0.34]

            metadata = {
                "prm_probabilities": {
                    "negative": float(target_probs[0]),
                    "neutral": float(target_probs[1]),
                    "positive": float(target_probs[2]),
                },
                "scoring_method": self.scoring_method,
                "step_position": len(all_steps),
                "context_length": len(reasoning_path),
            }

            return UncertaintyScore(
                value=target_score,
                method=self.name,
                metadata=metadata,
            )

        except Exception as e:
            return UncertaintyScore(
                value=0.5,
                method=self.name,
                metadata={"error": str(e), "note": "Error during PRM evaluation"},
            )

    def evaluate_path(self, question: str, reasoning_path: ReasoningPath) -> float:
        """
        Evaluate uncertainty for an entire reasoning path.

        This method scores all steps in the path and aggregates them using
        different strategies depending on the scoring method.

        Args:
            question: Original question being solved
            reasoning_path: Complete reasoning path to evaluate

        Returns:
            Overall confidence score for the path
        """
        if not reasoning_path.steps:
            return 0.0

        try:
            # Score all steps in the path
            step_contents = [step.content for step in reasoning_path.steps]
            step_scores = self._score_steps(question, step_contents)

            if not step_scores:
                return 0.0

            # Store scores in the reasoning steps
            for step, score in zip(reasoning_path.steps, step_scores):
                if self.name not in step.uncertainty_scores:
                    step.uncertainty_scores[self.name] = UncertaintyScore(
                        value=score,
                        method=self.name,
                        metadata={"from_path_evaluation": True},
                    )

            # Aggregate scores based on scoring method
            if self.scoring_method == "validity":
                # For validity, use minimum (weakest link approach)
                # A chain is only as strong as its weakest step
                path_score = min(step_scores)
            elif self.scoring_method == "redundancy":
                # For redundancy, use maximum
                # Flag if any step is redundant
                path_score = max(step_scores)
            else:  # positive_only
                # For positive-only, use mean
                path_score = sum(step_scores) / len(step_scores)

            return float(path_score)

        except Exception as e:
            return 0.5

    def batch_evaluate_steps(
        self,
        question: str,
        reasoning_path: List[ReasoningStep],
        steps_to_evaluate: List[ReasoningStep],
    ) -> List[UncertaintyScore]:
        """
        Efficiently evaluate multiple steps at once using the PRM model.

        This is more efficient than calling evaluate_step multiple times
        as it processes all steps in a single forward pass.

        Args:
            question: Original question being solved
            reasoning_path: Context steps
            steps_to_evaluate: List of steps to evaluate

        Returns:
            List of UncertaintyScores for each step
        """
        if not steps_to_evaluate:
            return []

        # Collect all steps
        all_steps = reasoning_path + steps_to_evaluate
        step_contents = [step.content for step in all_steps]

        # Score all steps at once
        scores = self._score_steps(question, step_contents)

        # Get probabilities for metadata
        try:
            input_texts = [question] + step_contents
            probs = self.model(inputs_text=input_texts)

            if isinstance(probs, list) and len(probs) > 0:
                step_probs = probs[0]
            else:
                step_probs = [[0.33, 0.33, 0.34]] * len(step_contents)
        except:
            step_probs = [[0.33, 0.33, 0.34]] * len(step_contents)

        # Extract scores for the steps to evaluate
        start_idx = len(reasoning_path)
        results = []

        for i, step in enumerate(steps_to_evaluate):
            step_idx = start_idx + i
            score = scores[step_idx] if step_idx < len(scores) else 0.5

            if step_idx < len(step_probs):
                target_probs = step_probs[step_idx]
            else:
                target_probs = [0.33, 0.33, 0.34]

            metadata = {
                "prm_probabilities": {
                    "negative": float(target_probs[0]),
                    "neutral": float(target_probs[1]),
                    "positive": float(target_probs[2]),
                },
                "scoring_method": self.scoring_method,
                "step_position": step_idx + 1,
                "batch_evaluation": True,
            }

            results.append(
                UncertaintyScore(
                    value=score,
                    method=self.name,
                    metadata=metadata,
                )
            )

        return results

    def get_step_probabilities(
        self, question: str, steps: List[str]
    ) -> List[Dict[str, float]]:
        """
        Get raw probability distributions for each step.

        This is useful for detailed analysis and debugging.

        Args:
            question: Original question
            steps: List of reasoning step contents

        Returns:
            List of probability dictionaries for each step
        """
        input_texts = [question] + steps

        try:
            probs = self.model(inputs_text=input_texts)

            if isinstance(probs, list) and len(probs) > 0:
                step_probs = probs[0]
            else:
                step_probs = probs

            # Skip the first token (question) and return probabilities
            results = []
            for i in range(1, len(step_probs)):
                results.append({
                    "negative": float(step_probs[i][0]),
                    "neutral": float(step_probs[i][1]),
                    "positive": float(step_probs[i][2]),
                })

            return results

        except Exception as e:
            # Return uniform distributions on error
            return [{"negative": 0.33, "neutral": 0.33, "positive": 0.34}] * len(steps)

    def get_method_info(self) -> Dict[str, Any]:
        """Get information about this PRM-based uncertainty method."""
        info = super().get_method_info()
        info.update({
            "model_path": self.pretrained_model_path,
            "featurizer_model": self.featurizer_model,
            "scoring_method": self.scoring_method,
            "device": self.device,
        })
        return info
