"""
Token probability-based uncertainty quantification methods.

This module implements uncertainty quantification methods that use token
log probabilities from language models to estimate uncertainty. Methods include:
- Perplexity-based uncertainty
- Token entropy-based uncertainty
- Mean log probability confidence
"""

import math
import numpy as np
from typing import List, Dict, Optional, Any
from ..core.interfaces import UncertaintyMethod
from ..core.data_structures import ReasoningStep, ReasoningPath, UncertaintyScore
from ..models.base import BaseLLM


class PerplexityBasedUQ(UncertaintyMethod):
    """
    Perplexity-based uncertainty quantification using token log probabilities.

    Perplexity measures how well a language model predicts a sequence. Lower
    perplexity indicates higher confidence in the generated text. This method
    computes perplexity from the token log probabilities and converts it to
    a confidence score.

    The perplexity is computed as:
        PPL = exp(-1/N * sum(log P(token_i)))

    where N is the number of tokens and P(token_i) is the probability of token i.
    """

    def __init__(
        self,
        llm: BaseLLM,
        temperature: float = 0.0,
        normalize_by_length: bool = True,
        perplexity_threshold: float = 100.0,
    ):
        """
        Initialize perplexity-based UQ method.

        Args:
            llm: Language model for generation with log probability support
            temperature: Temperature for generation (lower = more deterministic)
            normalize_by_length: Whether to normalize perplexity by sequence length
            perplexity_threshold: Maximum perplexity value for normalization
        """
        super().__init__("perplexity_based")
        self.llm = llm
        self.temperature = temperature
        self.normalize_by_length = normalize_by_length
        self.perplexity_threshold = perplexity_threshold

    def evaluate_step(
        self,
        question: str,
        reasoning_path: List[ReasoningStep],
        step_to_evaluate: ReasoningStep,
    ) -> UncertaintyScore:
        """
        Evaluate step uncertainty using perplexity.

        Args:
            question: Original question
            reasoning_path: Context steps
            step_to_evaluate: Step to evaluate

        Returns:
            UncertaintyScore with perplexity-based confidence
        """
        try:
            # Build prompt for step generation
            context_text = "\n".join(
                [f"Step {s.step_id}: {s.content}" for s in reasoning_path]
            )

            prompt = f"""Problem: {question}

Current progress:
{context_text}

Continue with the next logical step. Focus on making progress toward solving the problem.

Next step:"""

            # Generate with log probabilities
            result = self.llm.generate_with_log_probs(
                prompt, temperature=self.temperature
            )

            log_probs = result.get("log_probs", [])
            tokens = result.get("tokens", [])
            generated_text = result.get("text", "")

            if not log_probs:
                # Model doesn't support log probs, return neutral confidence
                return UncertaintyScore(
                    value=0.5,
                    method=self.name,
                    metadata={
                        "error": "Log probabilities not available from model",
                        "generated_text": generated_text,
                    },
                )

            # Compute perplexity
            perplexity = self._compute_perplexity(log_probs)

            # Compute additional metrics
            mean_log_prob = np.mean(log_probs) if log_probs else 0.0
            token_entropy = self._compute_token_entropy(result.get("top_log_probs", []))

            # Convert perplexity to confidence score
            confidence = self._perplexity_to_confidence(perplexity)

            metadata = {
                "perplexity": perplexity,
                "mean_log_prob": mean_log_prob,
                "token_entropy": token_entropy,
                "num_tokens": len(tokens),
                "generated_text": generated_text,
                "tokens": tokens[:10],  # First 10 tokens for inspection
            }

            return UncertaintyScore(
                value=confidence, method=self.name, metadata=metadata
            )

        except Exception as e:
            return UncertaintyScore(
                value=0.5, method=self.name, metadata={"error": str(e)}
            )

    def evaluate_path(self, question: str, reasoning_path: ReasoningPath) -> float:
        """
        Evaluate path uncertainty by averaging step perplexities.

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

    def _compute_perplexity(self, log_probs: List[float]) -> float:
        """
        Compute perplexity from log probabilities.

        Args:
            log_probs: List of log probabilities for each token

        Returns:
            Perplexity value
        """
        if not log_probs:
            return float("inf")

        # Perplexity = exp(-1/N * sum(log_probs))
        avg_neg_log_prob = -np.mean(log_probs)
        perplexity = np.exp(avg_neg_log_prob)

        return float(perplexity)

    def _compute_token_entropy(self, top_log_probs: List[Dict[str, float]]) -> float:
        """
        Compute average entropy across token distributions.

        Args:
            top_log_probs: List of dicts mapping tokens to log probs

        Returns:
            Average entropy value
        """
        if not top_log_probs:
            return 0.0

        entropies = []
        for token_dist in top_log_probs:
            if not token_dist:
                continue

            # Convert log probs to probs
            log_probs = list(token_dist.values())
            probs = np.exp(log_probs)
            probs = probs / np.sum(probs)  # Normalize

            # Compute entropy: -sum(p * log(p))
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

        return float(np.mean(entropies)) if entropies else 0.0

    def _perplexity_to_confidence(self, perplexity: float) -> float:
        """
        Convert perplexity to confidence score.

        Lower perplexity = higher confidence
        Uses a sigmoid-like mapping to convert perplexity to [0, 1]

        Args:
            perplexity: Perplexity value

        Returns:
            Confidence score in [0, 1]
        """
        if math.isinf(perplexity) or math.isnan(perplexity):
            return 0.1

        # Normalize perplexity by threshold
        normalized_ppl = perplexity / self.perplexity_threshold

        # Use inverse sigmoid mapping: confidence = 1 / (1 + normalized_ppl)
        # This maps low perplexity -> high confidence
        confidence = 1.0 / (1.0 + normalized_ppl)

        # Ensure confidence is in valid range
        return max(0.1, min(1.0, confidence))


class MeanLogProbUQ(UncertaintyMethod):
    """
    Mean log probability-based uncertainty quantification.

    This method uses the mean log probability of generated tokens directly
    as a confidence measure. Higher mean log probability indicates higher
    confidence in the generation.
    """

    def __init__(self, llm: BaseLLM, temperature: float = 0.0):
        """
        Initialize mean log prob UQ method.

        Args:
            llm: Language model for generation with log probability support
            temperature: Temperature for generation
        """
        super().__init__("mean_log_prob")
        self.llm = llm
        self.temperature = temperature

    def evaluate_step(
        self,
        question: str,
        reasoning_path: List[ReasoningStep],
        step_to_evaluate: ReasoningStep,
    ) -> UncertaintyScore:
        """
        Evaluate step uncertainty using mean log probability.

        Args:
            question: Original question
            reasoning_path: Context steps
            step_to_evaluate: Step to evaluate

        Returns:
            UncertaintyScore with mean log prob confidence
        """
        try:
            # Build prompt
            context_text = "\n".join(
                [f"Step {s.step_id}: {s.content}" for s in reasoning_path]
            )

            prompt = f"""Problem: {question}

Current progress:
{context_text}

Continue with the next logical step. Focus on making progress toward solving the problem.

Next step:"""

            # Generate with log probabilities
            result = self.llm.generate_with_log_probs(
                prompt, temperature=self.temperature
            )

            log_probs = result.get("log_probs", [])
            tokens = result.get("tokens", [])
            generated_text = result.get("text", "")

            if not log_probs:
                return UncertaintyScore(
                    value=0.5,
                    method=self.name,
                    metadata={
                        "error": "Log probabilities not available",
                        "generated_text": generated_text,
                    },
                )

            # Compute mean log probability
            mean_log_prob = np.mean(log_probs)

            # Convert to confidence (log probs are typically negative)
            # Map from [-inf, 0] to [0, 1]
            # Using: confidence = exp(mean_log_prob)
            confidence = float(np.exp(mean_log_prob))

            # Ensure confidence is in valid range
            confidence = max(0.1, min(1.0, confidence))

            metadata = {
                "mean_log_prob": float(mean_log_prob),
                "min_log_prob": float(np.min(log_probs)),
                "max_log_prob": float(np.max(log_probs)),
                "std_log_prob": float(np.std(log_probs)),
                "num_tokens": len(tokens),
                "generated_text": generated_text,
            }

            return UncertaintyScore(
                value=confidence, method=self.name, metadata=metadata
            )

        except Exception as e:
            return UncertaintyScore(
                value=0.5, method=self.name, metadata={"error": str(e)}
            )

    def evaluate_path(self, question: str, reasoning_path: ReasoningPath) -> float:
        """
        Evaluate path uncertainty by averaging step confidences.

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
                context = reasoning_path.steps[:i]
                score = self.evaluate_step(question, context, step)
                step.uncertainty_scores[self.name] = score
                confidences.append(score.value)

        return sum(confidences) / len(confidences) if confidences else 0.0


class TokenEntropyBasedUQ(UncertaintyMethod):
    """
    Token entropy-based uncertainty quantification.

    This method uses the average entropy of the token probability distributions
    at each position as an uncertainty measure. Higher entropy indicates more
    uncertainty in token selection.
    """

    def __init__(self, llm: BaseLLM, temperature: float = 0.0):
        """
        Initialize token entropy UQ method.

        Args:
            llm: Language model for generation with log probability support
            temperature: Temperature for generation
        """
        super().__init__("token_entropy")
        self.llm = llm
        self.temperature = temperature

    def evaluate_step(
        self,
        question: str,
        reasoning_path: List[ReasoningStep],
        step_to_evaluate: ReasoningStep,
    ) -> UncertaintyScore:
        """
        Evaluate step uncertainty using token entropy.

        Args:
            question: Original question
            reasoning_path: Context steps
            step_to_evaluate: Step to evaluate

        Returns:
            UncertaintyScore with entropy-based confidence
        """
        try:
            # Build prompt
            context_text = "\n".join(
                [f"Step {s.step_id}: {s.content}" for s in reasoning_path]
            )

            prompt = f"""Problem: {question}

Current progress:
{context_text}

Continue with the next logical step. Focus on making progress toward solving the problem.

Next step:"""

            # Generate with log probabilities
            result = self.llm.generate_with_log_probs(
                prompt, temperature=self.temperature
            )

            top_log_probs = result.get("top_log_probs", [])
            generated_text = result.get("text", "")

            if not top_log_probs:
                return UncertaintyScore(
                    value=0.5,
                    method=self.name,
                    metadata={
                        "error": "Top log probabilities not available",
                        "generated_text": generated_text,
                    },
                )

            # Compute average token entropy
            entropies = []
            for token_dist in top_log_probs:
                if not token_dist:
                    continue

                # Convert log probs to probs
                log_probs = list(token_dist.values())
                probs = np.exp(log_probs)
                probs = probs / np.sum(probs)  # Normalize

                # Compute entropy
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)

            if not entropies:
                return UncertaintyScore(
                    value=0.5,
                    method=self.name,
                    metadata={
                        "error": "Could not compute token entropies",
                        "generated_text": generated_text,
                    },
                )

            mean_entropy = float(np.mean(entropies))
            max_entropy = float(np.log(5))  # Max entropy for 5 tokens

            # Convert entropy to confidence
            # High entropy = low confidence
            normalized_entropy = mean_entropy / max_entropy
            confidence = 1.0 - normalized_entropy

            # Ensure confidence is in valid range
            confidence = max(0.1, min(1.0, confidence))

            metadata = {
                "mean_entropy": mean_entropy,
                "min_entropy": float(np.min(entropies)),
                "max_entropy": float(np.max(entropies)),
                "std_entropy": float(np.std(entropies)),
                "num_tokens": len(top_log_probs),
                "generated_text": generated_text,
            }

            return UncertaintyScore(
                value=confidence, method=self.name, metadata=metadata
            )

        except Exception as e:
            return UncertaintyScore(
                value=0.5, method=self.name, metadata={"error": str(e)}
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
                context = reasoning_path.steps[:i]
                score = self.evaluate_step(question, context, step)
                step.uncertainty_scores[self.name] = score
                confidences.append(score.value)

        return sum(confidences) / len(confidences) if confidences else 0.0


# Factory functions for easy instantiation
def create_perplexity_uq(
    llm: BaseLLM,
    temperature: float = 0.0,
    perplexity_threshold: float = 100.0,
) -> PerplexityBasedUQ:
    """
    Create a perplexity-based UQ method.

    Args:
        llm: Language model with log probability support
        temperature: Generation temperature
        perplexity_threshold: Threshold for perplexity normalization

    Returns:
        PerplexityBasedUQ instance
    """
    return PerplexityBasedUQ(
        llm=llm, temperature=temperature, perplexity_threshold=perplexity_threshold
    )


def create_mean_log_prob_uq(llm: BaseLLM, temperature: float = 0.0) -> MeanLogProbUQ:
    """
    Create a mean log probability UQ method.

    Args:
        llm: Language model with log probability support
        temperature: Generation temperature

    Returns:
        MeanLogProbUQ instance
    """
    return MeanLogProbUQ(llm=llm, temperature=temperature)


def create_token_entropy_uq(
    llm: BaseLLM, temperature: float = 0.0
) -> TokenEntropyBasedUQ:
    """
    Create a token entropy-based UQ method.

    Args:
        llm: Language model with log probability support
        temperature: Generation temperature

    Returns:
        TokenEntropyBasedUQ instance
    """
    return TokenEntropyBasedUQ(llm=llm, temperature=temperature)
