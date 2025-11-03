"""
Example demonstrating token probability-based uncertainty quantification.

This script shows how to use the perplexity-based, mean log probability, and
token entropy-based uncertainty quantification methods with different LLM backends.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bmuq.models.huggingface_llm import HuggingFaceLLM
from bmuq.models.openai_llm import OpenAILLM
from bmuq.models.vllm_llm import VLLMLLM
from bmuq.uncertainty.token_probability_uq import (
    PerplexityBasedUQ,
    MeanLogProbUQ,
    TokenEntropyBasedUQ,
    create_perplexity_uq,
    create_mean_log_prob_uq,
    create_token_entropy_uq,
)
from bmuq.core.data_structures import ReasoningStep


def example_with_huggingface():
    """Example using HuggingFace model."""
    print("\n" + "=" * 80)
    print("Example 1: Token Probability-based UQ with HuggingFace Model")
    print("=" * 80)

    # Initialize HuggingFace model (using a small model for quick testing)
    print("\nInitializing HuggingFace model...")
    llm = HuggingFaceLLM(
        model_name="gpt2",  # Small model for testing
        device="auto",
        temperature=0.7,
        max_new_tokens=100,
    )

    # Create uncertainty quantification methods
    perplexity_uq = create_perplexity_uq(llm, temperature=0.0)
    mean_log_prob_uq = create_mean_log_prob_uq(llm, temperature=0.0)
    token_entropy_uq = create_token_entropy_uq(llm, temperature=0.0)

    # Example problem
    question = "What is 15 * 24?"

    # Create initial reasoning steps
    reasoning_path = [
        ReasoningStep(step_id=1, content="Let me break down this multiplication"),
    ]

    step_to_evaluate = ReasoningStep(
        step_id=2, content="15 * 24 can be calculated as (15 * 20) + (15 * 4)"
    )

    # Evaluate with different methods
    print(f"\nQuestion: {question}")
    print("\nEvaluating reasoning step with different UQ methods...")

    # Perplexity-based
    print("\n1. Perplexity-based UQ:")
    perplexity_score = perplexity_uq.evaluate_step(
        question, reasoning_path, step_to_evaluate
    )
    print(f"   Confidence: {perplexity_score.value:.3f}")
    print(f"   Perplexity: {perplexity_score.metadata.get('perplexity', 'N/A'):.2f}")
    print(
        f"   Mean log prob: {perplexity_score.metadata.get('mean_log_prob', 'N/A'):.3f}"
    )

    # Mean log probability-based
    print("\n2. Mean Log Probability UQ:")
    mean_log_prob_score = mean_log_prob_uq.evaluate_step(
        question, reasoning_path, step_to_evaluate
    )
    print(f"   Confidence: {mean_log_prob_score.value:.3f}")
    print(
        f"   Mean log prob: {mean_log_prob_score.metadata.get('mean_log_prob', 'N/A'):.3f}"
    )

    # Token entropy-based
    print("\n3. Token Entropy-based UQ:")
    token_entropy_score = token_entropy_uq.evaluate_step(
        question, reasoning_path, step_to_evaluate
    )
    print(f"   Confidence: {token_entropy_score.value:.3f}")
    print(
        f"   Mean entropy: {token_entropy_score.metadata.get('mean_entropy', 'N/A'):.3f}"
    )


def example_with_openai():
    """Example using OpenAI model."""
    print("\n" + "=" * 80)
    print("Example 2: Token Probability-based UQ with OpenAI Model")
    print("=" * 80)

    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("\nSkipping OpenAI example: OPENAI_API_KEY environment variable not set.")
        return

    print("\nInitializing OpenAI model...")
    try:
        llm = OpenAILLM(model="gpt-3.5-turbo", temperature=0.7)

        # Create perplexity-based UQ method
        perplexity_uq = create_perplexity_uq(llm, temperature=0.0)

        # Example problem
        question = "What is 15 * 24?"

        reasoning_path = [
            ReasoningStep(step_id=1, content="Let me break down this multiplication"),
        ]

        step_to_evaluate = ReasoningStep(
            step_id=2, content="15 * 24 can be calculated as (15 * 20) + (15 * 4)"
        )

        print(f"\nQuestion: {question}")
        print("\nEvaluating with Perplexity-based UQ...")

        score = perplexity_uq.evaluate_step(question, reasoning_path, step_to_evaluate)
        print(f"   Confidence: {score.value:.3f}")
        print(f"   Perplexity: {score.metadata.get('perplexity', 'N/A'):.2f}")
        print(f"   Mean log prob: {score.metadata.get('mean_log_prob', 'N/A'):.3f}")

    except Exception as e:
        print(f"\nError with OpenAI example: {e}")


def example_with_vllm():
    """Test log probability extraction directly."""
    print("\n" + "=" * 80)
    print("Example 3: Testing Log Probability Extraction with VLLM")
    print("=" * 80)
    if not os.getenv("HF_TOKEN"):
        print(
            "\nSkipping VLLM example: Huggingface Token environment variable not set."
        )
        return

    print("\nInitializing HuggingFace model...")
    llm = VLLMLLM(model="meta-llama/Meta-Llama-3-8B", top_logprobs=100, temperature=0.7)

    # Test prompt
    prompt = "What is 2 + 2?"

    print(f"\nPrompt: {prompt}")
    print("\nGenerating with log probabilities...")

    result = llm.generate_with_log_probs(prompt, max_tokens=30, temperature=0.0)

    print(f"\nGenerated text: {result['text']}")
    print(f"\nNumber of tokens: {len(result['tokens'])}")

    if result["log_probs"]:
        print("\nFirst 5 tokens with log probabilities:")
        for i, (token, log_prob) in enumerate(
            zip(result["tokens"][:5], result["log_probs"][:5])
        ):
            prob = 2.71828**log_prob  # exp(log_prob)
            print(
                f"  {i+1}. Token: '{token}' | Log prob: {log_prob:.3f} | Prob: {prob:.3f}"
            )

        # Compute and display perplexity
        import numpy as np

        perplexity = np.exp(-np.mean(result["log_probs"]))
        print(f"\nPerplexity: {perplexity:.2f}")
    else:
        print("\nNo log probabilities available (model may not support it)")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Token Probability-Based Uncertainty Quantification Examples")
    print("=" * 80)

    # Run examples
    try:
        example_with_huggingface()
    except Exception as e:
        print(f"\nError in HuggingFace example: {e}")

    try:
        example_with_openai()
    except Exception as e:
        print(f"\nError in OpenAI example: {e}")

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
