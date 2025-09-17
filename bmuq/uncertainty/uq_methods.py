from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from ..core.interfaces import UncertaintyMethod
from ..core.data_structures import (
    ReasoningStep,
    ReasoningPath,
    UncertaintyScore,
)


class EntailmentChecker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise, hypothesis):
        inputs = self.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
        prediction = int(
            torch.argmax(torch.tensor(probabilities))
        )  # 0=contradiction, 1=entailment, 2=neutral
        return prediction

    @classmethod
    def from_pretrained(cls, model_name):
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(model, tokenizer)


class SequentialConsistencyBasedUQ(UncertaintyMethod):
    """
    Computes pairwise entailment consistency between reasoning steps.
    Higher score = more consistent reasoning chain.
    """

    def __init__(self, entailment_checker):
        super().__init__(name="consistency")
        self.entailment_checker = entailment_checker

    def _compute_consistency_score(self, reasoning_steps: List[str]) -> float:
        if len(reasoning_steps) < 2:
            return 1.0
        total_score = 0.0
        count = 0
        for i in range(len(reasoning_steps) - 1):
            premise = reasoning_steps[i]
            hypothesis = reasoning_steps[i + 1]
            prediction = self.entailment_checker.check_entailment(premise, hypothesis)
            total_score += 1.0 if prediction == 2 else 0.0  # only count "entailment"
            count += 1
        return total_score / count if count > 0 else 0.0

    def evaluate_step(
        self,
        previous_steps: List[ReasoningStep],
        candidate_step: ReasoningStep,
    ) -> float:
        reasoning_texts = [s.content for s in previous_steps] + [candidate_step.content]
        score = self._compute_consistency_score(reasoning_texts)
        candidate_step.add_uncertainty_score(self.name, score)
        return score

    def evaluate_path(self, question: str, reasoning_path: ReasoningPath) -> float:
        texts = [s.content for s in reasoning_path.steps]
        path_score = self._compute_consistency_score(texts)
        reasoning_path.total_confidence = path_score
        return path_score


class SemanticEntropy:
    def __init__(self, entailment_checker, strict_entailment=False, verbose=False):
        """
        Args:
            entailment_checker: an EntailmentChecker instance
            strict_entailment: if True, only pairs mutually entailing are considered equivalent
            verbose: if True, prints debug information about clustering
        """
        self.entailment_checker = entailment_checker
        self.strict_entailment = strict_entailment
        self.verbose = verbose

    def get_semantic_ids(self, strings_list):
        def are_equivalent(text1, text2):
            implication_1 = self.entailment_checker.check_entailment(text1, text2)
            implication_2 = self.entailment_checker.check_entailment(text2, text1)

            if self.verbose:
                print(f"  Pair: [{text1}] ↔ [{text2}]")
                print(
                    f"    -> entailment1={implication_1}, entailment2={implication_2}"
                )

            if self.strict_entailment:
                return (implication_1 == 2) and (implication_2 == 2)
            else:
                implications = [implication_1, implication_2]
                # Equivalent if no contradictions and not both neutral
                return (0 not in implications) and (implications != [1, 1])

        semantic_set_ids = [-1] * len(strings_list)
        next_id = 0
        for i, string1 in enumerate(strings_list):
            if semantic_set_ids[i] == -1:
                semantic_set_ids[i] = next_id
                if self.verbose:
                    print(f"Assigning cluster {next_id} to: {string1}")
                for j in range(i + 1, len(strings_list)):
                    if semantic_set_ids[j] == -1:
                        if are_equivalent(string1, strings_list[j]):
                            semantic_set_ids[j] = next_id
                            if self.verbose:
                                print(
                                    f"  -> {strings_list[j]} added to cluster {next_id}"
                                )
                next_id += 1

        if self.verbose:
            print(f"Final cluster assignments: {semantic_set_ids}")

        return semantic_set_ids

    def cluster_assignment_entropy(self, semantic_ids):
        n = len(semantic_ids)
        counts = np.bincount(semantic_ids)
        probabilities = counts / n

        if self.verbose:
            print(f"Cluster counts: {counts}, probabilities: {probabilities}")

        # Avoid log(0) issues
        return max(min(-np.sum(probabilities * np.log(probabilities + 1e-12)), 1.0), 0)

    def compute(self, responses):
        if self.verbose:
            print(
                f"\n=== Computing Semantic Entropy for group of {len(responses)} items ==="
            )
        semantic_ids = self.get_semantic_ids(responses)
        entropy = self.cluster_assignment_entropy(semantic_ids)
        # if self.verbose:
        print(f"Computed entropy: {entropy:.4f}\n")
        return entropy


class SemanticEntropyBasedUQ(UncertaintyMethod):
    """
    Computes semantic entropy over groups of candidate completions.
    If `add_consistency` is True, combines group entropy with individual consistency.
    """

    def __init__(
        self,
        semantic_entropy,
        add_consistency: bool = False,
        consistency_method: Optional[SequentialConsistencyBasedUQ] = None,
    ):
        super().__init__(name="semantic_entropy")
        self.semantic_entropy = semantic_entropy
        self.add_consistency = add_consistency
        self.consistency_method = consistency_method

    def compute_group_entropy(self, group: List[str]) -> float:
        if len(group) <= 1:
            return 0.0
        return self.semantic_entropy.compute(group)

    def evaluate_step(
        self,
        entropy_score: float,
        candidate_step: ReasoningStep,
        previous_steps: List[ReasoningStep],
    ) -> float:
        """
        In this context, `evaluate_step` works on a single candidate, but uses its group info
        stored in metadata (if available) to compute shared entropy.
        """
        # entropy_score = self._compute_group_entropy(group)

        if self.add_consistency and self.consistency_method:
            # Combine with consistency score
            consistency_score = self.consistency_method.evaluate_step(
                previous_steps, candidate_step
            )
            final_score = max(0.0, min(1.0, (1.0 - entropy_score) * consistency_score))
        else:
            final_score = 1.0 - entropy_score  # lower entropy => higher confidence

        # candidate_step.add_uncertainty_score(
        #     self.name, final_score, metadata={"entropy": entropy_score}
        # )

        return UncertaintyScore(
            value=final_score, method=self.name, metadata={"entropy": entropy_score}
        )

    def evaluate_path(self, reasoning_path: ReasoningPath) -> float:
        """
        Compute mean confidence across steps (or a custom aggregation).
        """
        if not reasoning_path.steps:
            reasoning_path.total_confidence = 0.0
            return 0.0

        confidences = [
            step.uncertainty_scores.get(
                self.name, UncertaintyScore(0.5, self.name)
            ).value
            for step in reasoning_path.steps
        ]
        path_score = float(np.mean(confidences))
        reasoning_path.total_confidence = path_score
        return path_score


# ------------------- EXAMPLE USAGE -------------------

if __name__ == "__main__":
    # Load a pre-trained entailment model
    model = AutoModelForSequenceClassification.from_pretrained(
        "cross-encoder/nli-roberta-base"
    )
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-roberta-base")

    entailment_checker = EntailmentChecker(model, tokenizer)
    semantic_entropy = SemanticEntropy(entailment_checker)
    semantic_entropy_uq = SemanticEntropyBasedUQ(semantic_entropy)

    groups = [
        [
            "The man is eating pizza.",
            "A man eats a pizza slice.",
            "Someone is eating food.",
        ],
        ["A man drives a car.", "A car moves on a road."],
        ["The sky is blue."],  # single element → entropy 0
    ]

    scores = semantic_entropy_uq.compute_groupwise_entropy(groups)
    for i, s in enumerate(scores):
        print(f"Group {i} Semantic Entropy: {s:.4f}")
