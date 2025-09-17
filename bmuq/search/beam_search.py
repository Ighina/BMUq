"""
Beam search implementation for Chain of Thought reasoning.
"""

import copy
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from ..core.data_structures import ReasoningStep, ReasoningPath
from ..core.interfaces import UncertaintyMethod
from ..models.base import BaseLLM
from .base_search import BaseSearchAlgorithm


class BeamSearchCoT(BaseSearchAlgorithm):
    """
    Beam search implementation for Chain of Thought with uncertainty quantification.
    """

    def __init__(
        self,
        llm: BaseLLM,
        uncertainty_method: UncertaintyMethod,
        beam_width: int = 3,
        max_depth: int = 8,
        diversity_penalty: float = 0.1,
        structured_format: bool = True,
    ):
        super().__init__(
            "beam_search_cot",
            llm,
            uncertainty_method,
            max_depth,
            structured_format=structured_format,
        )
        self.beam_width = beam_width
        self.diversity_penalty = diversity_penalty
        self.search_stats = {
            "paths_explored": 0,
            "steps_generated": 0,
            "diversity_penalties_applied": 0,
        }

    def search(self, question: str, **kwargs) -> List[ReasoningPath]:
        verbose = kwargs.get("verbose", False)

        self.search_stats = {
            "paths_explored": 0,
            "steps_generated": 0,
            "diversity_penalties_applied": 0,
        }

        current_beams = [ReasoningPath(steps=[], path_id=f"beam_0")]
        completed_paths = []

        if verbose:
            print(f"Starting beam search for: {question}")
            print(f"Beam width: {self.beam_width}, Max depth: {self.max_depth}")

        for depth in range(self.max_depth):
            if not current_beams:
                break

            expanded_groups = []
            group_to_beam = []
            all_candidates = []

            # Step 1: expand each beam → groups of candidates
            for beam_idx, current_beam in enumerate(current_beams):
                self.search_stats["paths_explored"] += 1

                if self.is_solution_complete(question, current_beam):
                    current_beam.is_complete = True
                    completed_paths.append(current_beam)
                    if verbose:
                        print(
                            f"  Depth {depth}: Completed path with {len(current_beam.steps)} steps"
                        )
                    continue

                candidates = self.generate_next_steps(
                    question, current_beam, num_candidates=self.beam_width
                )
                self.search_stats["steps_generated"] += len(candidates)

                if candidates:
                    expanded_groups.append([c.content for c in candidates])
                    group_to_beam.append((current_beam, candidates))
                    all_candidates.extend((current_beam, c) for c in candidates)

            if not expanded_groups:
                break  # no expansions possible

            # Step 2: if SemanticEntropyBasedUQ is being used → special handling
            if hasattr(self.uncertainty_method, "semantic_entropy"):

                group_entropies = []
                for group in expanded_groups:
                    group_entropies.append(
                        self.uncertainty_method.compute_group_entropy(group)
                    )

                # if not getattr(self.uncertainty_method, "add_consistency", False):
                #     # --- MODE 2a: Select entire group with lowest entropy ---
                #     best_group_idx = int(np.argmin(group_entropies))
                #     current_beams = [
                #         ReasoningPath(
                #             steps=group_to_beam[best_group_idx][0].steps + [cand],
                #             path_id=f"beam_selected_{depth}_{i}",
                #             total_confidence=1.0 - group_entropies[best_group_idx],
                #         )
                #         for i, cand in enumerate(group_to_beam[best_group_idx][1])
                #     ]
                #     continue  # go to next depth

                # else:
                # --- MODE 2b: Per-candidate scoring (entropy + consistency) ---
                next_beams = []
                for group_idx, (parent_beam, candidates) in enumerate(group_to_beam):
                    group_entropy = group_entropies[group_idx]
                    for cand in candidates:

                        # # consistency contribution
                        # consistency_score = 0.0
                        # if self.uncertainty_method.consistency_uq:
                        #     consistency_score = self.uncertainty_method.consistency_uq.compute_consistency_score(
                        #         [step.content for step in new_path.steps]
                        #     )
                        # total_score = group_entropy + consistency_score

                        # TODO: in theory assigning is redundant since it already uses add_uncertainty_score in the evaluate_step

                        cand.uncertainty_scores[self.uncertainty_method.name] = (
                            self.uncertainty_method.evaluate_step(
                                group_entropy, cand, parent_beam.steps
                            )
                        )

                        new_path = ReasoningPath(
                            steps=parent_beam.steps + [cand],
                            path_id=f"beam_{group_idx}_step_{len(parent_beam.steps)}",
                        )

                        new_path.total_confidence = (
                            self.uncertainty_method.evaluate_path(new_path)
                        )  # lower = better
                        next_beams.append(new_path)

                # sort and keep top beams
                next_beams.sort(key=lambda p: p.total_confidence, reverse=True)
                current_beams = next_beams[: self.beam_width]

            else:
                # fallback to standard per-candidate scoring
                next_beams = []
                for parent_beam, candidate in all_candidates:
                    new_path = ReasoningPath(
                        steps=parent_beam.steps + [candidate],
                        path_id=f"beam_step_{len(parent_beam.steps)}",
                    )
                    uncertainty_score = self.uncertainty_method.evaluate_step(
                        question, parent_beam.steps, candidate
                    )
                    candidate.uncertainty_scores[self.uncertainty_method.name] = (
                        uncertainty_score
                    )
                    path_confidence = self.uncertainty_method.evaluate_path(
                        question, new_path
                    )
                    new_path.total_confidence = path_confidence
                    next_beams.append(new_path)

                scored_beams = self._score_beams_with_diversity(next_beams)
                current_beams = [beam for _, beam in scored_beams[: self.beam_width]]

            if verbose:
                print(f"Depth {depth + 1}: {len(current_beams)} active beams")
                for i, beam in enumerate(current_beams[:3]):
                    print(f"  Beam {i+1}: Confidence {beam.total_confidence:.3f}")

        completed_paths.extend(current_beams)
        completed_paths.sort(key=lambda p: p.total_confidence, reverse=True)

        if verbose:
            print(f"Beam search completed: {len(completed_paths)} paths found")

        return completed_paths

    # class BeamSearchCoT(BaseSearchAlgorithm):
    #     """
    #     Beam search implementation for Chain of Thought with uncertainty quantification.

    #     Maintains fixed number of best paths at each step, with support for
    #     both per-step and group-based uncertainty methods.
    #     """

    #     def __init__(
    #         self,
    #         llm: BaseLLM,
    #         uncertainty_method: UncertaintyMethod,
    #         beam_width: int = 3,
    #         max_depth: int = 8,
    #         diversity_penalty: float = 0.1,
    #         structured_format: bool = True,
    #     ):
    #         super().__init__(
    #             "beam_search_cot",
    #             llm,
    #             uncertainty_method,
    #             max_depth,
    #             structured_format=structured_format,
    #         )
    #         self.beam_width = beam_width
    #         self.diversity_penalty = diversity_penalty
    #         self.search_stats = {
    #             "paths_explored": 0,
    #             "steps_generated": 0,
    #             "diversity_penalties_applied": 0,
    #         }

    #     def search(self, question: str, **kwargs) -> List[ReasoningPath]:
    #         verbose = kwargs.get("verbose", False)
    #         self.search_stats = {
    #             "paths_explored": 0,
    #             "steps_generated": 0,
    #             "diversity_penalties_applied": 0,
    #         }

    #         current_beams = [ReasoningPath(steps=[], path_id=f"beam_0")]
    #         completed_paths = []

    #         if verbose:
    #             print(f"Starting beam search for: {question}")
    #             print(f"Beam width: {self.beam_width}, Max depth: {self.max_depth}")

    #         for depth in range(self.max_depth):
    #             if not current_beams:
    #                 break

    #             next_beams = []

    #             for beam_idx, current_beam in enumerate(current_beams):
    #                 self.search_stats["paths_explored"] += 1

    #                 # Check if solution is already complete
    #                 if self.is_solution_complete(question, current_beam):
    #                     current_beam.is_complete = True
    #                     completed_paths.append(current_beam)
    #                     if verbose:
    #                         print(
    #                             f"  Depth {depth}: Completed path with {len(current_beam.steps)} steps"
    #                         )
    #                     continue

    #                 # Generate candidate steps
    #                 candidates = self.generate_next_steps(
    #                     question, current_beam, num_candidates=self.beam_width
    #                 )
    #                 self.search_stats["steps_generated"] += len(candidates)

    #                 if hasattr(self.uncertainty_method, "semantic_entropy"):
    #                     # Group-level uncertainty: compute once for candidates
    #                     group_entropy_scores = (
    #                         self.uncertainty_method.compute_groupwise_entropy(
    #                             [c.content for c in candidates]
    #                         )
    #                     )
    #                 else:
    #                     group_entropy_scores = [None] * len(candidates)

    #                 for i, candidate in enumerate(candidates):
    #                     new_path = copy.deepcopy(current_beam)
    #                     candidate_step = candidate

    #                     # --- STEP-LEVEL UNCERTAINTY ---
    #                     step_score = self.uncertainty_method.evaluate_step(
    #                         question, new_path.steps, candidate_step
    #                     )
    #                     candidate_step.add_uncertainty_score(
    #                         self.uncertainty_method.name,
    #                         step_score,
    #                         metadata={
    #                             "depth": depth,
    #                             "group_entropy": group_entropy_scores[i],
    #                         },
    #                     )

    #                     new_path.steps.append(candidate_step)

    #                     # --- PATH-LEVEL CONFIDENCE UPDATE ---
    #                     path_confidence = self.uncertainty_method.evaluate_path(
    #                         question, new_path
    #                     )
    #                     new_path.total_confidence = path_confidence

    #                     next_beams.append(new_path)

    #             # Diversity scoring + beam pruning
    #             if next_beams:
    #                 scored_beams = self._score_beams_with_diversity(next_beams)
    #                 current_beams = [beam for _, beam in scored_beams[: self.beam_width]]
    #             else:
    #                 current_beams = []

    #             if verbose:
    #                 print(f"Depth {depth + 1}: {len(current_beams)} active beams")
    #                 for i, beam in enumerate(current_beams[:3]):
    #                     print(f"  Beam {i+1}: Confidence {beam.total_confidence:.3f}")

    #         completed_paths.extend(current_beams)
    #         completed_paths.sort(key=lambda p: p.total_confidence, reverse=True)

    #         if verbose:
    #             print(f"Beam search completed: {len(completed_paths)} paths found")

    #         return completed_paths

    def _score_beams_with_diversity(
        self, beams: List[ReasoningPath]
    ) -> List[Tuple[float, ReasoningPath]]:
        """
        Score beams with diversity penalty to encourage exploration.

        Args:
            beams: List of reasoning paths to score

        Returns:
            List of (score, beam) tuples sorted by score descending
        """
        scored_beams = []

        for i, beam in enumerate(beams):
            base_score = beam.total_confidence
            diversity_penalty = 0.0

            # Calculate similarity with other beams
            for j, other_beam in enumerate(beams):
                if i != j:
                    similarity = self._calculate_path_similarity(beam, other_beam)
                    diversity_penalty += similarity * self.diversity_penalty

            if diversity_penalty > 0:
                self.search_stats["diversity_penalties_applied"] += 1

            final_score = base_score - diversity_penalty
            scored_beams.append((final_score, beam))

        # Sort by score descending
        scored_beams.sort(key=lambda x: x[0], reverse=True)
        return scored_beams

    def _calculate_path_similarity(
        self, path1: ReasoningPath, path2: ReasoningPath
    ) -> float:
        """
        Calculate similarity between two reasoning paths.

        Simple heuristic based on overlapping words in step content.
        """
        if not path1.steps or not path2.steps:
            return 0.0

        # Get words from all steps in each path
        words1 = set()
        words2 = set()

        for step in path1.steps:
            words1.update(step.content.lower().split())

        for step in path2.steps:
            words2.update(step.content.lower().split())

        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get beam search statistics."""
        return {
            **self.search_stats,
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "diversity_penalty": self.diversity_penalty,
        }

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get comprehensive algorithm information."""
        info = super().get_algorithm_info()
        info.update(
            {
                "beam_width": self.beam_width,
                "diversity_penalty": self.diversity_penalty,
                "search_stats": self.search_stats,
            }
        )
        return info
