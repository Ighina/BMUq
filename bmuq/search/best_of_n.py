"""
Best of N implementation for Chain of Thought reasoning.
"""

import copy
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from ..core.data_structures import ReasoningStep, ReasoningPath, UncertaintyScore
from ..core.interfaces import UncertaintyMethod
from ..models.base import BaseLLM
from .base_search import BaseSearchAlgorithm, MathReasoning


class BestNSearchCoT(BaseSearchAlgorithm):
    """
    Beam search implementation for Chain of Thought with uncertainty quantification.
    """

    def __init__(
        self,
        llm: BaseLLM,
        uncertainty_method: UncertaintyMethod,
        beam_width: int = 3,
        structured_format: bool = True,
        structured_output: bool = True,
    ):
        super().__init__(
            "best_of_n_cot",
            llm,
            uncertainty_method,
            max_depth=None,
            structured_format=structured_format,
            structured_output=structured_output,
        )
        self.beam_width = beam_width
        self.structured_output = structured_output
        self.diversity_penalty = None  # just a placeholder for back compatibility

    def _find_steps(self, response: str) -> List[str]:
        if self.structured_output:
            steps = [res.output for res in response.steps]
        elif response.lower().startswith("step"):
            steps = response.lower().split("step")
        elif response.lower().startswith("**step"):
            steps = response.lower().split("**step")
        else:
            steps = self.llm.generate(f"Parse this reasoning chain in a series of steps of the form **Step 1**: <step 1>\n**Step 2**: <step 2>, etc. {response}. JUST OUTPUT THE SERIES OF STEPS AS DESCRIBED.")
            steps = steps.lower().split("**step")
        return steps

    def _find_answer(self, response: str) -> str:
        if self.structured_output:
            answer = response.final_answer
        elif len(response.lower().split("final_answer")) > 1:
            answer = response[response.lower().rfind("final_answer") :]
        else:
            answer = self.llm.generate(
                f"Find the answer to the problem contained in this reasoning chain {response}. JUST OUTPUT THE GIVEN ANSWER AND NOTHING MORE."
            )
        return answer

    def generate_next_steps(
        self, question, paths, num_candidates=3, max_tokens=512, return_text=False
    ) -> Tuple[List[ReasoningPath], List[str]]:
        """
        Generate a set of length n of candidate solutions for the given problem.
        Args:
            question: The question to solve.
            paths: Current reasoning paths (not used in Best of N).
            num_candidates: Number of candidate solutions to generate.
            max_tokens: Maximum tokens for LLM response.
            return_text: Whether to return raw text responses.

        Returns:
            A tuple of (List of ReasoningPath, List of answers).

        """
        prompt = f"Problem: {question}. Think at the solution step-by-step."

        if return_text:
            paths = [""] * num_candidates

        answers = []
        for i in range(num_candidates):
            try:
                candidates = []
                structured_output = MathReasoning if self.structured_output else None
                response = self.llm.generate(
                    prompt, structured_output=structured_output, max_tokens=max_tokens
                )
                step_content = self._find_steps(response)
                answer = self._find_answer(response)

                answers.append(answer)

                for idx, step in enumerate(step_content):
                    if return_text:
                        step = step_content[idx]
                    else:
                        step = ReasoningStep(
                            step_id=idx,
                            content=step_content[idx],
                            dependencies=[],
                            metadata={"generation_attempt": i + 1},
                        )
                    candidates.append(step)

                if return_text:
                    paths[i] = candidates
                else:
                    paths[i].steps = candidates

            except Exception as e:
                print(f"Error generating step candidate {i+1}: {e}")
                continue

        return paths, answers

    def search(
        self,
        question: str,
        generated_outputs: Optional[Dict[str, List[List[str]]]] = None,
        **kwargs,
    ) -> List[ReasoningPath]:
        verbose = kwargs.get("verbose", False)

        self.search_stats = {
            "paths_explored": 0,
            "steps_generated": 0,
            "diversity_penalties_applied": 0,
        }

        current_beams = [
            ReasoningPath(steps=[], path_id=f"beam_{idx}")
            for idx in range(self.beam_width)
        ]

        if generated_outputs is not None:
            # precomputed outputs, save computation and ensure fair comparison (it works by calling the generate_next_steps outside the logic)
            completed_paths = self._convert_text_to_paths(
                generated_outputs["reasonings"]
            )
            answers = generated_outputs["answers"]
        else:
            completed_paths, answers = self.generate_next_steps(
                question, current_beams, self.beam_width
            )

        if verbose:
            print(f"Starting Best of N search for: {question}")
            print(f"Number of Candidates: {self.beam_width}")

        # # TODO: implement the various uncertainty methods in this different context in which every chain of thought is already genereated as a standalone chain!

        if hasattr(self.uncertainty_method, "semantic_entropy"):
            raise NotImplementedError()

            # TODO: not clear how to implement semantic entropy in this case... Should I compute the in-chain entropy or doing across different chains such as in CoTa method?

            # group_entropies = []

            # for group in expanded_groups:
            #     group_entropies.append(
            #         self.uncertainty_method.compute_group_entropy(group)
            #     )

            # # if not getattr(self.uncertainty_method, "add_consistency", False):
            # #     # --- MODE 2a: Select entire group with lowest entropy ---
            # #     best_group_idx = int(np.argmin(group_entropies))
            # #     current_beams = [
            # #         ReasoningPath(
            # #             steps=group_to_beam[best_group_idx][0].steps + [cand],
            # #             path_id=f"beam_selected_{depth}_{i}",
            # #             total_confidence=1.0 - group_entropies[best_group_idx],
            # #         )
            # #         for i, cand in enumerate(group_to_beam[best_group_idx][1])
            # #     ]
            # #     continue  # go to next depth

            # # else:
            # # --- MODE 2b: Per-candidate scoring (entropy + consistency) ---
            # next_beams = []
            # for group_idx, (parent_beam, candidates) in enumerate(group_to_beam):
            #     group_entropy = group_entropies[group_idx]
            #     for cand in candidates:

            #         # # consistency contribution
            #         # consistency_score = 0.0
            #         # if self.uncertainty_method.consistency_uq:
            #         #     consistency_score = self.uncertainty_method.consistency_uq.compute_consistency_score(
            #         #         [step.content for step in new_path.steps]
            #         #     )
            #         # total_score = group_entropy + consistency_score

            #         cand.uncertainty_scores[self.uncertainty_method.name] = (
            #             self.uncertainty_method.evaluate_step(
            #                 group_entropy, cand, parent_beam.steps
            #             )
            #         )

            #         new_path = ReasoningPath(
            #             steps=parent_beam.steps + [cand],
            #             path_id=f"beam_{group_idx}_step_{len(parent_beam.steps)}",
            #         )

            #         new_path.total_confidence = (
            #             self.uncertainty_method.evaluate_path(new_path)
            #         )  # lower = better
            #         next_beams.append(new_path)

            # # sort and keep top beams
            # next_beams.sort(key=lambda p: p.total_confidence, reverse=True)
            # current_beams = next_beams[: self.beam_width]

        elif hasattr(self.uncertainty_method, "coherence_args"):
            # This is the relative_coherence method, which compute
            if self.uncertainty_method.add_topic_score:
                q = question
            else:
                q = None
            for path in completed_paths:
                path.total_confidence = self.uncertainty_method.evaluate_path(path, q)

        elif hasattr(self.uncertainty_method, "total_coherence"):
            raise NotImplementedError()
            # TODO: create a method that takes into account the whole reasoning chain at once instead of doing it step by step (like relative coherence above)

        else:
            for path in completed_paths:
                for idx, candidate in enumerate(path.steps):
                    if not idx:
                        uncertainty_score = UncertaintyScore(
                            value=0, method=self.name, metadata=None
                        )
                    else:
                        uncertainty_score = self.uncertainty_method.evaluate_step(
                            question, path.steps[:idx], candidate
                        )
                    candidate.uncertainty_scores[self.uncertainty_method.name] = (
                        uncertainty_score
                    )
                path_confidence = self.uncertainty_method.evaluate_path(question, path)
                path.total_confidence = path_confidence

        answers = [
            x
            for _, x in sorted(
                zip(completed_paths, answers),
                key=lambda pair: pair[0].total_confidence,
                reverse=True,
            )
        ]
        completed_paths.sort(key=lambda p: p.total_confidence, reverse=True)

        if verbose:
            print(f"Best of N search completed: {len(completed_paths)} paths found")

        return completed_paths, answers

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

    def _convert_text_to_paths(self, texts: List[str]) -> List[ReasoningPath]:
        """
        Convert raw text responses into ReasoningPath objects.

        Args:
            texts: List of raw text responses from LLM

        Returns:
            List of ReasoningPath objects
        """
        paths = []
        
        for idx, text in enumerate(texts):
            if len(text)==1:
                text = self._find_steps(text[0])
            steps = [
                ReasoningStep(
                    step_id=step_idx,
                    content=step,
                    dependencies=[],
                    metadata={"generation_attempt": idx + 1},
                )
                for step_idx, step in enumerate(text)
            ]
            path = ReasoningPath(steps=steps, path_id=f"beam_{idx}")
            paths.append(path)
        return paths

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
