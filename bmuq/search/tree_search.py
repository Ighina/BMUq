"""
Tree search with uncertainty-guided exploration for Chain of Thought reasoning.
"""

import heapq
from typing import List, Dict, Optional, Any, Tuple
from ..core.data_structures import ReasoningStep, ReasoningPath
from ..core.interfaces import UncertaintyMethod
from ..models.base import BaseLLM
from .base_search import BaseSearchAlgorithm


class TreeSearchCoT(BaseSearchAlgorithm):
    """
    Tree search for Chain of Thought reasoning with uncertainty quantification.
    
    Uses uncertainty scores to guide exploration and prune low-confidence paths.
    """

    def __init__(self, llm: BaseLLM, uncertainty_method: UncertaintyMethod,
                 beam_width: int = 3, max_depth: int = 8, 
                 confidence_threshold: float = 0.1,
                 exploration_weight: float = 1.0):
        """
        Initialize tree search algorithm.
        
        Args:
            llm: Language model for step generation
            uncertainty_method: Method for uncertainty quantification
            beam_width: Number of paths to keep at each depth level
            max_depth: Maximum reasoning chain depth
            confidence_threshold: Minimum confidence to keep exploring a path
            exploration_weight: Weight for exploration vs exploitation
        """
        super().__init__("tree_search_cot", llm, uncertainty_method, max_depth)
        self.beam_width = beam_width
        self.confidence_threshold = confidence_threshold
        self.exploration_weight = exploration_weight
        
        # Statistics tracking
        self.search_stats = {
            "paths_explored": 0,
            "steps_generated": 0,
            "paths_pruned": 0,
            "completed_paths": 0
        }

    def search(self, question: str, **kwargs) -> List[ReasoningPath]:
        """
        Perform tree search to find high-confidence reasoning paths.
        
        Args:
            question: Question to solve
            **kwargs: Additional parameters (num_solutions, verbose, etc.)
            
        Returns:
            List of reasoning paths ranked by confidence
        """
        num_solutions = kwargs.get("num_solutions", 5)
        verbose = kwargs.get("verbose", False)
        
        # Reset statistics
        self.search_stats = {
            "paths_explored": 0,
            "steps_generated": 0, 
            "paths_pruned": 0,
            "completed_paths": 0
        }
        
        # Initialize search with empty path
        initial_path = ReasoningPath(steps=[], path_id="root")
        
        # Priority queue: (negative_score, path_counter, path)
        # Using negative score because heapq is a min-heap
        active_paths = [(0.0, 0, initial_path)]
        completed_paths = []
        path_counter = 1
        
        if verbose:
            print(f"Starting tree search for: {question}")
            print(f"Beam width: {self.beam_width}, Max depth: {self.max_depth}")
        
        # Main search loop
        for depth in range(self.max_depth):
            if not active_paths:
                break
                
            next_paths = []
            paths_processed = 0
            
            # Process active paths at current depth
            while active_paths and paths_processed < self.beam_width * 2:
                neg_score, _, current_path = heapq.heappop(active_paths)
                paths_processed += 1
                self.search_stats["paths_explored"] += 1
                
                # Check if path is already complete
                if self.is_solution_complete(question, current_path):
                    current_path.is_complete = True
                    completed_paths.append(current_path)
                    self.search_stats["completed_paths"] += 1
                    if verbose:
                        print(f"  Complete path found with {len(current_path.steps)} steps")
                    continue
                
                # Prune low-confidence paths
                if -neg_score < self.confidence_threshold and len(current_path.steps) > 0:
                    self.search_stats["paths_pruned"] += 1
                    continue
                
                # Generate candidate next steps
                candidates = self.generate_next_steps(question, current_path, num_candidates=3)
                self.search_stats["steps_generated"] += len(candidates)
                
                # Evaluate candidates and create new paths
                for candidate in candidates:
                    # Create new path with candidate step
                    new_steps = current_path.steps + [candidate]
                    new_path = ReasoningPath(
                        steps=new_steps,
                        path_id=f"path_{path_counter}",
                        metadata={"parent_path": current_path.path_id, "depth": depth + 1}
                    )
                    
                    # Evaluate uncertainty for the new step
                    uncertainty_score = self.uncertainty_method.evaluate_step(
                        question, current_path.steps, candidate
                    )
                    candidate.uncertainty_scores[self.uncertainty_method.name] = uncertainty_score
                    
                    # Update path confidence
                    path_confidence = self.uncertainty_method.evaluate_path(question, new_path)
                    new_path.total_confidence = path_confidence
                    
                    # Add exploration bonus for diversity
                    exploration_bonus = self.exploration_weight * (1.0 / (depth + 1))
                    adjusted_score = path_confidence + exploration_bonus
                    
                    next_paths.append((-adjusted_score, path_counter, new_path))
                    path_counter += 1
            
            # Keep only top beam_width paths for next iteration
            next_paths.sort(key=lambda x: x[0])  # Sort by negative score
            active_paths = next_paths[:self.beam_width]
            
            if verbose:
                print(f"Depth {depth + 1}: {len(active_paths)} active paths, "
                      f"{len(completed_paths)} completed")
                for i, (neg_score, _, path) in enumerate(active_paths[:3]):
                    print(f"  Path {i+1}: Confidence {-neg_score:.3f}, "
                          f"Steps: {len(path.steps)}")
        
        # Add remaining active paths to completed paths
        for neg_score, _, path in active_paths:
            completed_paths.append(path)
        
        # Sort by confidence and return top solutions
        completed_paths.sort(key=lambda p: p.total_confidence, reverse=True)
        
        if verbose:
            print(f"Search completed: {len(completed_paths)} total paths found")
            print(f"Statistics: {self.search_stats}")
        
        return completed_paths[:num_solutions]

    def generate_next_steps(self, question: str, current_path: ReasoningPath, 
                           num_candidates: int = 3) -> List[ReasoningStep]:
        """
        Generate candidate next steps with enhanced prompting for tree search.
        
        Args:
            question: Original question
            current_path: Current reasoning path
            num_candidates: Number of candidates to generate
            
        Returns:
            List of candidate reasoning steps
        """
        # Use different generation strategies for diversity
        candidates = []
        
        # Strategy 1: Direct continuation
        direct_candidates = self._generate_direct_continuation(
            question, current_path, num_candidates // 2 + 1
        )
        candidates.extend(direct_candidates)
        
        # Strategy 2: Alternative approaches (if we have some steps)
        if len(current_path.steps) > 0:
            alternative_candidates = self._generate_alternative_approaches(
                question, current_path, num_candidates // 2
            )
            candidates.extend(alternative_candidates)
        
        # Ensure we don't exceed requested number
        return candidates[:num_candidates]

    def _generate_direct_continuation(self, question: str, current_path: ReasoningPath, 
                                    num_candidates: int) -> List[ReasoningStep]:
        """Generate steps that directly continue current reasoning."""
        candidates = []
        steps_text = self._format_steps(current_path.steps)
        
        prompt = f"""Problem: {question}

Current solution progress:
{steps_text}

Continue with the next logical step that builds directly on the work above. 
Be specific and make clear progress toward solving the problem.

Next step:"""

        for i in range(num_candidates):
            try:
                response = self.llm.generate(prompt, max_tokens=150)
                step_content = self._clean_step_response(response)
                
                step = ReasoningStep(
                    step_id=len(current_path.steps),
                    content=step_content,
                    dependencies=self._infer_dependencies(step_content, current_path.steps),
                    metadata={
                        "generation_strategy": "direct_continuation",
                        "attempt": i + 1
                    }
                )
                candidates.append(step)
                
            except Exception as e:
                print(f"Error in direct continuation generation: {e}")
                continue
        
        return candidates

    def _generate_alternative_approaches(self, question: str, current_path: ReasoningPath,
                                       num_candidates: int) -> List[ReasoningStep]:
        """Generate steps that explore alternative solution approaches."""
        candidates = []
        steps_text = self._format_steps(current_path.steps)
        
        prompt = f"""Problem: {question}

Current solution attempt:
{steps_text}

Consider an alternative approach or verification step for this problem. 
You could:
- Try a different solution method
- Verify the current approach
- Consider edge cases
- Use a different mathematical technique

Alternative next step:"""

        for i in range(num_candidates):
            try:
                response = self.llm.generate(prompt, max_tokens=150, temperature=0.8)
                step_content = self._clean_step_response(response)
                
                step = ReasoningStep(
                    step_id=len(current_path.steps),
                    content=step_content,
                    dependencies=self._infer_dependencies(step_content, current_path.steps),
                    metadata={
                        "generation_strategy": "alternative_approach",
                        "attempt": i + 1
                    }
                )
                candidates.append(step)
                
            except Exception as e:
                print(f"Error in alternative approach generation: {e}")
                continue
        
        return candidates

    def _infer_dependencies(self, step_content: str, previous_steps: List[ReasoningStep]) -> List[int]:
        """
        Infer which previous steps this step depends on based on content.
        
        This is a simple heuristic - could be improved with more sophisticated NLP.
        """
        dependencies = []
        step_content_lower = step_content.lower()
        
        # Look for explicit references to previous steps
        for prev_step in previous_steps:
            prev_content = prev_step.content.lower()
            
            # Check for shared mathematical terms, variables, etc.
            if any(word in step_content_lower and word in prev_content 
                   for word in ['equation', 'substitute', 'value', 'result', 'answer']):
                dependencies.append(prev_step.step_id)
        
        # If no explicit dependencies found, assume dependence on recent steps
        if not dependencies and previous_steps:
            # Depend on the most recent step by default
            dependencies.append(previous_steps[-1].step_id)
        
        return dependencies

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get detailed search statistics."""
        return {
            **self.search_stats,
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "confidence_threshold": self.confidence_threshold,
            "exploration_weight": self.exploration_weight
        }

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get comprehensive algorithm information."""
        info = super().get_algorithm_info()
        info.update({
            "beam_width": self.beam_width,
            "confidence_threshold": self.confidence_threshold,
            "exploration_weight": self.exploration_weight,
            "search_stats": self.search_stats
        })
        return info