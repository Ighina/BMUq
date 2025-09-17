"""
Abstract interfaces for uncertainty quantification methods and search algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .data_structures import ReasoningStep, ReasoningPath, UncertaintyScore


class UncertaintyMethod(ABC):
    """Abstract base class for uncertainty quantification methods."""

    def __init__(self, name: str):
        """
        Initialize uncertainty method.
        
        Args:
            name: Name identifier for this method
        """
        self.name = name

    @abstractmethod
    def evaluate_step(self, question: str, reasoning_path: List[ReasoningStep], 
                     step_to_evaluate: ReasoningStep) -> UncertaintyScore:
        """
        Evaluate uncertainty for a single reasoning step.
        
        Args:
            question: The original question being solved
            reasoning_path: The reasoning path so far (context)
            step_to_evaluate: The step to evaluate uncertainty for
            
        Returns:
            UncertaintyScore containing the uncertainty quantification result
        """
        pass

    @abstractmethod
    def evaluate_path(self, question: str, reasoning_path: ReasoningPath) -> float:
        """
        Evaluate uncertainty for an entire reasoning path.
        
        Args:
            question: The original question being solved
            reasoning_path: Complete reasoning path to evaluate
            
        Returns:
            Overall confidence score for the path
        """
        pass

    def batch_evaluate_steps(self, question: str, reasoning_path: List[ReasoningStep],
                            steps_to_evaluate: List[ReasoningStep]) -> List[UncertaintyScore]:
        """
        Evaluate uncertainty for multiple steps (default implementation).
        
        Override this method to implement efficient batch evaluation.
        """
        return [self.evaluate_step(question, reasoning_path, step) for step in steps_to_evaluate]

    def get_method_info(self) -> Dict[str, Any]:
        """Get information about this uncertainty method."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "description": self.__doc__ or "No description available"
        }


class SearchAlgorithm(ABC):
    """Abstract base class for reasoning search algorithms."""

    def __init__(self, name: str, uncertainty_method: UncertaintyMethod):
        """
        Initialize search algorithm.
        
        Args:
            name: Name identifier for this search algorithm
            uncertainty_method: Uncertainty quantification method to use
        """
        self.name = name
        self.uncertainty_method = uncertainty_method

    @abstractmethod
    def search(self, question: str, **kwargs) -> List[ReasoningPath]:
        """
        Perform search to find reasoning paths.
        
        Args:
            question: The question to solve
            **kwargs: Additional search parameters
            
        Returns:
            List of reasoning paths ranked by confidence/quality
        """
        pass

    @abstractmethod
    def generate_next_steps(self, question: str, current_path: ReasoningPath, 
                           num_candidates: int = 3) -> List[ReasoningStep]:
        """
        Generate candidate next steps for the current reasoning path.
        
        Args:
            question: The original question
            current_path: Current reasoning path
            num_candidates: Number of candidate steps to generate
            
        Returns:
            List of candidate reasoning steps
        """
        pass

    @abstractmethod
    def is_solution_complete(self, question: str, path: ReasoningPath) -> bool:
        """
        Check if a reasoning path provides a complete solution.
        
        Args:
            question: The original question
            path: Reasoning path to check
            
        Returns:
            True if the solution is complete
        """
        pass

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about this search algorithm."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "uncertainty_method": self.uncertainty_method.get_method_info(),
            "description": self.__doc__ or "No description available"
        }


class BatchProcessor(ABC):
    """Abstract interface for batch processing multiple questions/problems."""

    @abstractmethod
    def process_batch(self, questions: List[str], **kwargs) -> List[List[ReasoningPath]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions to process
            **kwargs: Additional processing parameters
            
        Returns:
            List of reasoning path lists, one per question
        """
        pass