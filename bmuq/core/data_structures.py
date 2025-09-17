"""
Core data structures for reasoning chains and uncertainty quantification.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum


class ComparisonResult(Enum):
    """Results from comparing reasoning steps."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NOT_DIRECTLY_RELATED = "not_directly_related"


@dataclass
class UncertaintyScore:
    """Container for uncertainty quantification scores and metadata."""
    value: float
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_interval: Optional[tuple] = None
    
    def __post_init__(self):
        """Validate uncertainty score."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Uncertainty score must be between 0 and 1, got {self.value}")


@dataclass 
class ReasoningStep:
    """Represents a single step in the reasoning chain."""
    step_id: int
    content: str
    dependencies: List[int] = field(default_factory=list)
    uncertainty_scores: Dict[str, UncertaintyScore] = field(default_factory=dict)
    target: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def confidence(self) -> float:
        """Get primary confidence score (SelfCheck if available, otherwise average)."""
        if "selfcheck" in self.uncertainty_scores:
            return self.uncertainty_scores["selfcheck"].value
        elif self.uncertainty_scores:
            return sum(score.value for score in self.uncertainty_scores.values()) / len(self.uncertainty_scores)
        else:
            return 0.5  # Default neutral confidence

    def add_uncertainty_score(self, method: str, score: float, metadata: Optional[Dict[str, Any]] = None):
        """Add uncertainty score from a specific method."""
        self.uncertainty_scores[method] = UncertaintyScore(
            value=score,
            method=method,
            metadata=metadata or {}
        )

    def get_uncertainty_score(self, method: str) -> Optional[UncertaintyScore]:
        """Get uncertainty score for a specific method."""
        return self.uncertainty_scores.get(method)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "content": self.content,
            "dependencies": self.dependencies,
            "uncertainty_scores": {
                method: {
                    "value": score.value,
                    "method": score.method,
                    "metadata": score.metadata,
                    "confidence_interval": score.confidence_interval
                }
                for method, score in self.uncertainty_scores.items()
            },
            "target": self.target,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningStep':
        """Create ReasoningStep from dictionary."""
        step = cls(
            step_id=data["step_id"],
            content=data["content"],
            dependencies=data.get("dependencies", []),
            target=data.get("target", ""),
            metadata=data.get("metadata", {})
        )
        
        # Reconstruct uncertainty scores
        for method, score_data in data.get("uncertainty_scores", {}).items():
            step.uncertainty_scores[method] = UncertaintyScore(
                value=score_data["value"],
                method=score_data["method"],
                metadata=score_data.get("metadata", {}),
                confidence_interval=score_data.get("confidence_interval")
            )
        
        return step


@dataclass
class ReasoningPath:
    """Represents a complete reasoning path (sequence of steps)."""
    steps: List[ReasoningStep] = field(default_factory=list)
    total_confidence: float = 0.0
    is_complete: bool = False
    path_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize path confidence."""
        self.update_total_confidence()

    def add_step(self, step: ReasoningStep):
        """Add a step to the reasoning path."""
        self.steps.append(step)
        self.update_total_confidence()

    def update_total_confidence(self, method: str = "selfcheck", 
                              integration_func: Optional[callable] = None):
        """
        Update total confidence using specified integration function.
        
        Args:
            method: Uncertainty quantification method to use
            integration_func: Custom function to integrate step confidences
        """
        if not self.steps:
            self.total_confidence = 0.0
            return

        if integration_func:
            # Use custom integration function
            confidences = [step.uncertainty_scores.get(method, UncertaintyScore(0.5, method)).value 
                          for step in self.steps]
            self.total_confidence = integration_func(confidences)
        else:
            # Use default SelfCheck integration (from paper)
            self._update_selfcheck_confidence(method)

    def _update_selfcheck_confidence(self, method: str = "selfcheck"):
        """Update confidence using SelfCheck integration function from paper."""
        # Convert step confidences to check results (-1, 0, 1)
        check_results = []
        for step in self.steps:
            score = step.uncertainty_scores.get(method, UncertaintyScore(0.5, method)).value
            if score >= 0.8:
                check_results.append(1)  # support
            elif score <= 0.3:
                check_results.append(-1)  # contradict
            else:
                check_results.append(0)  # not directly related

        # Apply SelfCheck integration function (Equation 1 from paper)
        lambda_neg1 = 1.0  # Weight for contradictory evidence
        lambda_0 = 0.3     # Weight for uncertain evidence

        failed_checks = sum(1 for r in check_results if r == -1)
        uncertain_checks = sum(1 for r in check_results if r == 0)

        score = -lambda_neg1 * failed_checks - lambda_0 * uncertain_checks
        self.total_confidence = 2 * (1 / (1 + math.exp(-score)))  # 2 * sigmoid

    def get_step_by_id(self, step_id: int) -> Optional[ReasoningStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_dependency_steps(self, step: ReasoningStep) -> List[ReasoningStep]:
        """Get all steps that this step depends on."""
        return [self.get_step_by_id(dep_id) for dep_id in step.dependencies 
                if self.get_step_by_id(dep_id) is not None]

    def validate_dependencies(self) -> List[str]:
        """Validate that all step dependencies exist and return any issues."""
        issues = []
        for step in self.steps:
            for dep_id in step.dependencies:
                if not any(s.step_id == dep_id for s in self.steps):
                    issues.append(f"Step {step.step_id} depends on non-existent step {dep_id}")
        return issues

    def get_confidence_statistics(self, method: Optional[str] = None) -> Dict[str, float]:
        """Get confidence statistics for the path."""
        if method:
            confidences = [step.uncertainty_scores.get(method, UncertaintyScore(0.5, method)).value 
                          for step in self.steps]
        else:
            confidences = [step.confidence for step in self.steps]
        
        if not confidences:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        import statistics
        return {
            "mean": statistics.mean(confidences),
            "std": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            "min": min(confidences),
            "max": max(confidences)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "total_confidence": self.total_confidence,
            "is_complete": self.is_complete,
            "path_id": self.path_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningPath':
        """Create ReasoningPath from dictionary."""
        path = cls(
            steps=[ReasoningStep.from_dict(step_data) for step_data in data.get("steps", [])],
            total_confidence=data.get("total_confidence", 0.0),
            is_complete=data.get("is_complete", False),
            path_id=data.get("path_id"),
            metadata=data.get("metadata", {})
        )
        return path

    def __len__(self) -> int:
        """Get number of steps in path."""
        return len(self.steps)

    def __iter__(self):
        """Iterate over steps."""
        return iter(self.steps)

    def __getitem__(self, index: Union[int, slice]) -> Union[ReasoningStep, List[ReasoningStep]]:
        """Get step(s) by index."""
        return self.steps[index]