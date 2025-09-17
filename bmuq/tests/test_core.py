"""
Tests for core data structures and interfaces.
"""

import pytest
from bmuq.core.data_structures import ReasoningStep, ReasoningPath, UncertaintyScore


class TestUncertaintyScore:
    """Test UncertaintyScore data structure."""
    
    def test_valid_score(self):
        """Test creation of valid uncertainty score."""
        score = UncertaintyScore(value=0.8, method="selfcheck")
        assert score.value == 0.8
        assert score.method == "selfcheck"
        assert score.metadata == {}
    
    def test_invalid_score_range(self):
        """Test that invalid score ranges raise ValueError."""
        with pytest.raises(ValueError):
            UncertaintyScore(value=1.5, method="test")
        
        with pytest.raises(ValueError):
            UncertaintyScore(value=-0.1, method="test")
    
    def test_metadata(self):
        """Test uncertainty score with metadata."""
        metadata = {"stage_1": "extract_target", "comparison": "supports"}
        score = UncertaintyScore(
            value=0.9,
            method="selfcheck",
            metadata=metadata
        )
        assert score.metadata == metadata


class TestReasoningStep:
    """Test ReasoningStep data structure."""
    
    def test_step_creation(self):
        """Test basic step creation."""
        step = ReasoningStep(
            step_id=1,
            content="Subtract 3 from both sides: 2x = 4",
            dependencies=[0]
        )
        assert step.step_id == 1
        assert step.content == "Subtract 3 from both sides: 2x = 4"
        assert step.dependencies == [0]
        assert step.confidence == 0.5  # Default when no uncertainty scores
    
    def test_add_uncertainty_score(self):
        """Test adding uncertainty scores to step."""
        step = ReasoningStep(step_id=1, content="Test step")
        
        step.add_uncertainty_score("selfcheck", 0.8, {"stage": "comparison"})
        
        assert "selfcheck" in step.uncertainty_scores
        assert step.uncertainty_scores["selfcheck"].value == 0.8
        assert step.confidence == 0.8  # Should use selfcheck as primary
    
    def test_step_serialization(self):
        """Test step to/from dictionary conversion."""
        step = ReasoningStep(
            step_id=2,
            content="Divide by 2: x = 2",
            dependencies=[1],
            target="Find the value of x"
        )
        step.add_uncertainty_score("selfcheck", 0.9)
        
        # Convert to dict
        step_dict = step.to_dict()
        assert step_dict["step_id"] == 2
        assert step_dict["content"] == "Divide by 2: x = 2"
        assert step_dict["dependencies"] == [1]
        assert step_dict["target"] == "Find the value of x"
        assert "selfcheck" in step_dict["uncertainty_scores"]
        
        # Convert back from dict
        restored_step = ReasoningStep.from_dict(step_dict)
        assert restored_step.step_id == step.step_id
        assert restored_step.content == step.content
        assert restored_step.dependencies == step.dependencies
        assert restored_step.target == step.target
        assert "selfcheck" in restored_step.uncertainty_scores


class TestReasoningPath:
    """Test ReasoningPath data structure."""
    
    def test_empty_path(self):
        """Test empty reasoning path."""
        path = ReasoningPath()
        assert len(path.steps) == 0
        assert path.total_confidence == 0.0
        assert not path.is_complete
    
    def test_add_steps(self):
        """Test adding steps to path."""
        path = ReasoningPath()
        
        step1 = ReasoningStep(step_id=0, content="Given: 2x + 3 = 7")
        step2 = ReasoningStep(step_id=1, content="Subtract 3: 2x = 4", dependencies=[0])
        
        path.add_step(step1)
        path.add_step(step2)
        
        assert len(path.steps) == 2
        assert path.steps[0] == step1
        assert path.steps[1] == step2
    
    def test_get_step_by_id(self):
        """Test retrieving step by ID."""
        step1 = ReasoningStep(step_id=0, content="Step 1")
        step2 = ReasoningStep(step_id=1, content="Step 2")
        
        path = ReasoningPath(steps=[step1, step2])
        
        assert path.get_step_by_id(0) == step1
        assert path.get_step_by_id(1) == step2
        assert path.get_step_by_id(2) is None
    
    def test_dependency_validation(self):
        """Test dependency validation."""
        step1 = ReasoningStep(step_id=0, content="Step 1")
        step2 = ReasoningStep(step_id=1, content="Step 2", dependencies=[0])
        step3 = ReasoningStep(step_id=2, content="Step 3", dependencies=[5])  # Invalid dependency
        
        path = ReasoningPath(steps=[step1, step2, step3])
        
        issues = path.validate_dependencies()
        assert len(issues) == 1
        assert "depends on non-existent step 5" in issues[0]
    
    def test_confidence_statistics(self):
        """Test confidence statistics calculation."""
        step1 = ReasoningStep(step_id=0, content="Step 1")
        step1.add_uncertainty_score("selfcheck", 0.8)
        
        step2 = ReasoningStep(step_id=1, content="Step 2")
        step2.add_uncertainty_score("selfcheck", 0.6)
        
        path = ReasoningPath(steps=[step1, step2])
        
        stats = path.get_confidence_statistics("selfcheck")
        assert stats["mean"] == 0.7
        assert stats["min"] == 0.6
        assert stats["max"] == 0.8
        assert stats["std"] > 0  # Should be positive for different values
    
    def test_path_serialization(self):
        """Test path serialization."""
        step1 = ReasoningStep(step_id=0, content="Given equation")
        step1.add_uncertainty_score("selfcheck", 0.9)
        
        step2 = ReasoningStep(step_id=1, content="Solve for x", dependencies=[0])
        step2.add_uncertainty_score("selfcheck", 0.8)
        
        path = ReasoningPath(
            steps=[step1, step2],
            is_complete=True,
            path_id="test_path"
        )
        
        # Convert to dict
        path_dict = path.to_dict()
        assert path_dict["is_complete"] is True
        assert path_dict["path_id"] == "test_path"
        assert len(path_dict["steps"]) == 2
        
        # Convert back
        restored_path = ReasoningPath.from_dict(path_dict)
        assert len(restored_path.steps) == 2
        assert restored_path.is_complete is True
        assert restored_path.path_id == "test_path"
    
    def test_path_iteration(self):
        """Test path iteration and indexing."""
        step1 = ReasoningStep(step_id=0, content="Step 1")
        step2 = ReasoningStep(step_id=1, content="Step 2")
        step3 = ReasoningStep(step_id=2, content="Step 3")
        
        path = ReasoningPath(steps=[step1, step2, step3])
        
        # Test length
        assert len(path) == 3
        
        # Test indexing
        assert path[0] == step1
        assert path[1] == step2
        assert path[-1] == step3
        
        # Test slicing
        assert path[0:2] == [step1, step2]
        
        # Test iteration
        steps = list(path)
        assert steps == [step1, step2, step3]


if __name__ == "__main__":
    pytest.main([__file__])