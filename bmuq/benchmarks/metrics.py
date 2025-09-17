"""
Evaluation metrics for uncertainty quantification methods.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_recall_curve, auc


@dataclass
class MetricResult:
    """Result from computing a metric."""
    name: str
    value: float
    description: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "metadata": self.metadata
        }


class Metric(ABC):
    """Abstract base class for evaluation metrics."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def compute(self, results: List[Dict[str, Any]]) -> MetricResult:
        """
        Compute metric from evaluation results.
        
        Args:
            results: List of question evaluation results
            
        Returns:
            MetricResult containing computed metric
        """
        pass


class AccuracyMetric(Metric):
    """Accuracy metric for answer correctness."""
    
    def __init__(self):
        super().__init__(
            name="accuracy",
            description="Fraction of questions answered correctly"
        )
    
    def compute(self, results: List[Dict[str, Any]]) -> MetricResult:
        """Compute accuracy from results."""
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return MetricResult(
                name=self.name,
                value=0.0,
                description=self.description,
                metadata={"num_questions": len(results), "num_successful": 0}
            )
        
        correct_count = sum(1 for r in successful_results if r.get("correct", False))
        accuracy = correct_count / len(successful_results)
        
        return MetricResult(
            name=self.name,
            value=accuracy,
            description=self.description,
            metadata={
                "num_questions": len(results),
                "num_successful": len(successful_results),
                "num_correct": correct_count
            }
        )


class ConfidenceCorrelationMetric(Metric):
    """Correlation between confidence and correctness."""
    
    def __init__(self):
        super().__init__(
            name="confidence_correlation",
            description="Correlation between predicted confidence and answer correctness"
        )
    
    def compute(self, results: List[Dict[str, Any]]) -> MetricResult:
        """Compute confidence-correctness correlation."""
        successful_results = [r for r in results if r.get("success", False) and "confidence" in r]
        
        if len(successful_results) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                description=self.description,
                metadata={"insufficient_data": True, "num_points": len(successful_results)}
            )
        
        confidences = [r["confidence"] for r in successful_results]
        correctness = [1.0 if r.get("correct", False) else 0.0 for r in successful_results]
        
        # Compute Pearson and Spearman correlations
        pearson_corr, pearson_p = pearsonr(confidences, correctness)
        spearman_corr, spearman_p = spearmanr(confidences, correctness)
        
        return MetricResult(
            name=self.name,
            value=pearson_corr,
            description=self.description,
            metadata={
                "pearson_correlation": pearson_corr,
                "pearson_p_value": pearson_p,
                "spearman_correlation": spearman_corr,
                "spearman_p_value": spearman_p,
                "num_points": len(successful_results)
            }
        )


class UncertaintyQualityMetric(Metric):
    """Quality of uncertainty estimates using AUPR."""
    
    def __init__(self):
        super().__init__(
            name="uncertainty_quality",
            description="Area under precision-recall curve for uncertainty estimates"
        )
    
    def compute(self, results: List[Dict[str, Any]]) -> MetricResult:
        """Compute uncertainty quality using AUPR."""
        successful_results = [r for r in results if r.get("success", False) and "confidence" in r]
        
        if len(successful_results) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                description=self.description,
                metadata={"insufficient_data": True}
            )
        
        # Use 1 - confidence as uncertainty score
        uncertainties = [1.0 - r["confidence"] for r in successful_results]
        # Use incorrectness as positive class (high uncertainty should correlate with incorrectness)
        incorrectness = [0.0 if r.get("correct", False) else 1.0 for r in successful_results]
        
        if sum(incorrectness) == 0:
            # All answers are correct, perfect scenario
            return MetricResult(
                name=self.name,
                value=1.0,
                description=self.description,
                metadata={"all_correct": True, "num_points": len(successful_results)}
            )
        
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(incorrectness, uncertainties)
        aupr = auc(recall, precision)
        
        return MetricResult(
            name=self.name,
            value=aupr,
            description=self.description,
            metadata={
                "aupr": aupr,
                "num_points": len(successful_results),
                "num_incorrect": sum(incorrectness)
            }
        )


class PathDiversityMetric(Metric):
    """Diversity of reasoning paths found."""
    
    def __init__(self):
        super().__init__(
            name="path_diversity",
            description="Average diversity of reasoning paths"
        )
    
    def compute(self, results: List[Dict[str, Any]]) -> MetricResult:
        """Compute path diversity."""
        successful_results = [r for r in results if r.get("success", False) and "reasoning_path" in r]
        
        if len(successful_results) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                description=self.description,
                metadata={"insufficient_data": True}
            )
        
        # Simple diversity metric based on path length variation
        path_lengths = [len(r["reasoning_path"]) for r in successful_results]
        length_diversity = np.std(path_lengths) / (np.mean(path_lengths) + 1e-8)
        
        # Could be enhanced with semantic similarity measures
        
        return MetricResult(
            name=self.name,
            value=length_diversity,
            description=self.description,
            metadata={
                "path_length_std": np.std(path_lengths),
                "path_length_mean": np.mean(path_lengths),
                "num_paths": len(successful_results)
            }
        )


class SearchEfficiencyMetric(Metric):
    """Efficiency of search algorithm."""
    
    def __init__(self):
        super().__init__(
            name="search_efficiency",
            description="Efficiency of search measured as accuracy per unit time"
        )
    
    def compute(self, results: List[Dict[str, Any]]) -> MetricResult:
        """Compute search efficiency."""
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return MetricResult(
                name=self.name,
                value=0.0,
                description=self.description,
                metadata={"no_successful_results": True}
            )
        
        # Calculate average time per question
        total_time = sum(r.get("time_seconds", 0) for r in successful_results)
        avg_time = total_time / len(successful_results)
        
        # Calculate accuracy
        correct_count = sum(1 for r in successful_results if r.get("correct", False))
        accuracy = correct_count / len(successful_results)
        
        # Efficiency = accuracy / time (higher is better)
        efficiency = accuracy / max(avg_time, 1e-8)
        
        return MetricResult(
            name=self.name,
            value=efficiency,
            description=self.description,
            metadata={
                "accuracy": accuracy,
                "avg_time_seconds": avg_time,
                "total_time_seconds": total_time,
                "num_questions": len(successful_results)
            }
        )


class MethodComparisonMetric(Metric):
    """Compare multiple uncertainty quantification methods."""
    
    def __init__(self):
        super().__init__(
            name="method_comparison",
            description="Comparative analysis of uncertainty quantification methods"
        )
    
    def compute(self, results: List[Dict[str, Any]]) -> MetricResult:
        """
        Compare methods if multiple method results are available.
        
        Expected format: results should contain "method" field indicating which method was used.
        """
        # Group results by method
        method_groups = {}
        for result in results:
            method = result.get("method", "unknown")
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(result)
        
        if len(method_groups) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                description=self.description,
                metadata={"insufficient_methods": True, "methods_found": list(method_groups.keys())}
            )
        
        # Compute metrics for each method
        method_metrics = {}
        for method, method_results in method_groups.items():
            successful = [r for r in method_results if r.get("success", False)]
            if successful:
                accuracy = sum(1 for r in successful if r.get("correct", False)) / len(successful)
                avg_confidence = sum(r.get("confidence", 0) for r in successful) / len(successful)
                method_metrics[method] = {
                    "accuracy": accuracy,
                    "avg_confidence": avg_confidence,
                    "num_questions": len(successful)
                }
        
        # Find best performing method by accuracy
        best_method = max(method_metrics.items(), key=lambda x: x[1]["accuracy"])
        best_accuracy = best_method[1]["accuracy"]
        
        return MetricResult(
            name=self.name,
            value=best_accuracy,
            description=self.description,
            metadata={
                "method_metrics": method_metrics,
                "best_method": best_method[0],
                "num_methods_compared": len(method_metrics)
            }
        )


# Registry of available metrics
AVAILABLE_METRICS = {
    "accuracy": AccuracyMetric,
    "confidence_correlation": ConfidenceCorrelationMetric,
    "uncertainty_quality": UncertaintyQualityMetric,
    "path_diversity": PathDiversityMetric,
    "search_efficiency": SearchEfficiencyMetric,
    "method_comparison": MethodComparisonMetric,
}


def calculate_metrics(results: List[Dict[str, Any]], 
                     metric_names: List[str]) -> Dict[str, MetricResult]:
    """
    Calculate specified metrics from evaluation results.
    
    Args:
        results: List of question evaluation results
        metric_names: List of metric names to compute
        
    Returns:
        Dictionary mapping metric names to MetricResult objects
        
    Raises:
        ValueError: If unknown metric name is specified
    """
    computed_metrics = {}
    
    for metric_name in metric_names:
        if metric_name not in AVAILABLE_METRICS:
            available = list(AVAILABLE_METRICS.keys())
            raise ValueError(f"Unknown metric '{metric_name}'. Available metrics: {available}")
        
        try:
            metric_class = AVAILABLE_METRICS[metric_name]
            metric = metric_class()
            result = metric.compute(results)
            computed_metrics[metric_name] = result
        except Exception as e:
            print(f"Error computing metric '{metric_name}': {e}")
            # Add error result
            computed_metrics[metric_name] = MetricResult(
                name=metric_name,
                value=0.0,
                description=f"Error computing {metric_name}",
                metadata={"error": str(e)}
            )
    
    return computed_metrics


def get_metric_info() -> Dict[str, Dict[str, str]]:
    """Get information about available metrics."""
    info = {}
    for name, metric_class in AVAILABLE_METRICS.items():
        metric = metric_class()
        info[name] = {
            "name": metric.name,
            "description": metric.description
        }
    return info