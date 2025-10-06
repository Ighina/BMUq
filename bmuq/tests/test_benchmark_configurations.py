"""
Test suite for validating various BMUq configurations work with the benchmark interface.

This test suite creates multiple configuration variations and validates that:
1. Each configuration can be instantiated properly
2. The benchmark interface accepts and processes the configuration
3. The benchmark runs without errors
4. Results are generated in expected format
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from bmuq.config.settings import (
    BMUqConfig,
    LLMConfig,
    UncertaintyConfig,
    SearchConfig,
    BenchmarkConfig,
    save_config,
    load_config
)
from bmuq.benchmarks.benchmark import BMUqBenchmark, BenchmarkResult
from bmuq.benchmarks.datasets import Dataset


class TestBenchmarkConfigurations:
    """Test various configurations with the benchmark interface."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset for testing."""
        return Dataset(
            name="test_dataset",
            questions=[
                {"question": "What is 2+2?", "answer": "4", "id": 0},
                {"question": "What is 3*5?", "answer": "15", "id": 1},
                {"question": "What is 10/2?", "answer": "5", "id": 2},
            ]
        )

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def create_base_config(self) -> BMUqConfig:
        """Create a base configuration for testing."""
        return BMUqConfig(
            llm=LLMConfig(
                provider="mock",
                model="mock-llm",
                temperature=0.5,
                max_tokens=100,
                max_retries=1
            ),
            uncertainty=UncertaintyConfig(method="random_baseline"),
            search=SearchConfig(
                algorithm="beam_search",
                beam_width=2,
                max_depth=3
            ),
            benchmark=BenchmarkConfig(
                dataset="custom",
                num_questions=3,
                shuffle=False,
                verbose=False,
                metrics=["accuracy"]
            ),
            experiment_name="test_config",
            random_seed=42
        )

    def test_basic_configuration_validation(self):
        """Test that basic configuration is valid."""
        config = self.create_base_config()
        issues = config.validate()
        assert len(issues) == 0, f"Configuration validation failed: {issues}"

    def test_all_llm_providers_configuration(self):
        """Test configurations for all LLM providers."""
        providers = ["mock", "openai", "huggingface", "ollama"]

        for provider in providers:
            config = self.create_base_config()
            config.llm.provider = provider
            config.experiment_name = f"test_{provider}"

            # Set provider-specific configurations
            if provider == "openai":
                config.llm.model = "gpt-3.5-turbo"
                config.llm.api_key = "test-key"
            elif provider == "huggingface":
                config.llm.model = "microsoft/DialoGPT-medium"
                config.llm.hf_device = "cpu"
            elif provider == "ollama":
                config.llm.model = "llama2"
                config.llm.ollama_base_url = "http://localhost:11434"

            issues = config.validate()
            assert len(issues) == 0, f"Configuration for {provider} failed validation: {issues}"

    def test_all_uncertainty_methods_configuration(self):
        """Test configurations for all uncertainty methods."""
        uncertainty_methods = [
            "selfcheck",
            "entropy_based",
            "consistency_based",
            "random_baseline",
            "semantic_entropy",
            "coherence_based"
        ]

        for method in uncertainty_methods:
            config = self.create_base_config()
            config.uncertainty.method = method
            config.experiment_name = f"test_{method}"

            # Set method-specific parameters
            if method == "semantic_entropy":
                config.uncertainty.entailment_model = "cross-encoder/nli-roberta-base"
            elif method == "coherence_based":
                config.uncertainty.extra_params = {
                    "coherence_method": "mean_cosine_similarity",
                    "model_name": "all-MiniLM-L6-v2"
                }

            issues = config.validate()
            assert len(issues) == 0, f"Configuration for {method} failed validation: {issues}"

    def test_all_search_algorithms_configuration(self):
        """Test configurations for all search algorithms."""
        search_algorithms = ["tree_search", "beam_search", "best_of_n"]

        for algorithm in search_algorithms:
            config = self.create_base_config()
            config.search.algorithm = algorithm
            config.experiment_name = f"test_{algorithm}"

            issues = config.validate()
            assert len(issues) == 0, f"Configuration for {algorithm} failed validation: {issues}"

    @patch('bmuq.models.mock_llm.MockLLM')
    def test_benchmark_instantiation_with_mock_llm(self, mock_llm_class, mock_dataset):
        """Test benchmark instantiation with various configurations."""
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_llm.get_usage_stats.return_value = MagicMock(
            total_tokens=100,
            total_cost=0.001,
            to_dict=lambda: {"total_tokens": 100, "total_cost": 0.001}
        )
        mock_llm_class.return_value = mock_llm

        config = self.create_base_config()

        try:
            benchmark = BMUqBenchmark(config)
            assert benchmark is not None
            assert benchmark.config == config
            assert benchmark.llm is not None
            assert benchmark.uncertainty_method is not None
            assert benchmark.search_algorithm is not None
        except Exception as e:
            pytest.fail(f"Failed to instantiate benchmark: {e}")

    @patch('bmuq.models.mock_llm.MockLLM')
    @patch('bmuq.search.beam_search.BeamSearchCoT')
    @patch('bmuq.uncertainty.base_methods.RandomBaselineUQ')
    def test_benchmark_run_with_mock_components(
        self,
        mock_uncertainty,
        mock_search,
        mock_llm_class,
        mock_dataset,
        temp_output_dir
    ):
        """Test running benchmark with mocked components."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_llm.get_usage_stats.return_value = MagicMock(
            total_tokens=100,
            total_cost=0.001,
            to_dict=lambda: {"total_tokens": 100, "total_cost": 0.001}
        )
        mock_llm_class.return_value = mock_llm

        # Mock search algorithm
        mock_search_instance = MagicMock()
        mock_path = MagicMock()
        mock_path.total_confidence = 0.8
        mock_path.steps = [MagicMock(), MagicMock()]
        mock_search_instance.search.return_value = [mock_path]
        mock_search_instance.get_search_statistics.return_value = {"total_searches": 3}
        mock_search.return_value = mock_search_instance

        # Mock uncertainty method
        mock_uncertainty_instance = MagicMock()
        mock_uncertainty_instance.get_method_info.return_value = {"method": "random_baseline"}
        mock_uncertainty.return_value = mock_uncertainty_instance

        # Mock evaluator results
        with patch('bmuq.benchmarks.evaluator.Evaluator') as mock_evaluator_class:
            mock_evaluator = MagicMock()
            mock_evaluator.evaluate_question.return_value = {
                "question_id": 0,
                "success": True,
                "confidence": 0.8,
                "path_length": 2,
                "predicted_answer": "4",
                "ground_truth": "4",
                "correct": True
            }
            mock_evaluator_class.return_value = mock_evaluator

            # Mock metrics calculation
            with patch('bmuq.benchmarks.metrics.calculate_metrics') as mock_metrics:
                mock_metrics.return_value = {
                    "accuracy": MagicMock(value=1.0, to_dict=lambda: {"value": 1.0})
                }

                config = self.create_base_config()
                config.benchmark.output_dir = temp_output_dir
                config.benchmark.num_questions = 1

                benchmark = BMUqBenchmark(config)
                result = benchmark.run(
                    dataset=mock_dataset,
                    num_questions=1,
                    save_results=True
                )

                # Validate result structure
                assert isinstance(result, BenchmarkResult)
                assert result.experiment_name == config.experiment_name
                assert result.success_rate >= 0.0
                assert result.success_rate <= 1.0
                assert len(result.question_results) > 0
                assert "accuracy" in result.metrics

    def test_configuration_comparison_mode(self):
        """Test configuration comparison functionality."""
        config = self.create_base_config()
        methods = ["random_baseline", "consistency_based"]

        # Mock the benchmark run to avoid actual LLM calls
        with patch.object(BMUqBenchmark, 'run') as mock_run:
            mock_result = MagicMock(spec=BenchmarkResult)
            mock_result.success_rate = 0.5
            mock_result.metrics = {"accuracy": MagicMock(value=0.5)}
            mock_run.return_value = mock_result

            benchmark = BMUqBenchmark(config)
            results = benchmark.run_comparison(
                uncertainty_methods=methods,
                num_questions=2
            )

            assert isinstance(results, dict)
            assert len(results) == len(methods)
            for method in methods:
                assert method in results

    def test_configuration_serialization_roundtrip(self, temp_output_dir):
        """Test that configurations can be saved and loaded properly."""
        original_config = self.create_base_config()
        config_path = Path(temp_output_dir) / "test_config.yaml"

        # Save configuration
        save_config(original_config, config_path, format="yaml")
        assert config_path.exists()

        # Load configuration
        loaded_config = load_config(config_path)

        # Validate loaded configuration
        assert loaded_config.experiment_name == original_config.experiment_name
        assert loaded_config.llm.provider == original_config.llm.provider
        assert loaded_config.llm.model == original_config.llm.model
        assert loaded_config.uncertainty.method == original_config.uncertainty.method
        assert loaded_config.search.algorithm == original_config.search.algorithm

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        # Test invalid LLM provider
        config = self.create_base_config()
        config.llm.provider = "invalid_provider"
        issues = config.validate()
        assert len(issues) > 0

        # Test invalid uncertainty method
        config = self.create_base_config()
        config.uncertainty.method = "invalid_method"
        issues = config.validate()
        assert len(issues) > 0

        # Test invalid search algorithm
        config = self.create_base_config()
        config.search.algorithm = "invalid_algorithm"
        issues = config.validate()
        assert len(issues) > 0

    def test_configuration_edge_cases(self):
        """Test edge cases in configuration."""
        config = self.create_base_config()

        # Test minimum values
        config.search.beam_width = 1
        config.search.max_depth = 1
        config.llm.temperature = 0.0
        issues = config.validate()
        assert len(issues) == 0

        # Test maximum values
        config.llm.temperature = 2.0
        issues = config.validate()
        assert len(issues) == 0

        # Test boundary violations
        config.search.beam_width = 0  # Should be invalid
        issues = config.validate()
        assert len(issues) > 0

        config.search.beam_width = 1  # Fix it
        config.llm.temperature = 3.0  # Should be invalid
        issues = config.validate()
        assert len(issues) > 0

    @pytest.mark.parametrize("provider,method,algorithm", [
        ("mock", "selfcheck", "tree_search"),
        ("mock", "entropy_based", "beam_search"),
        ("mock", "consistency_based", "best_of_n"),
        ("mock", "random_baseline", "tree_search"),
    ])
    def test_configuration_combinations(self, provider, method, algorithm):
        """Test various combinations of configuration options."""
        config = BMUqConfig(
            llm=LLMConfig(
                provider=provider,
                model="mock-llm" if provider == "mock" else "gpt-3.5-turbo",
                temperature=0.5
            ),
            uncertainty=UncertaintyConfig(method=method),
            search=SearchConfig(
                algorithm=algorithm,
                beam_width=2,
                max_depth=3
            ),
            benchmark=BenchmarkConfig(
                dataset="custom",
                num_questions=1
            ),
            experiment_name=f"test_{provider}_{method}_{algorithm}"
        )

        issues = config.validate()
        assert len(issues) == 0, f"Configuration failed: {issues}"

        # Test instantiation
        try:
            benchmark = BMUqBenchmark(config)
            assert benchmark is not None
        except Exception as e:
            # Some combinations might fail due to missing dependencies
            # This is expected behavior, not a test failure
            if "not installed" in str(e) or "Unsupported" in str(e):
                pytest.skip(f"Skipping due to missing dependency: {e}")
            else:
                pytest.fail(f"Unexpected error: {e}")


class TestConfigurationIntegration:
    """Integration tests for configurations with actual benchmark runs."""

    def test_end_to_end_mock_run(self):
        """Test complete end-to-end run with mock components."""
        config = BMUqConfig(
            llm=LLMConfig(provider="mock", model="mock-llm"),
            uncertainty=UncertaintyConfig(method="random_baseline"),
            search=SearchConfig(algorithm="beam_search", beam_width=1, max_depth=2),
            benchmark=BenchmarkConfig(
                dataset="custom",
                num_questions=1,
                verbose=False,
                save_intermediate=False
            ),
            experiment_name="integration_test"
        )

        # Create simple test data
        test_dataset = Dataset(
            name="integration_test",
            questions=[{"question": "Test question", "answer": "Test answer", "id": 0}]
        )

        with patch('bmuq.models.mock_llm.MockLLM') as mock_llm_class:
            # Setup comprehensive mocks for the entire pipeline
            mock_llm = MagicMock()
            mock_llm.get_usage_stats.return_value = MagicMock(
                total_tokens=10,
                total_cost=0.001,
                to_dict=lambda: {"total_tokens": 10, "total_cost": 0.001}
            )
            mock_llm_class.return_value = mock_llm

            with patch('bmuq.search.beam_search.BeamSearchCoT') as mock_search:
                mock_search_instance = MagicMock()
                mock_path = MagicMock()
                mock_path.total_confidence = 0.5
                mock_path.steps = [MagicMock()]
                mock_search_instance.search.return_value = [mock_path]
                mock_search_instance.get_search_statistics.return_value = {}
                mock_search.return_value = mock_search_instance

                with patch('bmuq.uncertainty.base_methods.RandomBaselineUQ') as mock_uq:
                    mock_uq_instance = MagicMock()
                    mock_uq_instance.get_method_info.return_value = {"method": "random_baseline"}
                    mock_uq.return_value = mock_uq_instance

                    with patch('bmuq.benchmarks.evaluator.Evaluator') as mock_evaluator_class:
                        mock_evaluator = MagicMock()
                        mock_evaluator.evaluate_question.return_value = {
                            "question_id": 0,
                            "success": True,
                            "confidence": 0.5,
                            "path_length": 1
                        }
                        mock_evaluator_class.return_value = mock_evaluator

                        with patch('bmuq.benchmarks.metrics.calculate_metrics') as mock_metrics:
                            mock_metrics.return_value = {
                                "accuracy": MagicMock(
                                    value=1.0,
                                    to_dict=lambda: {"value": 1.0}
                                )
                            }

                            benchmark = BMUqBenchmark(config)
                            result = benchmark.run(
                                dataset=test_dataset,
                                save_results=False
                            )

                            # Validate complete result structure
                            assert isinstance(result, BenchmarkResult)
                            assert result.experiment_name == "integration_test"
                            assert 0.0 <= result.success_rate <= 1.0
                            assert len(result.question_results) == 1
                            assert isinstance(result.llm_stats, dict)
                            assert isinstance(result.metadata, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])