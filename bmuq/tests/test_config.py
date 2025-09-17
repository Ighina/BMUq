"""
Tests for configuration management.
"""

import pytest
import tempfile
import json
from pathlib import Path
from bmuq.config import (
    BMUqConfig, LLMConfig, UncertaintyConfig, SearchConfig, BenchmarkConfig,
    save_config, load_config, get_preset_config, list_available_presets
)


class TestLLMConfig:
    """Test LLM configuration."""
    
    def test_default_config(self):
        """Test default LLM configuration."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4-turbo-preview"
        assert config.temperature == 0.7
        assert config.max_tokens == 150
        assert config.max_retries == 3
    
    def test_custom_config(self):
        """Test custom LLM configuration."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3",
            temperature=0.5,
            api_key="test-key"
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-3"
        assert config.temperature == 0.5
        assert config.api_key == "test-key"


class TestUncertaintyConfig:
    """Test uncertainty configuration."""
    
    def test_default_config(self):
        """Test default uncertainty configuration."""
        config = UncertaintyConfig()
        assert config.method == "selfcheck"
        assert config.lambda_neg1 == 1.0
        assert config.lambda_0 == 0.3
        assert config.num_samples == 5
    
    def test_entropy_config(self):
        """Test entropy-based configuration."""
        config = UncertaintyConfig(
            method="entropy_based",
            num_samples=10,
            sampling_temperature=0.9
        )
        assert config.method == "entropy_based"
        assert config.num_samples == 10
        assert config.sampling_temperature == 0.9


class TestSearchConfig:
    """Test search configuration."""
    
    def test_default_config(self):
        """Test default search configuration."""
        config = SearchConfig()
        assert config.algorithm == "tree_search"
        assert config.beam_width == 3
        assert config.max_depth == 8
        assert config.confidence_threshold == 0.1
    
    def test_beam_search_config(self):
        """Test beam search configuration."""
        config = SearchConfig(
            algorithm="beam_search",
            beam_width=5,
            diversity_penalty=0.2
        )
        assert config.algorithm == "beam_search"
        assert config.beam_width == 5
        assert config.diversity_penalty == 0.2


class TestBenchmarkConfig:
    """Test benchmark configuration."""
    
    def test_default_config(self):
        """Test default benchmark configuration."""
        config = BenchmarkConfig()
        assert config.dataset == "custom"
        assert config.shuffle is True
        assert config.seed == 42
        assert "accuracy" in config.metrics
    
    def test_custom_config(self):
        """Test custom benchmark configuration."""
        config = BenchmarkConfig(
            dataset="gsm8k",
            num_questions=100,
            output_dir="custom_results",
            metrics=["accuracy", "confidence_correlation"]
        )
        assert config.dataset == "gsm8k"
        assert config.num_questions == 100
        assert config.output_dir == "custom_results"
        assert len(config.metrics) == 2


class TestBMUqConfig:
    """Test main BMUq configuration."""
    
    def test_default_config(self):
        """Test default BMUq configuration."""
        config = BMUqConfig()
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.uncertainty, UncertaintyConfig)
        assert isinstance(config.search, SearchConfig)
        assert isinstance(config.benchmark, BenchmarkConfig)
        assert config.experiment_name == "default_experiment"
        assert config.random_seed == 42
    
    def test_custom_config(self):
        """Test custom BMUq configuration."""
        llm_config = LLMConfig(provider="mock", model="mock-llm")
        uncertainty_config = UncertaintyConfig(method="entropy_based")
        
        config = BMUqConfig(
            llm=llm_config,
            uncertainty=uncertainty_config,
            experiment_name="test_experiment",
            description="Test configuration"
        )
        
        assert config.llm.provider == "mock"
        assert config.uncertainty.method == "entropy_based"
        assert config.experiment_name == "test_experiment"
        assert config.description == "Test configuration"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = BMUqConfig()
        issues = config.validate()
        assert len(issues) == 0
        
        # Invalid temperature
        config.llm.temperature = 3.0
        issues = config.validate()
        assert any("temperature" in issue for issue in issues)
        
        # Invalid uncertainty method
        config.llm.temperature = 0.7  # Fix temperature
        config.uncertainty.method = "invalid_method"
        issues = config.validate()
        assert any("Uncertainty method" in issue for issue in issues)
        
        # Invalid beam width
        config.uncertainty.method = "selfcheck"  # Fix method
        config.search.beam_width = 0
        issues = config.validate()
        assert any("Beam width" in issue for issue in issues)
    
    def test_config_dict_conversion(self):
        """Test configuration to/from dictionary conversion."""
        config = BMUqConfig(experiment_name="test_conversion")
        
        # Convert to dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["experiment_name"] == "test_conversion"
        assert "llm" in config_dict
        assert "uncertainty" in config_dict
        
        # Convert back from dict
        restored_config = BMUqConfig.from_dict(config_dict)
        assert restored_config.experiment_name == config.experiment_name
        assert restored_config.llm.provider == config.llm.provider
        assert restored_config.uncertainty.method == config.uncertainty.method


class TestConfigFilesIO:
    """Test configuration file save/load operations."""
    
    def test_save_load_yaml(self):
        """Test saving and loading YAML configuration."""
        config = BMUqConfig(
            experiment_name="yaml_test",
            description="Test YAML save/load"
        )
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = f.name
        
        try:
            # Save configuration
            save_config(config, config_path, format="yaml")
            assert Path(config_path).exists()
            
            # Load configuration
            loaded_config = load_config(config_path)
            assert loaded_config.experiment_name == config.experiment_name
            assert loaded_config.description == config.description
            
            # Validate loaded config
            issues = loaded_config.validate()
            assert len(issues) == 0
            
        finally:
            Path(config_path).unlink()
    
    def test_save_load_json(self):
        """Test saving and loading JSON configuration."""
        config = BMUqConfig(
            experiment_name="json_test",
            random_seed=123
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config_path = f.name
        
        try:
            # Save configuration
            save_config(config, config_path, format="json")
            assert Path(config_path).exists()
            
            # Load configuration
            loaded_config = load_config(config_path)
            assert loaded_config.experiment_name == config.experiment_name
            assert loaded_config.random_seed == config.random_seed
            
        finally:
            Path(config_path).unlink()
    
    def test_invalid_format(self):
        """Test handling of invalid file format."""
        config = BMUqConfig()
        
        with pytest.raises(ValueError, match="Unsupported format"):
            save_config(config, "test.txt", format="txt")
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as f:
            f.write("{invalid json content")
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Failed to parse"):
                load_config(config_path)
        finally:
            Path(config_path).unlink()


class TestPresets:
    """Test preset configurations."""
    
    def test_list_presets(self):
        """Test listing available presets."""
        presets = list_available_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0
        
        # Check that default preset exists
        preset_names = [p["name"] for p in presets]
        assert "default" in preset_names
        assert "selfcheck_tree" in preset_names
    
    def test_get_preset_config(self):
        """Test getting preset configuration."""
        config = get_preset_config("default")
        assert isinstance(config, BMUqConfig)
        assert config.experiment_name == "default_experiment"
        
        # Validate preset
        issues = config.validate()
        assert len(issues) == 0
    
    def test_selfcheck_tree_preset(self):
        """Test SelfCheck tree search preset."""
        config = get_preset_config("selfcheck_tree")
        assert config.uncertainty.method == "selfcheck"
        assert config.search.algorithm == "tree_search"
        assert config.llm.provider == "openai"
    
    def test_fast_development_preset(self):
        """Test fast development preset."""
        config = get_preset_config("fast_development")
        assert config.llm.provider == "mock"
        assert config.uncertainty.method == "random_baseline"
        assert config.benchmark.num_questions == 10
    
    def test_invalid_preset(self):
        """Test handling of invalid preset name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("nonexistent_preset")
    
    def test_custom_preset_modification(self):
        """Test creating custom preset by modifying existing one."""
        from bmuq.config.presets import create_custom_preset
        
        modifications = {
            "search.beam_width": 5,
            "llm.temperature": 0.3,
            "experiment_name": "modified_preset"
        }
        
        config = create_custom_preset("selfcheck_tree", modifications)
        
        assert config.search.beam_width == 5
        assert config.llm.temperature == 0.3
        assert config.experiment_name == "modified_preset"
        
        # Original settings should remain
        assert config.uncertainty.method == "selfcheck"
        assert config.search.algorithm == "tree_search"


if __name__ == "__main__":
    pytest.main([__file__])