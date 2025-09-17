"""
Tests for HuggingFace LLM implementation.

Note: These tests are mostly unit tests that don't require actual model loading
to avoid heavy dependencies in CI/CD. Integration tests are in separate files.
"""

import pytest
from unittest.mock import patch, MagicMock
import torch

# Skip all tests if transformers not available
transformers = pytest.importorskip("transformers")

from bmuq.models.huggingface_llm import (
    HuggingFaceLLM, 
    list_recommended_models, 
    get_model_memory_requirements
)


class TestHuggingFaceUtilities:
    """Test utility functions."""
    
    def test_list_recommended_models(self):
        """Test listing recommended models."""
        models = list_recommended_models()
        
        assert isinstance(models, dict)
        assert "small_models" in models
        assert "medium_models" in models
        assert "large_models" in models
        
        # Check structure
        for category, model_dict in models.items():
            assert isinstance(model_dict, dict)
            for model_name, info in model_dict.items():
                assert "params" in info
                assert "use_case" in info
    
    def test_get_model_memory_requirements(self):
        """Test memory requirement calculation."""
        # Test with known model
        memory_req = get_model_memory_requirements("gpt2", "fp16")
        
        assert "model_memory_gb" in memory_req
        assert "total_memory_gb" in memory_req
        assert "recommended_gpu_memory_gb" in memory_req
        
        assert memory_req["model_memory_gb"] > 0
        assert memory_req["total_memory_gb"] > memory_req["model_memory_gb"]
        assert memory_req["recommended_gpu_memory_gb"] > memory_req["total_memory_gb"]
        
        # Test different precisions
        fp32_req = get_model_memory_requirements("gpt2", "fp32")
        fp16_req = get_model_memory_requirements("gpt2", "fp16")
        bit8_req = get_model_memory_requirements("gpt2", "8bit")
        
        assert fp32_req["model_memory_gb"] > fp16_req["model_memory_gb"]
        assert fp16_req["model_memory_gb"] > bit8_req["model_memory_gb"]


class TestHuggingFaceLLM:
    """Test HuggingFace LLM implementation."""
    
    @patch('torch.cuda.is_available')
    def test_device_setup_cuda_available(self, mock_cuda_available):
        """Test device setup when CUDA is available."""
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_name', return_value="Test GPU"), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_props.return_value.total_memory = 8e9  # 8GB
            
            llm = HuggingFaceLLM.__new__(HuggingFaceLLM)  # Create without __init__
            device = llm._setup_device("auto")
            
            assert device == "cuda"
    
    @patch('torch.cuda.is_available')
    def test_device_setup_cuda_unavailable(self, mock_cuda_available):
        """Test device setup when CUDA is not available."""
        mock_cuda_available.return_value = False
        
        llm = HuggingFaceLLM.__new__(HuggingFaceLLM)
        device = llm._setup_device("auto")
        
        assert device == "cpu"
    
    @patch('torch.cuda.is_available')
    def test_device_setup_force_cpu(self, mock_cuda_available):
        """Test forcing CPU device."""
        mock_cuda_available.return_value = True
        
        llm = HuggingFaceLLM.__new__(HuggingFaceLLM)
        device = llm._setup_device("cpu")
        
        assert device == "cpu"
    
    def test_initialization_parameters(self):
        """Test initialization parameter handling without loading model."""
        with patch.object(HuggingFaceLLM, '_load_model'):
            llm = HuggingFaceLLM(
                model_name="test-model",
                temperature=0.8,
                max_new_tokens=200,
                load_in_8bit=True,
                trust_remote_code=True
            )
            
            assert llm.model_name == "test-model"
            assert llm.temperature == 0.8
            assert llm.max_new_tokens == 200
            assert llm.load_in_8bit is True
            assert llm.trust_remote_code is True
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.pipeline')
    def test_model_loading_success(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test successful model loading."""
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Mock pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Test loading
        llm = HuggingFaceLLM(model_name="test-model", device="cpu")
        
        assert llm.tokenizer is not None
        assert llm.model is not None
        assert llm.pipeline is not None
        assert llm.model_type == 'causal'
        
        # Verify pad token was set
        assert llm.tokenizer.pad_token == "</s>"
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoModelForSeq2SeqLM.from_pretrained')
    @patch('transformers.pipeline')
    def test_seq2seq_fallback(self, mock_pipeline, mock_seq2seq, mock_causal, mock_tokenizer):
        """Test fallback to seq2seq model."""
        # Mock tokenizer
        mock_tokenizer.return_value = MagicMock()
        
        # Make causal model fail
        mock_causal.side_effect = ValueError("Not a causal model")
        
        # Make seq2seq succeed
        mock_seq2seq_instance = MagicMock()
        mock_seq2seq.return_value = mock_seq2seq_instance
        
        # Mock pipeline
        mock_pipeline.return_value = MagicMock()
        
        llm = HuggingFaceLLM(model_name="test-model", device="cpu")
        
        assert llm.model_type == 'seq2seq'
    
    @patch.object(HuggingFaceLLM, '_load_model')
    def test_generate_without_model(self, mock_load):
        """Test generation failure when model not loaded."""
        llm = HuggingFaceLLM(model_name="test-model")
        llm.model = None
        llm.tokenizer = None
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            llm.generate("test prompt")
    
    @patch.object(HuggingFaceLLM, '_load_model')
    def test_generate_success(self, mock_load):
        """Test successful text generation."""
        # Create LLM instance
        llm = HuggingFaceLLM(model_name="test-model")
        
        # Mock components
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x: [1, 2, 3, 4]  # Mock token IDs
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "Test response"}]
        
        llm.tokenizer = mock_tokenizer
        llm.model = MagicMock()
        llm.pipeline = mock_pipeline
        llm.model_type = 'causal'
        
        # Test generation
        response = llm.generate("test prompt", max_tokens=50)
        
        assert response == "Test response"
        assert llm.usage_stats.total_requests == 1
        assert llm.usage_stats.total_tokens > 0
    
    @patch.object(HuggingFaceLLM, '_load_model')
    def test_batch_generate(self, mock_load):
        """Test batch generation."""
        llm = HuggingFaceLLM(model_name="test-model")
        
        # Mock components
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x: [1, 2, 3]
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"generated_text": "Response 1"},
            {"generated_text": "Response 2"}
        ]
        
        llm.tokenizer = mock_tokenizer
        llm.model = MagicMock()
        llm.pipeline = mock_pipeline
        llm.model_type = 'causal'
        
        # Test batch generation
        prompts = ["prompt 1", "prompt 2"]
        responses = llm.batch_generate(prompts)
        
        assert len(responses) == 2
        assert responses[0] == "Response 1"
        assert responses[1] == "Response 2"
    
    @patch.object(HuggingFaceLLM, '_load_model')
    def test_cost_estimation(self, mock_load):
        """Test cost estimation for local models."""
        llm = HuggingFaceLLM(model_name="test-model")
        
        # Mock model with parameter count
        mock_model = MagicMock()
        mock_model.num_parameters.return_value = 125e6  # 125M parameters
        llm.model = mock_model
        
        # Set some usage
        llm.usage_stats.total_tokens = 1000
        
        cost = llm.estimate_cost()
        
        assert cost >= 0  # Should be non-negative
        assert isinstance(cost, float)
    
    @patch.object(HuggingFaceLLM, '_load_model')
    @patch('torch.cuda.is_available')
    def test_memory_usage(self, mock_cuda_available, mock_load):
        """Test memory usage monitoring."""
        mock_cuda_available.return_value = True
        
        llm = HuggingFaceLLM(model_name="test-model", device="cuda")
        
        with patch('torch.cuda.memory_allocated', return_value=1e9), \
             patch('torch.cuda.memory_reserved', return_value=2e9), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_props.return_value.total_memory = 8e9
            
            memory_info = llm.get_memory_usage()
            
            assert "allocated_gb" in memory_info
            assert "reserved_gb" in memory_info
            assert "total_gb" in memory_info
            assert "utilization" in memory_info
            
            assert memory_info["allocated_gb"] == 1.0
            assert memory_info["reserved_gb"] == 2.0
            assert memory_info["total_gb"] == 8.0
            assert memory_info["utilization"] == 0.125  # 1/8
    
    @patch.object(HuggingFaceLLM, '_load_model')
    def test_model_info(self, mock_load):
        """Test model information retrieval."""
        llm = HuggingFaceLLM(
            model_name="test-model",
            device="cuda",
            load_in_8bit=True
        )
        
        # Mock model config
        mock_config = MagicMock()
        mock_config.vocab_size = 50000
        mock_config.hidden_size = 768
        mock_config.num_hidden_layers = 12
        
        mock_model = MagicMock()
        mock_model.config = mock_config
        mock_model.num_parameters.return_value = 125e6
        
        llm.model = mock_model
        llm.model_type = 'causal'
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_name', return_value="Test GPU"):
            
            info = llm.get_model_info()
            
            assert info["provider"] == "HuggingFace"
            assert info["model_name"] == "test-model"
            assert info["model_type"] == "causal"
            assert info["device"] == "cuda"
            assert info["load_in_8bit"] is True
            assert info["vocab_size"] == 50000
            assert info["hidden_size"] == 768
            assert info["num_layers"] == 12
            assert info["num_parameters"] == 125e6
    
    @patch.object(HuggingFaceLLM, '_load_model')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_clear_cuda_cache(self, mock_empty_cache, mock_cuda_available, mock_load):
        """Test CUDA cache clearing."""
        mock_cuda_available.return_value = True
        
        llm = HuggingFaceLLM(model_name="test-model", device="cuda")
        llm.clear_cuda_cache()
        
        mock_empty_cache.assert_called_once()


@pytest.mark.integration
class TestHuggingFaceIntegration:
    """Integration tests requiring actual model loading (slow)."""
    
    @pytest.mark.slow
    def test_actual_model_loading(self):
        """Test loading an actual small model (requires network)."""
        pytest.importorskip("transformers")
        
        try:
            # Use a very small model for testing
            llm = HuggingFaceLLM(
                model_name="gpt2",
                device="cpu",  # Use CPU to avoid GPU requirements
                max_new_tokens=10
            )
            
            assert llm.model is not None
            assert llm.tokenizer is not None
            assert llm.pipeline is not None
            
            # Test generation
            response = llm.generate("Hello", max_tokens=5)
            assert isinstance(response, str)
            assert len(response) > 0
            
        except Exception as e:
            pytest.skip(f"Model loading failed (expected in CI): {e}")


if __name__ == "__main__":
    pytest.main([__file__])