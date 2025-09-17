"""
Tests for LLM models and interfaces.
"""

import pytest
from unittest.mock import patch, MagicMock
from bmuq.models.base import BaseLLM, LLMUsageStats
from bmuq.models.mock_llm import MockLLM
from bmuq.models.openai_llm import OpenAILLM


class TestLLMUsageStats:
    """Test LLMUsageStats data structure."""
    
    def test_default_stats(self):
        """Test default usage statistics."""
        stats = LLMUsageStats()
        assert stats.total_requests == 0
        assert stats.total_tokens == 0
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.estimated_cost_usd == 0.0


class TestMockLLM:
    """Test MockLLM implementation."""
    
    def test_mock_llm_creation(self):
        """Test MockLLM initialization."""
        llm = MockLLM(temperature=0.8, response_delay=0.1)
        assert llm.model == "mock-llm"
        assert llm.temperature == 0.8
        assert llm.response_delay == 0.1
    
    def test_basic_generation(self):
        """Test basic text generation."""
        llm = MockLLM(response_delay=0.0)  # No delay for testing
        
        response = llm.generate("What is 2+2?")
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Check usage stats updated
        stats = llm.get_usage_stats()
        assert stats.total_requests == 1
        assert stats.total_tokens > 0
    
    def test_pattern_matching(self):
        """Test pattern-based response generation."""
        llm = MockLLM(response_delay=0.0)
        
        # Test extract target pattern
        response = llm.generate("extract target of this step")
        assert "aims to solve" in response.lower() or "isolating" in response.lower()
        
        # Test comparison pattern
        response = llm.generate("compare these two solutions")
        assert response.lower() in ["supports", "contradicts", "not_directly_related"]
    
    def test_custom_responses(self):
        """Test adding custom response patterns."""
        llm = MockLLM(response_delay=0.0)
        
        # Add custom response
        llm.add_custom_response("test_pattern", "custom test response")
        
        # The mock LLM doesn't directly use custom patterns in generate,
        # but we can test that they're stored
        assert "test_pattern" in llm.step_responses
        assert llm.step_responses["test_pattern"] == "custom test response"
    
    def test_batch_generation(self):
        """Test batch generation."""
        llm = MockLLM(response_delay=0.0)
        
        prompts = ["What is 1+1?", "What is 2+2?", "What is 3+3?"]
        responses = llm.batch_generate(prompts)
        
        assert len(responses) == 3
        assert all(isinstance(r, str) for r in responses)
        
        # Check usage stats
        stats = llm.get_usage_stats()
        assert stats.total_requests == 3
    
    def test_cost_estimation(self):
        """Test cost estimation (should be 0 for mock)."""
        llm = MockLLM()
        llm.generate("test prompt")
        
        cost = llm.estimate_cost()
        assert cost == 0.0
    
    def test_model_info(self):
        """Test model information retrieval."""
        llm = MockLLM(response_delay=0.2)
        
        info = llm.get_model_info()
        assert info["provider"] == "Mock"
        assert info["model"] == "mock-llm"
        assert info["response_delay"] == 0.2
        assert "available_patterns" in info


class TestOpenAILLM:
    """Test OpenAILLM implementation."""
    
    def test_openai_llm_creation_no_key(self):
        """Test OpenAI LLM creation without API key should raise error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key must be provided"):
                OpenAILLM()
    
    def test_openai_llm_creation_with_key(self):
        """Test OpenAI LLM creation with API key."""
        llm = OpenAILLM(api_key="test-key", model="gpt-3.5-turbo")
        assert llm.api_key == "test-key"
        assert llm.model == "gpt-3.5-turbo"
    
    @patch('openai.OpenAI')
    def test_generation_success(self, mock_openai):
        """Test successful generation."""
        # Mock the OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.total_tokens = 50
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 30
        
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OpenAILLM(api_key="test-key")
        response = llm.generate("Test prompt")
        
        assert response == "Test response"
        
        # Check usage stats
        stats = llm.get_usage_stats()
        assert stats.total_requests == 1
        assert stats.total_tokens == 50
        assert stats.input_tokens == 20
        assert stats.output_tokens == 30
    
    @patch('openai.OpenAI')
    def test_generation_api_error(self, mock_openai):
        """Test handling of API errors."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Simulate API error
        import openai
        mock_client.chat.completions.create.side_effect = openai.APIError("Test error")
        
        llm = OpenAILLM(api_key="test-key")
        response = llm.generate("Test prompt")
        
        # Should return error message instead of raising exception
        assert "Error:" in response
    
    def test_cost_estimation(self):
        """Test cost estimation for different models."""
        # Test with GPT-4
        llm = OpenAILLM(api_key="test-key", model="gpt-4")
        llm.usage_stats.input_tokens = 1000
        llm.usage_stats.output_tokens = 500
        
        cost = llm.estimate_cost()
        assert cost > 0  # Should have positive cost
        
        # Test with GPT-3.5-turbo (cheaper)
        llm_cheap = OpenAILLM(api_key="test-key", model="gpt-3.5-turbo")
        llm_cheap.usage_stats.input_tokens = 1000
        llm_cheap.usage_stats.output_tokens = 500
        
        cost_cheap = llm_cheap.estimate_cost()
        assert cost_cheap > 0
        assert cost_cheap < cost  # Should be cheaper than GPT-4
    
    @patch('openai.OpenAI')
    def test_batch_generation(self, mock_openai):
        """Test batch generation."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Batch response"
        mock_response.usage.total_tokens = 30
        
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OpenAILLM(api_key="test-key")
        prompts = ["Prompt 1", "Prompt 2"]
        responses = llm.batch_generate(prompts)
        
        assert len(responses) == 2
        assert all(r == "Batch response" for r in responses)
        
        # Should have made 2 API calls
        assert mock_client.chat.completions.create.call_count == 2
    
    def test_model_info(self):
        """Test model information retrieval."""
        llm = OpenAILLM(api_key="test-key", model="gpt-4-turbo")
        
        info = llm.get_model_info()
        assert info["provider"] == "OpenAI"
        assert info["model"] == "gpt-4-turbo"
        assert "api_version" in info


class TestBaseLLM:
    """Test BaseLLM abstract interface."""
    
    def test_base_llm_abstract(self):
        """Test that BaseLLM cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLM("test-model")
    
    def test_usage_stats_reset(self):
        """Test usage statistics reset."""
        llm = MockLLM()
        
        # Generate some usage
        llm.generate("test")
        assert llm.get_usage_stats().total_requests == 1
        
        # Reset stats
        llm.reset_usage_stats()
        stats = llm.get_usage_stats()
        assert stats.total_requests == 0
        assert stats.total_tokens == 0


if __name__ == "__main__":
    pytest.main([__file__])