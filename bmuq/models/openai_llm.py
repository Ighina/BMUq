"""
OpenAI API implementation for uncertainty quantification.
"""

import os
import time
from typing import List, Optional, Dict, Any
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseLLM, LLMUsageStats


class OpenAILLM(BaseLLM):
    """OpenAI API wrapper for LLM interactions with error handling and retry logic."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview",
                 temperature: float = 0.7, max_retries: int = 3):
        """
        Initialize OpenAI LLM.

        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: Model to use (gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo, etc.)
            temperature: Generation temperature (0.0-2.0)
            max_retries: Maximum number of retries for failed requests
        """
        super().__init__(model, temperature, max_retries)
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

        self.client = openai.OpenAI(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    def generate(self, prompt: str, max_tokens: int = 150, temperature: Optional[float] = None, structured_output = None) -> str:
        """
        Generate text using OpenAI API with retry logic.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature

        Returns:
            Generated text response
        """
        try:
            temp = temperature if temperature is not None else self.temperature

            if structured_output is None:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides clear, step-by-step mathematical reasoning."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=temp,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
            else:
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": "You are a helpful assistant that provides clear, step-by-step mathematical reasoning."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    text_format=structured_output
                )


            # Update usage tracking
            if hasattr(response, 'usage') and response.usage:
                self.usage_stats.total_tokens += response.usage.total_tokens
                if hasattr(response.usage, 'prompt_tokens'):
                    self.usage_stats.input_tokens += response.usage.prompt_tokens
                if hasattr(response.usage, 'completion_tokens'):
                    self.usage_stats.output_tokens += response.usage.completion_tokens
            
            self.usage_stats.total_requests += 1

            # Extract and return the generated text
            if structured_output is not None:
                return response.output_parsed
            content = response.choices[0].message.content.strip()
            return content

        except openai.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            print("Waiting 60 seconds before retry...")
            time.sleep(60)
            raise

        except openai.APITimeoutError as e:
            print(f"API timeout: {e}")
            raise

        except openai.APIError as e:
            print(f"OpenAI API error: {e}")
            return "Error: Unable to generate response due to API error."

        except Exception as e:
            print(f"Unexpected error in generation: {e}")
            return "Error: Unexpected error occurred during generation."

    def batch_generate(self, prompts: List[str], max_tokens: int = 150, 
                      temperature: Optional[float] = None) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Note: OpenAI doesn't have native batch API for chat completions,
        so this implements sequential generation with error handling.
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, max_tokens, temperature)
                results.append(result)
            except Exception as e:
                print(f"Error generating response for prompt: {e}")
                results.append("Error: Failed to generate response.")
        return results

    def estimate_cost(self) -> float:
        """Estimate cost in USD based on token usage."""
        # Pricing as of 2024 (check OpenAI pricing page for current rates)
        pricing = {
            "gpt-4": {"input": 0.03/1000, "output": 0.06/1000},
            "gpt-4-turbo": {"input": 0.01/1000, "output": 0.03/1000},
            "gpt-4-turbo-preview": {"input": 0.01/1000, "output": 0.03/1000},
            "gpt-3.5-turbo": {"input": 0.0005/1000, "output": 0.0015/1000},
            "gpt-3.5-turbo-16k": {"input": 0.003/1000, "output": 0.004/1000},
        }

        # Find matching pricing model
        model_key = self.model
        if model_key not in pricing:
            # Try to match partial model names
            for key in pricing.keys():
                if key in self.model:
                    model_key = key
                    break
            else:
                # Use average if model not found
                avg_input = sum(p["input"] for p in pricing.values()) / len(pricing)
                avg_output = sum(p["output"] for p in pricing.values()) / len(pricing)
                model_pricing = {"input": avg_input, "output": avg_output}
        else:
            model_pricing = pricing[model_key]

        input_cost = self.usage_stats.input_tokens * model_pricing["input"]
        output_cost = self.usage_stats.output_tokens * model_pricing["output"]
        
        total_cost = input_cost + output_cost
        self.usage_stats.estimated_cost_usd = total_cost
        return total_cost

    def generate_with_log_probs(self, prompt: str, max_tokens: int = 150,
                                 temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate text and return log probabilities for each token.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature

        Returns:
            Dictionary containing:
                - 'text': Generated text
                - 'log_probs': List of log probabilities for each generated token
                - 'tokens': List of generated tokens (as strings)
                - 'top_log_probs': List of dicts with top-k tokens and their log probs
        """
        try:
            temp = temperature if temperature is not None else self.temperature

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides clear, step-by-step mathematical reasoning."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens,
                temperature=temp,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logprobs=True,  # Request log probabilities
                top_logprobs=5   # Get top 5 alternative tokens at each position
            )

            # Update usage tracking
            if hasattr(response, 'usage') and response.usage:
                self.usage_stats.total_tokens += response.usage.total_tokens
                if hasattr(response.usage, 'prompt_tokens'):
                    self.usage_stats.input_tokens += response.usage.prompt_tokens
                if hasattr(response.usage, 'completion_tokens'):
                    self.usage_stats.output_tokens += response.usage.completion_tokens

            self.usage_stats.total_requests += 1

            # Extract generated text
            content = response.choices[0].message.content.strip()

            # Extract log probabilities
            log_probs = []
            tokens = []
            top_log_probs = []

            if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                logprobs_data = response.choices[0].logprobs
                if hasattr(logprobs_data, 'content') and logprobs_data.content:
                    for token_data in logprobs_data.content:
                        # Get the token that was generated
                        tokens.append(token_data.token)
                        log_probs.append(token_data.logprob)

                        # Get top alternative tokens
                        if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                            top_k_dict = {
                                alt.token: alt.logprob
                                for alt in token_data.top_logprobs
                            }
                            top_log_probs.append(top_k_dict)
                        else:
                            top_log_probs.append({})

            return {
                'text': content,
                'log_probs': log_probs,
                'tokens': tokens,
                'top_log_probs': top_log_probs
            }

        except openai.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            print("Waiting 60 seconds before retry...")
            time.sleep(60)
            raise

        except openai.APITimeoutError as e:
            print(f"API timeout: {e}")
            raise

        except openai.APIError as e:
            print(f"OpenAI API error: {e}")
            # Fallback to regular generation
            text = self.generate(prompt, max_tokens, temperature)
            return {
                'text': text,
                'log_probs': [],
                'tokens': [],
                'top_log_probs': []
            }

        except Exception as e:
            print(f"Unexpected error in generation with log probs: {e}")
            # Fallback to regular generation
            text = self.generate(prompt, max_tokens, temperature)
            return {
                'text': text,
                'log_probs': [],
                'tokens': [],
                'top_log_probs': []
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI-specific model information."""
        info = super().get_model_info()
        info.update({
            "provider": "OpenAI",
            "api_version": openai.version.VERSION if hasattr(openai, 'version') else "unknown",
        })
        return info