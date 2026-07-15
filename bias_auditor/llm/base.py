"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import os

from bias_auditor.core.config import LLMConfig, LLMProvider


@dataclass
class LLMResponse:
    """Response from an LLM."""
    
    content: str
    model: str
    usage: dict[str, int]
    raw_response: Optional[Any] = None
    
    @property
    def input_tokens(self) -> int:
        return self.usage.get("input_tokens", 0)
    
    @property
    def output_tokens(self) -> int:
        return self.usage.get("output_tokens", 0)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.model
        self._cache: dict[str, LLMResponse] = {}
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.
        
        Parameters
        ----------
        prompt : str
            The user prompt.
        system_prompt : str, optional
            System prompt to set context.
        temperature : float, optional
            Override default temperature.
        max_tokens : int, optional
            Override default max tokens.
        
        Returns
        -------
        LLMResponse
            The LLM response.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        pass
    
    def _get_cache_key(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate cache key for a prompt."""
        return f"{self.model}:{hash(prompt)}:{hash(system_prompt or '')}"
    
    def _check_cache(self, prompt: str, system_prompt: Optional[str]) -> Optional[LLMResponse]:
        """Check if response is cached."""
        if not self.config.cache_responses:
            return None
        
        key = self._get_cache_key(prompt, system_prompt)
        return self._cache.get(key)
    
    def _store_cache(self, prompt: str, system_prompt: Optional[str], response: LLMResponse):
        """Store response in cache."""
        if self.config.cache_responses:
            key = self._get_cache_key(prompt, system_prompt)
            self._cache[key] = response


def get_llm_provider(config: LLMConfig) -> Optional[BaseLLMProvider]:
    """
    Factory function to get the appropriate LLM provider.
    
    Parameters
    ----------
    config : LLMConfig
        LLM configuration.
    
    Returns
    -------
    BaseLLMProvider or None
        The configured provider, or None if provider is NONE.
    """
    if config.provider == LLMProvider.NONE:
        return None
    
    if config.provider == LLMProvider.OPENAI:
        from bias_auditor.llm.providers.openai import OpenAIProvider
        return OpenAIProvider(config)
    
    if config.provider == LLMProvider.ANTHROPIC:
        from bias_auditor.llm.providers.anthropic import AnthropicProvider
        return AnthropicProvider(config)
    
    if config.provider == LLMProvider.AZURE:
        from bias_auditor.llm.providers.openai import AzureOpenAIProvider
        return AzureOpenAIProvider(config)
    
    if config.provider == LLMProvider.LOCAL:
        from bias_auditor.llm.providers.local import LocalProvider
        return LocalProvider(config)
    
    raise ValueError(f"Unknown LLM provider: {config.provider}")
