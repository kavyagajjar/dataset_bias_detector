"""OpenAI and Azure OpenAI providers."""

import os
from typing import Optional

from bias_auditor.core.config import LLMConfig
from bias_auditor.llm.base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None
    
    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
        return self._client
    
    def is_available(self) -> bool:
        """Check if OpenAI is properly configured."""
        if not self.api_key:
            return False
        try:
            from openai import OpenAI
            return True
        except ImportError:
            return False
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate completion using OpenAI."""
        # Check cache
        cached = self._check_cache(prompt, system_prompt)
        if cached:
            return cached
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        
        result = LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            raw_response=response,
        )
        
        # Cache response
        self._store_cache(prompt, system_prompt, result)
        
        return result


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI API provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.api_base = config.api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self._client = None
    
    @property
    def client(self):
        """Lazy load Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
                self._client = AzureOpenAI(
                    api_key=self.api_key,
                    api_version=self.api_version,
                    azure_endpoint=self.api_base,
                )
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
        return self._client
    
    def is_available(self) -> bool:
        """Check if Azure OpenAI is properly configured."""
        if not self.api_key or not self.api_base:
            return False
        try:
            from openai import AzureOpenAI
            return True
        except ImportError:
            return False
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate completion using Azure OpenAI."""
        # Check cache
        cached = self._check_cache(prompt, system_prompt)
        if cached:
            return cached
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,  # This is the deployment name in Azure
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        
        result = LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            raw_response=response,
        )
        
        self._store_cache(prompt, system_prompt, result)
        
        return result
