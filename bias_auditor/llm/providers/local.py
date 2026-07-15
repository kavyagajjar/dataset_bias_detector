"""Local LLM provider using Ollama or LiteLLM."""

import os
from typing import Optional

from bias_auditor.core.config import LLMConfig
from bias_auditor.llm.base import BaseLLMProvider, LLMResponse


class LocalProvider(BaseLLMProvider):
    """Local LLM provider using Ollama or compatible API."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434"
        )
        self._client = None
        
        # Default to a capable local model
        if not self.model or self.model == "gpt-4o":
            self.model = "llama3.1:8b"
    
    @property
    def client(self):
        """Lazy load local client via LiteLLM."""
        if self._client is None:
            try:
                import litellm
                self._client = litellm
                # Configure for Ollama
                litellm.api_base = self.api_base
            except ImportError:
                raise ImportError(
                    "LiteLLM package not installed. Install with: pip install litellm"
                )
        return self._client
    
    def is_available(self) -> bool:
        """Check if local LLM is available."""
        try:
            import litellm
            import requests
            
            # Check if Ollama is running
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate completion using local LLM."""
        # Check cache
        cached = self._check_cache(prompt, system_prompt)
        if cached:
            return cached
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Use Ollama format for model name
        model_name = f"ollama/{self.model}" if not self.model.startswith("ollama/") else self.model
        
        response = self.client.completion(
            model=model_name,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            api_base=self.api_base,
        )
        
        result = LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage={
                "input_tokens": response.usage.get("prompt_tokens", 0),
                "output_tokens": response.usage.get("completion_tokens", 0),
            },
            raw_response=response,
        )
        
        self._store_cache(prompt, system_prompt, result)
        
        return result


class OllamaProvider(BaseLLMProvider):
    """Direct Ollama provider without LiteLLM dependency."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434"
        )
        
        if not self.model or self.model == "gpt-4o":
            self.model = "llama3.1:8b"
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            import requests
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate completion using Ollama API directly."""
        import requests
        
        # Check cache
        cached = self._check_cache(prompt, system_prompt)
        if cached:
            return cached
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = requests.post(
            f"{self.api_base}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_predict": max_tokens or self.config.max_tokens,
                },
            },
            timeout=120,
        )
        
        response.raise_for_status()
        data = response.json()
        
        result = LLMResponse(
            content=data.get("response", ""),
            model=self.model,
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
            raw_response=data,
        )
        
        self._store_cache(prompt, system_prompt, result)
        
        return result
