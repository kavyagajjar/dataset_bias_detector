"""Anthropic Claude provider."""

import os
from typing import Optional

from bias_auditor.core.config import LLMConfig
from bias_auditor.llm.base import BaseLLMProvider, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

        # Default to Claude Opus 4.8 if not specified
        if not self.model or self.model == "gpt-4o":
            self.model = "claude-opus-4-8"

    @property
    def client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install anthropic"
                ) from None
        return self._client

    def is_available(self) -> bool:
        """Check if Anthropic is properly configured."""
        if not self.api_key:
            return False
        try:
            from anthropic import Anthropic  # noqa: F401
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
        """Generate completion using Anthropic Claude."""
        # Check cache
        cached = self._check_cache(prompt, system_prompt)
        if cached:
            return cached

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        # Claude Opus 4.7+ models reject sampling parameters (temperature/top_p)
        # with a 400 error, so temperature is intentionally not forwarded here.

        response = self.client.messages.create(**kwargs)

        result = LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw_response=response,
        )

        self._store_cache(prompt, system_prompt, result)

        return result
