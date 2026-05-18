"""OpenAI Provider implementation."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import BaseProvider

logger = logging.getLogger("agent-distiller")


class OpenAIProvider(BaseProvider):
    """OpenAI implementation using AsyncOpenAI."""

    def __init__(self, model_id: str = "gpt-4o-mini"):
        super().__init__(model_id)
        self._client: Optional[Any] = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI()
        return self._client

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            resp = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text.
        
        Using tiktoken for accurate estimation.
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4o")
            return len(encoding.encode(text))
        except (ImportError, KeyError):
            # Fallback for tiktoken missing or new models
            return len(text) // 4

    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate inference cost."""
        pricing = {
            "gpt-4o-mini": {"in": 0.00015, "out": 0.0006},  # per 1k tokens
            "gpt-4o": {"in": 0.0025, "out": 0.010},
            "gpt-4.5-preview": {"in": 0.075, "out": 0.15},
        }
        
        # Default gpt-4o-mini pricing if not found
        p = pricing.get(self.model_id, pricing["gpt-4o-mini"])
        cost = (input_tokens / 1000) * p["in"] + (output_tokens / 1000) * p["out"]
        return cost

    def get_training_cost(self, total_tokens: int, epochs: int) -> float:
        """Estimate fine-tuning cost."""
        pricing = {
            "gpt-4o-mini": 0.0003,      # per 1K tokens
            "gpt-4o": 0.003,
            "gpt-4.1-mini": 0.0003,
        }
        rate = pricing.get(self.model_id, 0.0003)
        return (total_tokens / 1000) * rate * epochs
