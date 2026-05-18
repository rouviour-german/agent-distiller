"""Base Provider interface for agent-distiller."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Run a chat completion."""
        ...

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text."""
        ...

    @abstractmethod
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for a specific call."""
        ...

    @abstractmethod
    def get_training_cost(self, total_tokens: int, epochs: int) -> float:
        """Estimate fine-tuning cost for this provider."""
        ...
