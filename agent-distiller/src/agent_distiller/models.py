"""Core data models for agent-distiller."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Pipeline Target
# ---------------------------------------------------------------------------

class PipelineTarget(ABC):
    """Base class — wrap your multi-agent pipeline for distillation."""

    def setup(self) -> None:
        """Initialize the pipeline. Called once."""

    @abstractmethod
    async def run(self, task_input: str) -> str:
        """Run the full multi-agent pipeline, return the final output."""
        ...

    def get_system_prompt(self) -> str:
        """System prompt for the distilled model.

        This should describe what the distilled model should do —
        essentially the combined behavior of all agents.
        """
        return "You are a helpful AI assistant."

    def teardown(self) -> None:
        """Cleanup. Called once."""


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    """A single input→output pair from a successful pipeline run."""

    input: str
    output: str
    quality_score: float = 0.0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    tokens: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "input": self.input,
            "output": self.output,
            "quality_score": round(self.quality_score, 2),
            "cost_usd": round(self.cost_usd, 4),
            "latency_s": round(self.latency_seconds, 2),
            "tokens": self.tokens,
        }

    def to_training_example(self, system_prompt: str = "") -> dict:
        """Convert to OpenAI fine-tuning format."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.input})
        messages.append({"role": "assistant", "content": self.output})
        return {"messages": messages}

    @classmethod
    def from_dict(cls, data: dict) -> Trace:
        return cls(
            input=data["input"],
            output=data["output"],
            quality_score=data.get("quality_score", 0),
            cost_usd=data.get("cost_usd", 0),
            latency_seconds=data.get("latency_s", data.get("latency_seconds", 0)),
            tokens=data.get("tokens", 0),
        )


# ---------------------------------------------------------------------------
# Distillation Report
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Side-by-side result for a single test task."""

    input: str
    pipeline_output: str
    distilled_output: str
    pipeline_quality: float = 0.0
    distilled_quality: float = 0.0
    pipeline_cost: float = 0.0
    distilled_cost: float = 0.0
    pipeline_latency: float = 0.0
    distilled_latency: float = 0.0
    pipeline_tokens: int = 0
    distilled_tokens: int = 0


@dataclass
class DistillationReport:
    """Full comparison report: pipeline vs. distilled model."""

    comparisons: list[ComparisonResult] = field(default_factory=list)
    pipeline_name: str = ""
    distilled_model: str = ""
    system_prompt: str = ""

    @property
    def num_tasks(self) -> int:
        return len(self.comparisons)

    @property
    def avg_pipeline_quality(self) -> float:
        if not self.comparisons:
            return 0.0
        return sum(c.pipeline_quality for c in self.comparisons) / len(self.comparisons)

    @property
    def avg_distilled_quality(self) -> float:
        if not self.comparisons:
            return 0.0
        return sum(c.distilled_quality for c in self.comparisons) / len(self.comparisons)

    @property
    def quality_retention(self) -> float:
        if self.avg_pipeline_quality == 0:
            return 0.0
        return self.avg_distilled_quality / self.avg_pipeline_quality

    @property
    def avg_pipeline_cost(self) -> float:
        if not self.comparisons:
            return 0.0
        return sum(c.pipeline_cost for c in self.comparisons) / len(self.comparisons)

    @property
    def avg_distilled_cost(self) -> float:
        if not self.comparisons:
            return 0.0
        return sum(c.distilled_cost for c in self.comparisons) / len(self.comparisons)

    @property
    def cost_reduction(self) -> float:
        if self.avg_pipeline_cost == 0:
            return 0.0
        return 1.0 - (self.avg_distilled_cost / self.avg_pipeline_cost)

    @property
    def avg_pipeline_latency(self) -> float:
        if not self.comparisons:
            return 0.0
        return sum(c.pipeline_latency for c in self.comparisons) / len(self.comparisons)

    @property
    def avg_distilled_latency(self) -> float:
        if not self.comparisons:
            return 0.0
        return sum(c.distilled_latency for c in self.comparisons) / len(self.comparisons)

    @property
    def latency_reduction(self) -> float:
        if self.avg_pipeline_latency == 0:
            return 0.0
        return 1.0 - (self.avg_distilled_latency / self.avg_pipeline_latency)

    @property
    def verdict(self) -> str:
        r = self.quality_retention
        if r >= 0.95:
            return "excellent"
        if r >= 0.90:
            return "good"
        if r >= 0.80:
            return "acceptable"
        return "insufficient"

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline_name,
            "distilled_model": self.distilled_model,
            "num_tasks": self.num_tasks,
            "verdict": self.verdict,
            "metrics": {
                "quality_pipeline": round(self.avg_pipeline_quality, 2),
                "quality_distilled": round(self.avg_distilled_quality, 2),
                "quality_retention": round(self.quality_retention, 3),
                "cost_pipeline": round(self.avg_pipeline_cost, 4),
                "cost_distilled": round(self.avg_distilled_cost, 4),
                "cost_reduction": round(self.cost_reduction, 3),
                "latency_pipeline": round(self.avg_pipeline_latency, 2),
                "latency_distilled": round(self.avg_distilled_latency, 2),
                "latency_reduction": round(self.latency_reduction, 3),
            },
        }
