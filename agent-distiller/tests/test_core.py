"""Comprehensive tests for agent-distiller."""

import json
import tempfile
from pathlib import Path

import pytest

from agent_distiller.models import (
    ComparisonResult,
    DistillationReport,
    PipelineTarget,
    Trace,
)
from agent_distiller.collectors import Collector
from agent_distiller.trainers import Trainer


# ─────────────────────────────────────────────
# Mock Pipeline
# ─────────────────────────────────────────────


class MockPipeline(PipelineTarget):
    """Deterministic mock pipeline for testing."""

    async def run(self, task_input: str) -> str:
        return (
            f"Research report on: {task_input}\n\n"
            f"## Key Findings\n"
            f"After thorough analysis by our research team, analyst division, "
            f"and writing department, we found that {task_input} is a rapidly "
            f"evolving area with significant implications for the industry. "
            f"The market is projected to grow substantially over the next "
            f"several years, driven by increased adoption and technological "
            f"advances. Key players include major tech companies and emerging "
            f"startups. Regulatory frameworks are still catching up to the "
            f"pace of innovation. We recommend monitoring these developments "
            f"closely and considering strategic investments."
        )

    def get_system_prompt(self) -> str:
        return "You are an expert research analyst."


class FailingPipeline(PipelineTarget):
    """Pipeline that always fails."""

    async def run(self, task_input: str) -> str:
        raise RuntimeError("Pipeline crashed!")


class ShortOutputPipeline(PipelineTarget):
    """Pipeline that returns output too short to be useful."""

    async def run(self, task_input: str) -> str:
        return "Short."


# ─────────────────────────────────────────────
# Trace Tests
# ─────────────────────────────────────────────


class TestTrace:
    def test_to_dict(self):
        trace = Trace(
            input="Compare LangGraph vs CrewAI",
            output="LangGraph is graph-based...",
            quality_score=8.5,
            cost_usd=0.13,
            latency_seconds=16.1,
            tokens=12340,
        )
        d = trace.to_dict()
        assert d["input"] == "Compare LangGraph vs CrewAI"
        assert d["quality_score"] == 8.5
        assert d["cost_usd"] == 0.13
        assert d["tokens"] == 12340

    def test_from_dict(self):
        data = {
            "input": "test",
            "output": "result",
            "quality_score": 9.0,
            "cost_usd": 0.05,
            "latency_s": 3.2,
            "tokens": 500,
        }
        trace = Trace.from_dict(data)
        assert trace.input == "test"
        assert trace.output == "result"
        assert trace.quality_score == 9.0
        assert trace.latency_seconds == 3.2

    def test_to_training_example_with_system_prompt(self):
        trace = Trace(input="Q", output="A")
        example = trace.to_training_example("You are an expert.")
        assert len(example["messages"]) == 3
        assert example["messages"][0]["role"] == "system"
        assert example["messages"][0]["content"] == "You are an expert."
        assert example["messages"][1]["role"] == "user"
        assert example["messages"][1]["content"] == "Q"
        assert example["messages"][2]["role"] == "assistant"
        assert example["messages"][2]["content"] == "A"

    def test_to_training_example_without_system_prompt(self):
        trace = Trace(input="Q", output="A")
        example = trace.to_training_example("")
        assert len(example["messages"]) == 2
        assert example["messages"][0]["role"] == "user"

    def test_roundtrip_dict(self):
        original = Trace(
            input="hello", output="world",
            quality_score=7.5, cost_usd=0.01,
            latency_seconds=1.0, tokens=100,
        )
        restored = Trace.from_dict(original.to_dict())
        assert restored.input == original.input
        assert restored.output == original.output
        assert restored.quality_score == original.quality_score

    def test_to_dict_json_serializable(self):
        trace = Trace(input="a", output="b")
        json.dumps(trace.to_dict())  # Must not raise


# ─────────────────────────────────────────────
# DistillationReport Tests
# ─────────────────────────────────────────────


class TestDistillationReport:
    def _make_comparison(
        self, p_quality=8.0, d_quality=7.5,
        p_cost=0.13, d_cost=0.008,
        p_latency=16.0, d_latency=2.0,
    ) -> ComparisonResult:
        return ComparisonResult(
            input="test",
            pipeline_output="pipeline output",
            distilled_output="distilled output",
            pipeline_quality=p_quality,
            distilled_quality=d_quality,
            pipeline_cost=p_cost,
            distilled_cost=d_cost,
            pipeline_latency=p_latency,
            distilled_latency=d_latency,
        )

    def test_quality_retention(self):
        report = DistillationReport(
            comparisons=[self._make_comparison(8.0, 7.4)]
        )
        assert abs(report.quality_retention - 0.925) < 0.01

    def test_cost_reduction(self):
        report = DistillationReport(
            comparisons=[self._make_comparison(p_cost=0.13, d_cost=0.008)]
        )
        assert report.cost_reduction > 0.9  # >90% reduction

    def test_latency_reduction(self):
        report = DistillationReport(
            comparisons=[self._make_comparison(p_latency=16.0, d_latency=2.0)]
        )
        assert report.latency_reduction > 0.8  # >80% reduction

    def test_verdict_excellent(self):
        report = DistillationReport(
            comparisons=[self._make_comparison(8.0, 7.8)]
        )
        assert report.verdict == "excellent"

    def test_verdict_good(self):
        report = DistillationReport(
            comparisons=[self._make_comparison(8.0, 7.3)]
        )
        assert report.verdict == "good"

    def test_verdict_acceptable(self):
        report = DistillationReport(
            comparisons=[self._make_comparison(8.0, 6.5)]
        )
        assert report.verdict == "acceptable"

    def test_verdict_insufficient(self):
        report = DistillationReport(
            comparisons=[self._make_comparison(8.0, 5.0)]
        )
        assert report.verdict == "insufficient"

    def test_multiple_comparisons(self):
        report = DistillationReport(
            comparisons=[
                self._make_comparison(8.0, 7.5, 0.13, 0.01),
                self._make_comparison(9.0, 8.0, 0.15, 0.008),
                self._make_comparison(7.0, 6.5, 0.10, 0.012),
            ]
        )
        assert report.num_tasks == 3
        assert abs(report.avg_pipeline_quality - 8.0) < 0.01
        assert abs(report.avg_distilled_quality - 7.333) < 0.01

    def test_to_dict_serializable(self):
        report = DistillationReport(
            pipeline_name="TestPipeline",
            distilled_model="ft:gpt-4o-mini:test",
            comparisons=[self._make_comparison()],
        )
        d = report.to_dict()
        json.dumps(d)  # Must not raise
        assert d["pipeline"] == "TestPipeline"
        assert d["verdict"] in ["excellent", "good", "acceptable", "insufficient"]
        assert "quality_retention" in d["metrics"]
        assert "cost_reduction" in d["metrics"]

    def test_empty_report(self):
        report = DistillationReport()
        assert report.num_tasks == 0
        assert report.quality_retention == 0.0
        assert report.cost_reduction == 0.0
        assert report.verdict == "insufficient"


# ─────────────────────────────────────────────
# Collector Tests
# ─────────────────────────────────────────────


class TestCollector:
    @pytest.mark.asyncio
    async def test_collect_from_mock_pipeline(self):
        collector = Collector(
            pipeline=MockPipeline(),
            min_quality=0,  # Accept everything
        )
        traces = await collector.collect(["Test topic 1", "Test topic 2"])
        assert len(traces) == 2
        assert all(t.output for t in traces)
        # Latency might be 0 for near-instant mock calls
        assert all(t.latency_seconds >= 0 for t in traces)

    @pytest.mark.asyncio
    async def test_collect_failing_pipeline(self):
        collector = Collector(
            pipeline=FailingPipeline(),
            min_quality=0,
        )
        traces = await collector.collect(["test"])
        assert len(traces) == 0  # Failed runs should be skipped

    @pytest.mark.asyncio
    async def test_collect_short_output_filtered(self):
        collector = Collector(
            pipeline=ShortOutputPipeline(),
            min_quality=0,
        )
        traces = await collector.collect(["test"])
        assert len(traces) == 0  # Too short, should be filtered

    def test_save_and_load_traces(self, tmp_path):
        traces = [
            Trace(input="q1", output="a1", quality_score=8.0),
            Trace(input="q2", output="a2", quality_score=9.0),
        ]
        path = str(tmp_path / "traces.jsonl")
        Collector.save_traces(traces, path)

        loaded = Collector.load_traces(path)
        assert len(loaded) == 2
        assert loaded[0].input == "q1"
        assert loaded[1].quality_score == 9.0

    def test_save_traces_creates_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "traces.jsonl")
        Collector.save_traces([Trace(input="x", output="y")], path)
        assert Path(path).exists()

    @pytest.mark.asyncio
    async def test_collect_from_file(self, tmp_path):
        tasks_file = tmp_path / "tasks.jsonl"
        tasks_file.write_text(
            '{"input": "Topic A"}\n'
            '{"input": "Topic B"}\n'
            '{"input": "Topic C"}\n'
        )

        collector = Collector(pipeline=MockPipeline(), min_quality=0)
        traces = await collector.collect_from_file(str(tasks_file), max_tasks=2)
        assert len(traces) == 2


# ─────────────────────────────────────────────
# Trainer Tests
# ─────────────────────────────────────────────


class TestTrainer:
    def test_prepare_training_data(self, tmp_path):
        traces = [
            Trace(input=f"q{i}", output=f"a{i}") for i in range(10)
        ]
        # Save traces first
        traces_path = str(tmp_path / "traces.jsonl")
        Collector.save_traces(traces, traces_path)

        trainer = Trainer(validation_split=0.2)
        train_path, val_path = trainer.prepare_training_data(
            traces,
            system_prompt="You are a researcher.",
            output_path=str(tmp_path / "train.jsonl"),
            validation_path=str(tmp_path / "val.jsonl"),
        )

        # Check training file
        with open(train_path) as f:
            train_lines = [json.loads(l) for l in f if l.strip()]
        assert len(train_lines) == 8  # 80% of 10

        # Check format
        example = train_lines[0]
        assert "messages" in example
        assert example["messages"][0]["role"] == "system"
        assert example["messages"][1]["role"] == "user"
        assert example["messages"][2]["role"] == "assistant"

        # Check validation file
        with open(val_path) as f:
            val_lines = [l for l in f if l.strip()]
        assert len(val_lines) == 2  # 20% of 10

    def test_estimate_cost(self, tmp_path):
        traces = [Trace(input="q", output="a" * 400, tokens=100) for _ in range(50)]
        path = str(tmp_path / "traces.jsonl")
        Collector.save_traces(traces, path)

        trainer = Trainer(base_model="gpt-4o-mini", epochs=3)
        est = trainer.estimate_cost(path)

        assert est["traces"] == 50
        assert est["total_tokens"] == 5000
        assert est["epochs"] == 3
        assert est["estimated_cost_usd"] > 0
        assert est["base_model"] == "gpt-4o-mini"

    def test_estimate_cost_different_models(self, tmp_path):
        traces = [Trace(input="q", output="a", tokens=1000) for _ in range(100)]
        path = str(tmp_path / "traces.jsonl")
        Collector.save_traces(traces, path)

        mini = Trainer(base_model="gpt-4o-mini").estimate_cost(path, 3)
        full = Trainer(base_model="gpt-4o").estimate_cost(path, 3)
        assert full["estimated_cost_usd"] > mini["estimated_cost_usd"]


# ─────────────────────────────────────────────
# Pipeline Target Tests
# ─────────────────────────────────────────────


class TestPipelineTarget:
    def test_default_system_prompt(self):
        class Bare(PipelineTarget):
            async def run(self, task_input: str) -> str:
                return "output"
        assert Bare().get_system_prompt() == "You are a helpful AI assistant."

    def test_custom_system_prompt(self):
        assert MockPipeline().get_system_prompt() == "You are an expert research analyst."

    @pytest.mark.asyncio
    async def test_mock_pipeline_output(self):
        pipeline = MockPipeline()
        output = await pipeline.run("AI agents")
        assert "AI agents" in output
        assert len(output) > 100
