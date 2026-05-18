"""Collector — runs a pipeline on tasks and captures successful traces."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from ..models import PipelineTarget, Trace
from ..providers.openai import OpenAIProvider

logger = logging.getLogger("agent-distiller")


class Collector:
    """Run a multi-agent pipeline on tasks and collect successful traces.

    Traces are filtered by quality (LLM judge) and cost.
    Only successful, high-quality runs are saved for distillation.
    """

    def __init__(
        self,
        pipeline: PipelineTarget,
        min_quality: float = 7.0,
        max_cost_per_run: float = 1.0,
        judge_model: str = "gpt-4o-mini",
        parallel: int = 1,
    ):
        self.pipeline = pipeline
        self.min_quality = min_quality
        self.max_cost_per_run = max_cost_per_run
        self.parallel = parallel
        self.judge_provider = OpenAIProvider(judge_model)

    async def collect_from_file(
        self, tasks_path: str, max_tasks: Optional[int] = None
    ) -> list[Trace]:
        """Load tasks from a JSONL file and collect traces."""
        tasks = []
        with open(tasks_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        tasks.append(data.get("input", data.get("task", "")))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line: {line[:50]}...")

        if max_tasks:
            tasks = tasks[:max_tasks]

        return await self.collect(tasks)

    async def collect(self, tasks: list[str]) -> list[Trace]:
        """Run pipeline on tasks and return high-quality traces."""
        self.pipeline.setup()
        traces: list[Trace] = []

        try:
            # Semaphores for concurrency management
            semaphore = asyncio.Semaphore(self.parallel)

            async def _sem_process_task(t: str):
                async with semaphore:
                    return await self._process_task(t)

            batch_tasks = [_sem_process_task(t) for t in tasks]
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Trace):
                    traces.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Task failed: {result}")
                elif result is None:
                    # Filtered out (low quality, error, etc.)
                    pass

        finally:
            self.pipeline.teardown()

        logger.info(
            f"Collected {len(traces)} traces from {len(tasks)} tasks "
            f"({len(traces)/max(len(tasks), 1)*100:.0f}% yield)"
        )
        return traces

    async def _process_task(self, task_input: str) -> Optional[Trace]:
        """Process a single task and return a Trace if quality is sufficient."""
        start = time.perf_counter()
        try:
            output = await self.pipeline.run(task_input)
        except Exception as e:
            logger.debug(f"Pipeline error for task '{task_input[:50]}...': {e}")
            return None

        latency = time.perf_counter() - start

        # Basic output validation
        if not output or len(output.strip()) < 50:
            logger.debug(f"Output too short ({len(output) if output else 0} chars), skipping")
            return None

        # Token estimation using provider
        tokens = self.judge_provider.estimate_tokens(task_input + output)

        # Quality check via LLM judge
        quality = await self._judge_quality(task_input, output)
        if quality < self.min_quality:
            logger.debug(f"Quality {quality:.1f} below threshold {self.min_quality}")
            return None

        trace = Trace(
            input=task_input,
            output=output,
            quality_score=quality,
            latency_seconds=latency,
            tokens=tokens,
        )
        logger.info(
            f"Collected trace: quality={quality:.1f}, "
            f"latency={latency:.2f}s, tokens={tokens:,}"
        )
        return trace

    async def _judge_quality(self, task_input: str, output: str) -> float:
        """Score output quality 0-10 using an LLM judge."""
        system_prompt = (
            "Rate this AI output on a scale of 0-10 for quality, "
            "completeness, accuracy, and usefulness. Respond with "
            'only JSON: {"score": <0-10>}'
        )
        user_prompt = f"Task: {task_input}\n\nOutput:\n{output[:4000]}\n\nScore:"
        
        try:
            raw = await self.judge_provider.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ], temperature=0)

            try:
                # Use regex for fallback to find scores in strings like "The score is 8.5"
                match = re.search(r'"score":\s*(\d+(\.\d+)?)', raw)
                if match:
                    return float(match.group(1))
                
                # Check for direct JSON
                data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
                return float(data.get("score", 0))
            except (json.JSONDecodeError, AttributeError, ValueError):
                nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", raw)
                return float(nums[0]) if nums else 5.0

        except Exception as e:
            logger.warning(f"Judge error: {e}")
            return 5.0

    @staticmethod
    def save_traces(traces: list[Trace], output_path: str) -> None:
        """Save traces to a JSONL file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace.to_dict()) + "\n")
        logger.info(f"Saved {len(traces)} traces to {path}")

    @staticmethod
    def load_traces(path: str) -> list[Trace]:
        """Load traces from a JSONL file."""
        traces = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    traces.append(Trace.from_dict(json.loads(line)))
        return traces
