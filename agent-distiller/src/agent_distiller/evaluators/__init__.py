"""Evaluator — compare pipeline vs. distilled model quality."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Optional

from ..models import ComparisonResult, DistillationReport, PipelineTarget
from ..providers.openai import OpenAIProvider

logger = logging.getLogger("agent-distiller")


class Evaluator:
    """Run pipeline and distilled model on the same tasks and compare."""

    def __init__(
        self,
        pipeline: PipelineTarget,
        distilled_model: str,
        judge_model: str = "gpt-4o",
        system_prompt: str = "",
        parallel: int = 1,
    ):
        self.pipeline = pipeline
        self.distilled_model = distilled_model
        self.parallel = parallel
        self.system_prompt = system_prompt or pipeline.get_system_prompt()
        self.distilled_provider = OpenAIProvider(distilled_model)
        self.judge_provider = OpenAIProvider(judge_model)

    async def evaluate_from_file(
        self, tasks_path: str, max_tasks: Optional[int] = None
    ) -> DistillationReport:
        """Load test tasks and run evaluation."""
        tasks = []
        with open(tasks_path) as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        tasks.append(data.get("input", data.get("task", "")))
                    except json.JSONDecodeError:
                        continue
        if max_tasks:
            tasks = tasks[:max_tasks]
        return await self.evaluate(tasks)

    async def evaluate(self, tasks: list[str]) -> DistillationReport:
        """Run both systems on tasks and generate comparison report."""
        report = DistillationReport(
            pipeline_name=self.pipeline.__class__.__name__,
            distilled_model=self.distilled_model,
            system_prompt=self.system_prompt,
        )

        self.pipeline.setup()
        try:
            # Semaphores for concurrent evaluation
            semaphore = asyncio.Semaphore(self.parallel)

            async def _sem_compare(t: str):
                async with semaphore:
                    return await self._compare_one(t)

            batch_tasks = [_sem_compare(t) for t in tasks]
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, ComparisonResult):
                    report.comparisons.append(result)
                    logger.info(
                        f"Task evaluated: pipeline={result.pipeline_quality:.1f} "
                        f"distilled={result.distilled_quality:.1f}"
                    )
                else:
                    logger.warning(f"Evaluation task failed: {result}")

        finally:
            self.pipeline.teardown()

        return report

    async def _compare_one(self, task_input: str) -> ComparisonResult:
        """Run both pipeline and distilled model, score both outputs."""
        # 1. Run pipeline
        p_start = time.monotonic()
        try:
            p_output = await self.pipeline.run(task_input)
        except Exception as e:
            logger.debug(f"Pipeline error: {e}")
            p_output = f"[Pipeline error: {e}]"
        p_latency = time.monotonic() - p_start
        p_tokens = self.judge_provider.estimate_tokens(task_input + p_output)

        # 2. Run distilled model
        d_start = time.monotonic()
        try:
            d_output = await self._call_distilled(task_input)
        except Exception as e:
            logger.debug(f"Distilled model error: {e}")
            d_output = f"[Distilled model error: {e}]"
        d_latency = time.monotonic() - d_start
        d_tokens = self.distilled_provider.estimate_tokens(task_input + d_output)

        # 3. Score both outputs
        p_quality = await self._judge(task_input, p_output)
        d_quality = await self._judge(task_input, d_output)

        # 4. Calculate costs
        p_cost = self.judge_provider.get_cost(self.judge_provider.estimate_tokens(task_input), self.judge_provider.estimate_tokens(p_output))
        d_cost = self.distilled_provider.get_cost(self.distilled_provider.estimate_tokens(task_input), self.distilled_provider.estimate_tokens(d_output))

        return ComparisonResult(
            input=task_input,
            pipeline_output=p_output,
            distilled_output=d_output,
            pipeline_quality=p_quality,
            distilled_quality=d_quality,
            pipeline_latency=p_latency,
            distilled_latency=d_latency,
            pipeline_tokens=p_tokens,
            distilled_tokens=d_tokens,
            pipeline_cost=p_cost,
            distilled_cost=d_cost,
        )

    async def _call_distilled(self, task_input: str) -> str:
        """Call the distilled (fine-tuned) model."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": task_input})

        return await self.distilled_provider.chat(messages, temperature=0)

    async def _judge(self, task_input: str, output: str) -> float:
        """Score output quality 0-10."""
        system_prompt = (
            "Rate this output 0-10 for quality, accuracy, "
            "completeness, and usefulness. Respond with only "
            'JSON: {"score": <0-10>}'
        )
        user_prompt = f"Task: {task_input}\n\nOutput:\n{output[:4000]}"
        
        try:
            raw = await self.judge_provider.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ], temperature=0)

            try:
                match = re.search(r'"score":\s*(\d+(\.\d+)?)', raw)
                if match:
                    return float(match.group(1))
                data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
                return float(data.get("score", 0))
            except (json.JSONDecodeError, AttributeError, ValueError):
                nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", raw)
                return float(nums[0]) if nums else 5.0
        except Exception as e:
            logger.warning(f"Judge error: {e}")
            return 5.0
