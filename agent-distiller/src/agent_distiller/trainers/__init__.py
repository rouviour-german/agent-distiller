"""Trainer — fine-tune a single model on collected pipeline traces."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

from ..models import Trace
from ..providers.openai import OpenAIProvider

logger = logging.getLogger("agent-distiller")


class Trainer:
    """Convert traces to fine-tuning data and run a fine-tuning job."""

    def __init__(
        self,
        base_model: str = "gpt-4o-mini",
        provider_name: str = "openai",
        epochs: int = 3,
        validation_split: float = 0.1,
    ):
        self.base_model = base_model
        self.provider_name = provider_name
        self.epochs = epochs
        self.validation_split = validation_split
        
        if provider_name == "openai":
            self.provider = OpenAIProvider(base_model)
        else:
            raise NotImplementedError(f"Provider '{provider_name}' not yet supported.")

    def prepare_training_data(
        self,
        traces: list[Trace],
        system_prompt: str,
        output_path: str = "training_data.jsonl",
        validation_path: Optional[str] = "validation_data.jsonl",
    ) -> tuple[str, Optional[str]]:
        """Convert traces to fine-tuning JSONL format."""
        examples = [t.to_training_example(system_prompt) for t in traces]

        # Split into train/validation
        split_idx = max(1, int(len(examples) * (1 - self.validation_split)))
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:] if self.validation_split > 0 else []

        # Write training file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in train_examples:
                f.write(json.dumps(ex) + "\n")

        logger.info(
            f"Training data: {len(train_examples)} examples → {output_path}"
        )

        # Write validation file
        val_path = None
        if val_examples and validation_path:
            with open(validation_path, "w") as f:
                for ex in val_examples:
                    f.write(json.dumps(ex) + "\n")
            val_path = validation_path
            logger.info(
                f"Validation data: {len(val_examples)} examples → {val_path}"
            )

        return output_path, val_path

    async def train(
        self,
        traces_path: str,
        system_prompt: str,
        epochs: Optional[int] = None,
    ) -> str:
        """Run the full fine-tuning pipeline. Returns the model ID."""
        # Load traces
        traces = []
        with open(traces_path) as f:
            for line in f:
                if line.strip():
                    traces.append(Trace.from_dict(json.loads(line)))

        logger.info(f"Loaded {len(traces)} traces from {traces_path}")

        # Prepare training data
        train_path, val_path = self.prepare_training_data(
            traces, system_prompt
        )

        # Estimate cost
        total_tokens = sum(t.tokens for t in traces)
        estimated_cost = self.provider.get_training_cost(total_tokens, epochs or self.epochs)
        logger.info(
            f"Estimated training cost: ${estimated_cost:.2f} "
            f"({total_tokens:,} tokens × {epochs or self.epochs} epochs)"
        )

        if self.provider_name == "openai":
            return await self._train_openai(
                train_path, val_path, epochs or self.epochs
            )
        else:
            raise NotImplementedError(
                f"Provider '{self.provider_name}' not yet supported. "
                f"Supported: openai"
            )

    async def _train_openai(
        self, train_path: str, val_path: Optional[str], epochs: int
    ) -> str:
        """Fine-tune using OpenAI API."""
        import openai

        # Use the provider's client if possible, but fine-tuning is special
        # OpenAI fine-tuning still uses the client objects
        async_client = openai.AsyncOpenAI()

        # Upload training file
        logger.info("Uploading training file...")
        with open(train_path, "rb") as f:
            train_file = await async_client.files.create(file=f, purpose="fine-tune")

        # Upload validation file
        val_file_id = None
        if val_path:
            with open(val_path, "rb") as f:
                val_file = await async_client.files.create(file=f, purpose="fine-tune")
                val_file_id = val_file.id

        # Create fine-tuning job
        logger.info(f"Creating fine-tuning job (base: {self.base_model})...")
        job_params = {
            "training_file": train_file.id,
            "model": self.base_model,
            "hyperparameters": {"n_epochs": epochs},
        }
        if val_file_id:
            job_params["validation_file"] = val_file_id

        job = await async_client.fine_tuning.jobs.create(**job_params)
        logger.info(f"Job created: {job.id}")

        # Poll for completion
        while True:
            job = await async_client.fine_tuning.jobs.retrieve(job.id)
            status = job.status
            logger.info(f"Training status: {status}")

            if status == "succeeded":
                model_id = job.fine_tuned_model
                logger.info(f"Training complete! Model: {model_id}")
                return model_id or ""
            elif status in ("failed", "cancelled"):
                error = getattr(job, "error", {})
                raise RuntimeError(
                    f"Fine-tuning {status}: {error}"
                )

            await asyncio.sleep(60)  # Check every 60 seconds

    def estimate_cost(self, traces_path: str, epochs: Optional[int] = None) -> dict:
        """Estimate training cost without running."""
        traces = []
        with open(traces_path) as f:
            for line in f:
                if line.strip():
                    traces.append(Trace.from_dict(json.loads(line)))

        total_tokens = sum(t.tokens for t in traces)
        ep = epochs or self.epochs
        cost = self.provider.get_training_cost(total_tokens, ep)

        return {
            "traces": len(traces),
            "total_tokens": total_tokens,
            "avg_tokens_per_example": total_tokens // max(len(traces), 1),
            "epochs": ep,
            "estimated_cost_usd": round(cost, 4),
            "base_model": self.base_model,
        }
