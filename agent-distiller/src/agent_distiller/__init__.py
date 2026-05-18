"""agent-distiller: Compress multi-agent pipelines into single fine-tuned models."""

from .models import (
    ComparisonResult,
    DistillationReport,
    PipelineTarget,
    Trace,
)
from .collectors import Collector
from .trainers import Trainer
from .evaluators import Evaluator
from .reporters import HTMLReporter

__version__ = "0.1.0"
__all__ = [
    "Collector",
    "ComparisonResult",
    "DistillationReport",
    "Evaluator",
    "HTMLReporter",
    "PipelineTarget",
    "Trace",
    "Trainer",
]
