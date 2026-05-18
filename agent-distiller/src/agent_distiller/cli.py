"""CLI for agent-distiller."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import click
import yaml

from .collectors import Collector
from .evaluators import Evaluator
from .models import DistillationReport
from .reporters import HTMLReporter
from .trainers import Trainer


def _load_pipeline(path: str):
    if ":" not in path:
        click.echo(f"Error: Pipeline must be 'module:Class', got '{path}'")
        sys.exit(1)
    mod, cls = path.rsplit(":", 1)
    try:
        # Add current directory to path so it can find local modules
        import os
        sys.path.append(os.getcwd())
        return getattr(importlib.import_module(mod), cls)()
    except Exception as e:
        click.echo(f"Error loading '{path}': {e}")
        sys.exit(1)


def _print_report(report: DistillationReport) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        console.print()
        table = Table(
            title="agent-distiller — Distillation Report",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Pipeline", justify="right")
        table.add_column("Distilled", justify="right")
        table.add_column("Retention / Change", justify="right")

        q_ret = report.quality_retention
        q_style = "green" if q_ret >= 0.95 else "bright_green" if q_ret >= 0.9 else "yellow" if q_ret >= 0.8 else "red"

        table.add_row(
            "Quality (LLM judge)",
            f"{report.avg_pipeline_quality:.2f}/10",
            f"{report.avg_distilled_quality:.2f}/10",
            f"[{q_style}]{q_ret*100:.1f}%[/{q_style}]",
        )
        table.add_row(
            "Avg cost per run",
            f"${report.avg_pipeline_cost:.4f}",
            f"${report.avg_distilled_cost:.4f}",
            f"[green]-{report.cost_reduction*100:.1f}%[/green]",
        )
        table.add_row(
            "Avg latency",
            f"{report.avg_pipeline_latency:.2f}s",
            f"{report.avg_distilled_latency:.2f}s",
            f"[green]-{report.latency_reduction*100:.1f}%[/green]",
        )

        console.print(table)

        verdict = report.verdict
        v_style = {
            "excellent": "[green bold]✅ Excellent distillation — Ready for production[/]",
            "good": "[green]✅ Good distillation — Production-worthy[/]",
            "acceptable": "[yellow]⚠️ Acceptable (consider more training data)[/]",
            "insufficient": "[red]❌ Insufficient (need better base model or more traces)[/]",
        }.get(verdict, verdict)

        console.print(f"\n  Verdict: {v_style}")
        console.print(
            f"  Quality retained: [{q_style}]{q_ret:.1%}[/{q_style}] │ "
            f"Cost saved: [green]{report.cost_reduction:.1%}[/green]"
        )
        console.print()

    except ImportError:
        d = report.to_dict()
        print(f"\nDistillation Report")
        print(f"  Quality: {d['metrics']['quality_pipeline']:.2f} → {d['metrics']['quality_distilled']:.2f}")
        print(f"  Quality retained: {d['metrics']['quality_retention']:.1%}")
        print(f"  Cost reduction: {d['metrics']['cost_reduction']:.1%}")
        print(f"  Verdict: {d['verdict']}")


@click.group("agent-distill")
@click.version_option(version="0.1.0", prog_name="agent-distiller")
def cli():
    """Compress multi-agent pipelines into single fine-tuned models."""


@cli.command()
@click.option("--pipeline", required=True, help="Pipeline as 'module:Class'")
@click.option("--tasks", required=True, help="Tasks JSONL file")
@click.option("--output", "-o", default="traces.jsonl", help="Output traces file")
@click.option("--num-tasks", type=int, help="Max tasks to process")
@click.option("--min-quality", type=float, default=7.0, help="Min quality score (0-10)")
@click.option("--parallel", type=int, default=1, help="Parallel task execution")
def collect(pipeline, tasks, output, num_tasks, min_quality, parallel):
    """Collect traces from your multi-agent pipeline."""
    target = _load_pipeline(pipeline)
    collector = Collector(
        pipeline=target,
        min_quality=min_quality,
        parallel=parallel,
    )
    traces = asyncio.run(collector.collect_from_file(tasks, num_tasks))
    collector.save_traces(traces, output)
    click.echo(f"✅ Collected {len(traces)} traces → {output}")


@cli.command()
@click.option("--traces", required=True, help="Traces JSONL file")
@click.option("--base-model", default="gpt-4o-mini", help="Base model for fine-tuning")
@click.option("--system-prompt", default="", help="System prompt for distilled model")
@click.option("--epochs", type=int, default=3)
def train(traces, base_model, system_prompt, epochs):
    """Fine-tune a model on collected traces."""
    trainer = Trainer(base_model=base_model, epochs=epochs)
    if not system_prompt:
        system_prompt = "You are a helpful AI assistant."
    model_id = asyncio.run(trainer.train(traces, system_prompt, epochs))
    click.echo(f"✅ Fine-tuned model: {model_id}")


@cli.command()
@click.option("--traces", required=True, help="Traces JSONL file")
@click.option("--base-model", default="gpt-4o-mini")
@click.option("--epochs", type=int, default=3)
def estimate(traces, base_model, epochs):
    """Estimate training cost without running."""
    trainer = Trainer(base_model=base_model, epochs=epochs)
    est = trainer.estimate_cost(traces, epochs)
    click.echo(f"\nTraining cost estimate:")
    click.echo(f"  Traces: {est['traces']}")
    click.echo(f"  Total tokens: {est['total_tokens']:,}")
    click.echo(f"  Avg tokens/example: {est['avg_tokens_per_example']:,}")
    click.echo(f"  Epochs: {est['epochs']}")
    click.echo(f"  Estimated cost: ${est['estimated_cost_usd']:.2f}")
    click.echo(f"  Base model: {est['base_model']}")


@cli.command()
@click.option("--pipeline", required=True, help="Pipeline as 'module:Class'")
@click.option("--distilled-model", required=True, help="Fine-tuned model ID")
@click.option("--test-tasks", required=True, help="Test tasks JSONL file")
@click.option("--num-tasks", type=int, default=20)
@click.option("--judge-model", default="gpt-4o")
@click.option("-o", "--output", default="comparison.json")
@click.option("--html/--no-html", default=True, help="Generate HTML report")
def evaluate(pipeline, distilled_model, test_tasks, num_tasks, judge_model, output, html):
    """Evaluate distilled model against the original pipeline."""
    target = _load_pipeline(pipeline)
    evaluator = Evaluator(
        pipeline=target,
        distilled_model=distilled_model,
        judge_model=judge_model,
    )
    report = asyncio.run(evaluator.evaluate_from_file(test_tasks, num_tasks))
    _print_report(report)

    Path(output).write_text(json.dumps(report.to_dict(), indent=2))
    click.echo(f"📄 JSON Report saved → {output}")
    
    if html:
        html_path = output.replace(".json", ".html")
        reporter = HTMLReporter(html_path)
        reporter.generate(report)
        click.echo(f"🎨 HTML Report saved → {html_path}")


@cli.command("run")
@click.option("--config", required=True, type=click.Path(exists=True))
def run_all(config):
    """Run full Collect → Train → Evaluate pipeline from config."""
    with open(config) as f:
        cfg = yaml.safe_load(f)
    
    click.echo(f"🚀 Starting full distillation pipeline from {config}")
    
    # 1. Collect
    c_cfg = cfg.get("collect", {})
    target = _load_pipeline(cfg["pipeline"]["module"] + ":" + cfg["pipeline"]["class"])
    collector = Collector(
        pipeline=target,
        min_quality=c_cfg.get("min_quality", 7.0),
        parallel=c_cfg.get("parallel", 1),
        judge_model=c_cfg.get("judge_model", "gpt-4o-mini"),
    )
    click.echo("Step 1: Collecting traces...")
    traces = asyncio.run(collector.collect_from_file(
        c_cfg.get("tasks_file", "tasks.jsonl"), 
        c_cfg.get("max_tasks")
    ))
    traces_path = c_cfg.get("output", "traces.jsonl")
    collector.save_traces(traces, traces_path)
    
    # 2. Train
    t_cfg = cfg.get("train", {})
    trainer = Trainer(
        base_model=t_cfg.get("base_model", "gpt-4o-mini"),
        epochs=t_cfg.get("epochs", 3)
    )
    system_prompt = t_cfg.get("system_prompt", target.get_system_prompt())
    click.echo("Step 2: Fine-tuning model...")
    model_id = asyncio.run(trainer.train(traces_path, system_prompt))
    click.echo(f"✅ Fine-tuned model: {model_id}")
    
    # 3. Evaluate
    e_cfg = cfg.get("evaluate", {})
    evaluator = Evaluator(
        pipeline=target,
        distilled_model=model_id,
        judge_model=e_cfg.get("judge_model", "gpt-4o"),
        system_prompt=system_prompt,
    )
    click.echo("Step 3: Evaluating...")
    report = asyncio.run(evaluator.evaluate_from_file(
        e_cfg.get("test_tasks", "test_tasks.jsonl"),
        e_cfg.get("num_tasks", 20)
    ))
    _print_report(report)
    
    # Save reports
    report_json = e_cfg.get("output", "distillation_report.json")
    Path(report_json).write_text(json.dumps(report.to_dict(), indent=2))
    
    report_html = report_json.replace(".json", ".html")
    HTMLReporter(report_html).generate(report)
    
    click.echo(f"\n✨ Distillation complete!")
    click.echo(f"📊 Final Model: {model_id}")
    click.echo(f"📄 Report: {report_html}")


def main():
    cli()


if __name__ == "__main__":
    main()
