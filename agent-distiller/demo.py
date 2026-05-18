"""Demo script to generate a premium distraction report using mocks."""

import asyncio
import json
from pathlib import Path
from agent_distiller.models import ComparisonResult, DistillationReport
from agent_distiller.reporters import HTMLReporter

async def main():
    print("STARTING: Generating premium demonstration report...")
    
    # Create fake comparison results
    comparisons = [
        ComparisonResult(
            input="Compare LangGraph and CrewAI for production use",
            pipeline_output="# LangGraph vs CrewAI\n\nLangGraph is a graph-based orchestration framework...",
            distilled_output="# LangGraph vs CrewAI\n\nLangGraph is a sophisticated framework for building agents...",
            pipeline_quality=8.4,
            distilled_quality=8.1,
            pipeline_cost=0.1341,
            distilled_cost=0.0082,
            pipeline_latency=16.42,
            distilled_latency=2.15,
            pipeline_tokens=12340,
            distilled_tokens=1847,
        ),
        ComparisonResult(
            input="Analyze the impact of MCP protocol adoption in 2026",
            pipeline_output="# MCP Protocol Impact\n\nThe Model Context Protocol (MCP) has revolutionized...",
            distilled_output="# MCP Protocol Review\n\nThe adoption of MCP in 2026 is a major milestone...",
            pipeline_quality=9.1,
            distilled_quality=8.8,
            pipeline_cost=0.0982,
            distilled_cost=0.0075,
            pipeline_latency=14.18,
            distilled_latency=1.82,
            pipeline_tokens=10847,
            distilled_tokens=1562,
        )
    ]
    
    report = DistillationReport(
        comparisons=comparisons,
        pipeline_name="ResearchPipeline (3 agents)",
        distilled_model="ft:gpt-4o-mini:my-org:distilled-v1",
        system_prompt="You are an expert research analyst. Produce a thorough report..."
    )
    
    # Add base model to report data manually for demo
    report_dict = report.to_dict()
    report_dict["metrics"]["base_model"] = "gpt-4o-mini"
    
    # Actually we just pass the report object to the reporter
    output_path = "premium_demo_report.html"
    reporter = HTMLReporter(output_path)
    reporter.generate(report)
    
    # Generate JSON too
    with open("premium_demo_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
        
    print(f"Success! Premium report generated: {output_path}")
    print(f"Quality: {report.avg_distilled_quality:.1f} / {report.avg_pipeline_quality:.1f} ({report.quality_retention:.1%})")
    print(f"Cost: ${report.avg_distilled_cost:.4f} (vs ${report.avg_pipeline_cost:.4f})  - {report.cost_reduction:.1%} saved")

if __name__ == "__main__":
    asyncio.run(main())
