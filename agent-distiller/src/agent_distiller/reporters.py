"""HTML Reporter for agent-distiller distillation results."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from .models import DistillationReport


class HTMLReporter:
    """Generate a premium HTML report for distillation results."""

    def __init__(self, output_path: str = "report.html"):
        self.output_path = output_path

    def generate(self, report: DistillationReport) -> str:
        """Generate the HTML report from a DistillationReport object."""
        data = report.to_dict()
        metrics = data["metrics"]
        
        # Format metrics for display
        q_retention = metrics["quality_retention"] * 100
        cost_reduction = metrics["cost_reduction"] * 100
        lat_reduction = metrics["latency_reduction"] * 100
        
        verdict_class = f"verdict-{data['verdict']}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>agent-distiller — Distillation Report</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --primary: #38bdf8;
            --secondary: #818cf8;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --text-main: #f1f5f9;
            --text-dim: #94a3b8;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-color);
            background-image: radial-gradient(circle at 50% 50%, #1e1b4b 0%, #0f172a 100%);
            color: var(--text-main);
            line-height: 1.6;
            min-height: 100vh;
            padding: 2rem;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
        }}

        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .timestamp {{ color: var(--text-dim); font-size: 0.9rem; }}

        .verdict-badge {{
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 99px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1rem;
        }}
        .verdict-excellent {{ background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid var(--success); }}
        .verdict-good {{ background: rgba(56, 189, 248, 0.2); color: var(--primary); border: 1px solid var(--primary); }}
        .verdict-acceptable {{ background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }}
        .verdict-insufficient {{ background: rgba(239, 68, 68, 0.2); color: var(--error); border: 1px solid var(--error); }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}

        .stat-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            border-radius: 1rem;
            text-align: center;
            transition: transform 0.2s;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-label {{ font-size: 0.9rem; color: var(--text-dim); margin-bottom: 0.5rem; }}
        .stat-value {{ font-size: 2rem; font-weight: 700; color: var(--primary); }}
        .stat-sub {{ font-size: 0.8rem; color: var(--text-dim); margin-top: 0.5rem; }}

        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 1rem;
            overflow: hidden;
            margin-bottom: 3rem;
        }}

        th, td {{ padding: 1.25rem; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }}
        th {{ background: rgba(255, 255, 255, 0.03); color: var(--text-dim); font-weight: 600; text-transform: uppercase; font-size: 0.8rem; }}
        
        .retention-bar {{
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 1rem;
        }}
        .retention-fill {{
            height: 100%;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            border-radius: 4px;
        }}

        .info-section {{
            background: rgba(255, 255, 255, 0.02);
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 3rem;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}
        .info-item h3 {{ font-size: 0.9rem; color: var(--text-dim); margin-bottom: 0.5rem; }}
        .info-item p {{ font-weight: 500; font-family: monospace; }}

        footer {{
            text-align: center;
            color: var(--text-dim);
            font-size: 0.9rem;
            margin-top: 4rem;
        }}

        @media (max-width: 768px) {{
            .info-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>agent-distiller Report</h1>
            <p class="timestamp">Generated on {timestamp}</p>
            <div class="verdict-badge {verdict_class}">Verdict: {data['verdict']}</div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Quality Retention</div>
                <div class="stat-value">{q_retention:.1f}%</div>
                <div class="stat-sub">{metrics['quality_distilled']:.1f} vs {metrics['quality_pipeline']:.1f} (avg)</div>
                <div class="retention-bar">
                    <div class="retention-fill" style="width: {q_retention}%"></div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Cost Reduction</div>
                <div class="stat-value">{cost_reduction:.1f}%</div>
                <div class="stat-sub">${metrics['cost_distilled']:.4f} vs ${metrics['cost_pipeline']:.4f} / run</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Latency Reduction</div>
                <div class="stat-value">{lat_reduction:.1f}%</div>
                <div class="stat-sub">{metrics['latency_distilled']:.2f}s vs {metrics['latency_pipeline']:.2f}s (avg)</div>
            </div>
        </div>

        <div class="info-section">
            <div class="info-grid">
                <div class="info-item">
                    <h3>Pipeline</h3>
                    <p>{data['pipeline']}</p>
                </div>
                <div class="info-item">
                    <h3>Distilled Model</h3>
                    <p>{data['distilled_model']}</p>
                </div>
                <div class="info-item">
                    <h3>Tasks Evaluated</h3>
                    <p>{data['num_tasks']}</p>
                </div>
                <div class="info-item">
                    <h3>Base Model</h3>
                    <p>{metrics.get('base_model', 'N/A')}</p>
                </div>
            </div>
        </div>

        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Full Pipeline</th>
                    <th>Distilled Model</th>
                    <th>Efficiency</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Avg. Quality</td>
                    <td>{metrics['quality_pipeline']:.1f}/10</td>
                    <td>{metrics['quality_distilled']:.1f}/10</td>
                    <td>{q_retention:.1f}% retained</td>
                </tr>
                <tr>
                    <td>Avg. Cost</td>
                    <td>${metrics['cost_pipeline']:.4f}</td>
                    <td>${metrics['cost_distilled']:.4f}</td>
                    <td>{cost_reduction:.1f}% saved</td>
                </tr>
                <tr>
                    <td>Avg. Latency</td>
                    <td>{metrics['latency_pipeline']:.2f}s</td>
                    <td>{metrics['latency_distilled']:.2f}s</td>
                    <td>{lat_reduction:.1f}% faster</td>
                </tr>
            </tbody>
        </table>

        <footer>
            <p>Made with ❤️ by agent-distiller</p>
        </footer>
    </div>
</body>
</html>
"""
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_template)
            
        return self.output_path
