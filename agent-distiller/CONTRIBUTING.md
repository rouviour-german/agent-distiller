# Contributing to agent-distiller

## Setup
```bash
git clone https://github.com/daniellopez882/agent-distiller.git
cd agent-distiller
pip install -e ".[dev,all]"
pytest tests/ -v
```

## High-Impact Contributions

### Fine-Tuning Providers
- **Together AI** — fine-tune open-source models (Llama, Mistral, Qwen)
- **Local LoRA** — fine-tune locally via HuggingFace transformers + PEFT
- **Anthropic** — when Anthropic opens fine-tuning API
- **Fireworks AI** — serverless fine-tuning

### Features
- **Trace augmentation** — generate synthetic variations of successful traces
- **Tool-use distillation** — capture tool calls in training data
- **Chain-of-thought capture** — record intermediate reasoning, not just input→output
- **HTML comparison report** — shareable visual report with charts
- **Incremental distillation** — add new traces without full retrain

### Pipeline Adapters
- Pre-built `PipelineTarget` for LangGraph, CrewAI, agent-compose

## Code Style
- Python 3.10+ with type hints
- Lint: `ruff check src/ tests/`
- Tests for all three phases (collect, train, evaluate)
- Both mock and integration test patterns
