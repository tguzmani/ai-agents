# AI Agents Laboratory

Experiments with LangChain and LangGraph using OpenRouter as the LLM provider.

## Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) package manager
- An [OpenRouter](https://openrouter.ai/) API key

## Setup

1. Clone the repository:

```bash
git clone git@github.com:tguzmani/ai-agents.git
cd ai-agents/laboratory
```

2. Install dependencies:

```bash
uv sync
```

3. Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-or-v1-your-openrouter-key
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

## Usage

```bash
uv run python main.py
```
