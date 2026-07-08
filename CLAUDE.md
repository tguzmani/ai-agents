# CLAUDE.md

## Language

- **Always respond in Spanish.** Use a **neutral Venezuelan** register — NOT
  Argentine. That means:
  - Use **"tú"**, never "vos" (e.g. "si tú quieres", NOT "si vos querés").
  - Standard tuteo conjugations: "tú tienes / quieres / puedes", NOT
    "tú tenés / querés / podés".
  - Avoid Argentine slang and voseo imperatives ("quedate", "fijate", "dale") —
    prefer neutral forms ("quédate", "fíjate", "listo").
- Technical terms, code, and library/API names stay in English as usual.

## About this project

This entire repository is a **learning course** for the user to study AI agents,
LLMs, LangChain/LangSmith, and Python. It is not production software — the goal is
learning and experimentation, so exercises, rough edges, and "figuring things out"
are expected and welcome.

## About the user's background

- The user is an experienced software engineer with **~6 years of TypeScript/JavaScript**.
- They **knew Python from many years ago** (before their TS/JS career), so core
  programming concepts are second nature — they are NOT a beginner programmer.
- However, they have **not used Python much recently**, so some Python-specific
  idioms, tooling, and stdlib details may be rusty or unfamiliar.

## Tooling

- This project uses **`uv`** for Python environment and package management. Always
  prefix Python/pip commands with `uv`:
  - Run scripts: `uv run python foo.py` (NOT bare `python foo.py`)
  - Inspect packages: `uv pip show <pkg>` (NOT `pip show <pkg>`)
  - Add deps: `uv add <pkg>` (NOT `pip install <pkg>`)
  - The `TS/JS` parallel: `uv` ≈ `npm`/`pnpm` (manages a lockfile + `.venv`);
    `uv run` ≈ `npm run`/`npx` (executes inside the project env).

## How to help

- Assume strong general programming fluency. Skip explanations of basic concepts
  (variables, loops, types, async, OOP, etc.).
- Focus explanations on **what's different or Python-specific**, and when useful,
  **draw the parallel to TS/JS** (e.g. `os.environ` ≈ `process.env`,
  `python-dotenv` ≈ `dotenv`, `list[str]` ≈ `string[]`).
- Point out Python idioms, gotchas, and "the Pythonic way" when relevant.
- It's fine to be direct and flag bugs or anti-patterns in the exercise code.
