# LLM Deep Dive Workshop

# Prompt Engineering Patterns

A hands-on collection of prompt-engineering patterns and advanced LLM techniques that connects to **Azure OpenAI**, **OpenAI**, **Anthropic**, **Google Gemini**, or a locally running **Ollama** instance, with pluggable provider configuration, built-in token usage tracking, and a unified client interface. Each numbered script is a self-contained example that you can run, read, and adapt. It is designed as a teaching resource to demonstrate practical prompt-engineering patterns in Python.

---

> [!WARNING]
> **Testing Status:** This codebase has been fully tested with **Ollama** and **Azure OpenAI** only. Support for **OpenAI**, **Anthropic**, and **Google Gemini** is implemented and should work, but has **not been independently verified**. If you encounter issues with those providers, check that credentials are correctly set and the provider adapter is behaving as expected.

---

## Table of Contents

- [Prompt Examples](#prompt-examples)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Verifying Your Connection](#verifying-your-connection)
- [Installing Ollama](#installing-ollama)
- [LLM Provider Configuration](#llm-provider-configuration)
- [Troubleshooting & Common Issues](#troubleshooting--common-issues)
- [License](#license)

---

## Prompt Examples

This repository walks through **17 progressive examples** across two parts, each building on concepts introduced before it.

### Part 1 — Foundational Prompt Patterns (Examples 1–6)

These scripts use a **synchronous** client and focus on understanding prompt structure, few-shot learning, chain-of-thought reasoning, and structured data extraction.

---

#### `1_prompt_anatomy.py` — Prompt Structure & Anatomy

> **Key Feature: Side-by-side comparison of 5 prompt structures on the same task.**

Demonstrates how the structure of a prompt — not just its words — determines output quality. Uses the same meeting-notes summarization task across five increasingly detailed prompt designs:

| Style | What it shows |
|-------|--------------|
| Vague / no structure | Baseline — minimal instruction |
| Instruction only | Adds a clear directive |
| Instruction + constraints | Adds rules like bullet format and owner/deadline |
| All 5 components with delimiters | Instruction, context, constraints, examples, output format |
| Chain-of-thought variant | Forces step-by-step reasoning before the answer |

**Run it to see** how each structural layer improves precision and output consistency.

---

#### `2_few_shot_classifier.py` — Few-Shot Sentiment Classifier

> **Key Feature: Configurable shot count (0, 1, 3, 5) so you can directly observe the accuracy improvement curve.**

Builds a product-review sentiment classifier (labels: `POSITIVE`, `NEGATIVE`, `NEUTRAL`, `MIXED`) with a bank of five labelled examples. Runs the same set of test reviews at each shot level and prints a comparison table.

**Run it to see** how even one labelled example dramatically reduces ambiguous or incorrect classifications compared to zero-shot prompting.

---

#### `3_cot_math_solver.py` — Chain-of-Thought Math Solver

> **Key Feature: Three solving strategies compared — direct answer, zero-shot CoT ("Let's think step by step"), and structured few-shot CoT.**

Runs three multi-step word problems through each strategy side by side, reporting the answer and token count for each. Illustrates that direct prompting often produces wrong answers on problems requiring intermediate steps, while CoT reliably recovers the correct result.

**Run it to see** the accuracy gap close as reasoning steps are added to the prompt.

---

#### `4_system_prompt_config.py` — System Prompt Strategies

Compares three approaches to placing persona and instruction text:

1. Everything packed into the user turn (no system prompt)
2. Persona in system, task in user — the recommended split
3. Heavy system prompt with exhaustive constraints and rules

Demonstrates how the system/user split affects tone, instruction adherence, and verbosity. Useful for understanding where in the conversation to place stable vs. variable instructions.

---

#### `5_invoice_extractor.py` — Structured Invoice Extraction

> **Key Feature: Uses `instructor` + Pydantic for type-safe, validated extraction with automatic retry on schema violations.**

Parses messy plain-text invoice data into a typed `Invoice` Pydantic model (`vendor_name`, `invoice_number`, `line_items`, `total`, `currency`, etc.). Automatically selects the correct instructor extraction mode per provider:

- `TOOLS` mode for Azure / OpenAI (native function calling)
- `JSON` mode for Anthropic / Gemini
- `MD_JSON` mode for Ollama

**Run it to see** structured extraction that fails gracefully on bad data and validates field constraints (e.g. non-negative quantities and prices).

---

#### `6_prompt_debugger.py` — Prompt Ablation Debugger

> **Key Feature: Automated ablation testing — removes one prompt component at a time to identify which component is doing the work.**

Given a prompt broken into named components (persona, rules, examples, format instruction), it runs the full prompt and then N variants with one component removed each time. Outputs a comparison of how each removal changes the model's response, surfacing which components are load-bearing.

**Run it to see** which parts of a complex prompt actually drive the desired behaviour — and which are redundant.

---

### Part 2 — Advanced & Agentic Patterns (Examples 7–17)

These scripts use an **asynchronous** client (`asyncio`) and the built-in `TokenTracker` for usage monitoring. Each script prints a token-usage summary at the end.

---

#### `7_react_agent.py` — ReAct Agent from Scratch

> **Key Feature: Full Thought → Action → Observation loop implemented without any agent framework.**

Implements the ReAct (Reason + Act) pattern with two tools: `calculator` (safe `eval` of math expressions) and `web_search` (keyword-matched mock responses). The agent loops until it emits `finish[<answer>]`, with parse-error handling for malformed action lines.

---

#### `8_self_consistency.py` — Self-Consistency with Majority Voting

Samples N independent reasoning paths for the same problem in parallel, then aggregates answers via majority vote. Prints each path's full reasoning and highlights which answer wins. Demonstrates that ensemble decoding can recover the correct answer even when individual paths disagree.

---

#### `9_research_chain.py` — Sequential Research Chain

Implements a three-step pipeline:

1. **Plan** — LLM generates three subtopics for the given research topic (JSON output)
2. **Research** — Each subtopic is investigated in parallel
3. **Synthesize** — All findings are combined into a coherent summary

Shows how to pass structured outputs between chain steps with lean context handoffs.

---

#### `10_branching_chain.py` — Classify-and-Route Branching Chain

Classifies an incoming customer message into `billing`, `technical`, or `general`, then routes it to a specialist system prompt for that category. Demonstrates conditional branching in a prompt pipeline without hardcoded if/else logic in the model call path.

---

#### `11_iterative_refinement.py` — Draft → Critique → Revise Loop

Runs a draft-critique-revise cycle with a quality-score-based early-stopping condition. The critic rates the draft 1–10 and signals `APPROVED` when quality reaches the threshold, short-circuiting the remaining iterations. Shows how to build self-improving pipelines without fixed iteration counts.

---

#### `12_role_based_prompting.py` — Role-Based Prompting Patterns

Demonstrates three production-ready role patterns:

1. **Expert consultant** — Security reviewer that always flags at least one CWE and provides a corrected code fix
2. **Audience-adaptive explainer** — Same concept explained differently for a 10-year-old, a developer, and an executive
3. **Multi-persona debate** — Three distinct personas argue different angles, then a synthesizer draws a conclusion

---

#### `13_meta_prompt_optimizer.py` — Meta-Prompt Optimizer

Uses an LLM to iteratively rewrite a prompt over N rounds, optionally seeded with real failure examples. Each round, the optimizer analyzes weaknesses in the current prompt and produces a strictly improved version. Useful for automating prompt engineering when you have a test set of failures.

---

#### `14_tree_of_thoughts.py` — Tree of Thoughts

Implements a simplified ToT search with configurable breadth and depth:

- At each step, generates `breadth` distinct candidate approaches
- Evaluates which is most promising
- Continues down the selected branch for `depth` steps

Gives a window into how systematic search over reasoning paths can tackle problems that single-chain prompting struggles with.

---

#### `15_self_critique_loop.py` — Self-Critique Loop (Reflexion)

Generate → Critique → Reflect → Regenerate. Failed attempts accumulate verbal reflections that are injected into the next attempt's system prompt as "lessons learned". Demonstrates the Reflexion pattern where the model improves by remembering its own past mistakes within a session.

---

#### `16_constitutional_ai.py` — Constitutional AI (Principles-Based Revision)

Generates an initial unconstrained response, checks it against a configurable list of principles (e.g. accuracy, safety, tone), and issues a targeted revision only for the principles that fail. If all principles pass, the original response is returned unchanged, avoiding unnecessary rewrites.

---

#### `17_least_to_most.py` — Least-to-Most Decomposition

Decomposes a complex problem into sub-problems ordered from simplest to hardest, then solves each incrementally — using solutions to earlier sub-problems as context for harder ones. Complements chain-of-thought by structuring the problem before reasoning begins.

---

## Project Structure

```
LLMDeepDiveWorkshop/
├── README.md                  # You are here
├── LICENSE                    # MIT licence
├── requirements.txt           # Python dependencies
├── keys/
│   └── .env                   # Your API credentials (not committed)
└── examples/
    ├── provider_config.py     # Unified provider registry, client factory & helpers
    ├── part_1/                # Foundational prompt patterns (1–6)
    │   ├── 1_prompt_anatomy.py
    │   ├── 2_few_shot_classifier.py
    │   ├── 3_cot_math_solver.py
    │   ├── 4_system_prompt_config.py
    │   ├── 5_invoice_extractor.py
    │   └── 6_prompt_debugger.py
    └── part_2/                # Advanced & agentic patterns (7–17)
        ├── 7_react_agent.py
        ├── 8_self_consistency.py
        ├── 9_research_chain.py
        ├── 10_branching_chain.py
        ├── 11_iterative_refinement.py
        ├── 12_role_based_prompting.py
        ├── 13_meta_prompt_optimizer.py
        ├── 14_tree_of_thoughts.py
        ├── 15_self_critique_loop.py
        ├── 16_constitutional_ai.py
        └── 17_least_to_most.py
```

---

## How It Works

The diagram below shows the full request flow from your environment variables to a model response in any example script.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         keys/.env                                   │
│  LLM_PROVIDER=azure   AZURE_OPENAI_API_KEY=...   AZURE_...          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  python-dotenv loads at import time
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    provider_config.py                               │
│                                                                     │
│  1. validate_provider()  — checks all required env vars are set     │
│  2. _create_sync_client()  /  _create_async_client()                │
│     ├─ azure     → openai.AzureOpenAI(...)                          │
│     ├─ openai    → openai.OpenAI(...)                               │
│     ├─ anthropic → _AnthropicAdapter  (translates SDK shape)        │
│     ├─ gemini    → openai.OpenAI(base_url=gemini_endpoint, ...)     │
│     └─ ollama    → openai.OpenAI(base_url="localhost:11434/v1", ..) │
│  3. Returns (client, MODEL_NAME, PROVIDER)                          │
│                                                                     │
│  Async path also returns TokenTracker + chat() helper               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  init_sync_client() / init_async_client()
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Example Script                                   │
│                                                                     │
│  client, MODEL, PROVIDER = init_sync_client()    ← Part 1          │
│  _, MODEL, _            = init_async_client()    ← Part 2          │
│                                                                     │
│  client.chat.completions.create(                                    │
│      model=MODEL,                                                   │
│      messages=[...],                                                │
│  )   ← identical call regardless of provider                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │   LLM Provider   │
                    │  (Azure / OAI /  │
                    │  Anthropic /     │
                    │  Gemini / Ollama)│
                    └──────────────────┘
```

**How the system operates:**

1. **Single `.env` file, single variable to switch providers.** Setting `LLM_PROVIDER=ollama` vs `LLM_PROVIDER=azure` is the only change needed to redirect all traffic. Credentials for each provider are namespaced (e.g. `AZURE_OPENAI_API_KEY` vs `GEMINI_API_KEY`) so multiple providers can coexist in `.env` and you switch between them by changing one line.

2. **Normalised client interface.** Every provider is wrapped to expose `client.chat.completions.create(...)` — the standard OpenAI SDK shape. Azure and OpenAI use the SDK natively. Gemini and Ollama are accessed via their OpenAI-compatible endpoints. Anthropic has a thin adapter that translates system messages and response structure to match the OpenAI shape.

3. **Sync vs Async split.** Part 1 examples use `init_sync_client()` — straightforward blocking calls, ideal for sequential demonstrations. Part 2 examples use `init_async_client()` and `asyncio`, enabling parallel LLM calls (e.g. sampling N reasoning paths simultaneously in `8_self_consistency.py` or researching subtopics in parallel in `9_research_chain.py`).

4. **Built-in token tracking (Part 2).** `TokenTracker` accumulates prompt and completion token counts across every `chat()` call in a script and prints a usage summary at the end. This lets you observe the token cost of each pattern directly from your terminal.

---

## Setup & Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10 or higher** — required for `match`/`case` syntax and `asyncio` improvements used in the codebase. Check with:

  ```bash
  python3 --version
  ```

- **An API key for at least one provider:**

  | Provider | Where to obtain |
  |----------|----------------|
  | Azure OpenAI | From your workshop organiser or the [Azure Portal](https://portal.azure.com) |
  | OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
  | Anthropic | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
  | Google Gemini | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
  | Ollama | No key needed — runs entirely locally. See [Installing Ollama](#installing-ollama) |

---

### 1. Clone the Repository

```bash
git clone https://github.com/aryanmishra98-08/LLMDeepDiveWorkshop.git
cd LLMDeepDiveWorkshop
git checkout Session-2
```

---

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

You should now see `(.venv)` prefixed in your terminal prompt.

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `openai` | OpenAI / Azure OpenAI SDK — also used as the unified interface for Ollama and Gemini |
| `anthropic` | Anthropic SDK, wrapped by an adapter in `provider_config.py` |
| `python-dotenv` | Loads credentials from `keys/.env` at runtime |
| `instructor` | Structured extraction via Pydantic models (example 5) |

---

### 4. Configure API Keys

Inside the `keys/` directory, create a file named `.env`:

```bash
mkdir -p keys
touch keys/.env
```

Open `keys/.env` in your editor and populate it. All fields for every provider are shown below — **fill in the values for the provider(s) you plan to use**, and set `LLM_PROVIDER` to activate one of them.

```env
# ── Active provider ─────────────────────────────────────────────────
# Set this to: azure | openai | anthropic | gemini | ollama
LLM_PROVIDER=azure

# ── Azure OpenAI ─────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_MODEL_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# ── OpenAI ───────────────────────────────────────────────────────────
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o

# ── Anthropic ────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL_NAME=claude-sonnet-4-20250514

# ── Google Gemini ────────────────────────────────────────────────────
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.0-flash

# ── Ollama (local) ───────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL_NAME=qwen2.5-coder:7b
```

Immediately protect your secrets:

```bash
echo "keys/.env" >> .gitignore
```

> [!IMPORTANT]
> **Never hard-code API keys in source files.** Always load them from a `.env` file and ensure `.env` is listed in `.gitignore` before your first commit. Exposed keys can incur unexpected charges.

---

## Running the App

All examples are run from the **repository root** (not from inside `examples/`), so that the relative import path to `provider_config.py` resolves correctly.

### Part 1 — Synchronous Examples

```bash
python examples/part_1/1_prompt_anatomy.py
python examples/part_1/2_few_shot_classifier.py
python examples/part_1/3_cot_math_solver.py
python examples/part_1/4_system_prompt_config.py
python examples/part_1/5_invoice_extractor.py
python examples/part_1/6_prompt_debugger.py
```

### Part 2 — Async / Agentic Examples

```bash
python examples/part_2/7_react_agent.py
python examples/part_2/8_self_consistency.py
python examples/part_2/9_research_chain.py
python examples/part_2/10_branching_chain.py
python examples/part_2/11_iterative_refinement.py
python examples/part_2/12_role_based_prompting.py
python examples/part_2/13_meta_prompt_optimizer.py
python examples/part_2/14_tree_of_thoughts.py
python examples/part_2/15_self_critique_loop.py
python examples/part_2/16_constitutional_ai.py
python examples/part_2/17_least_to_most.py
```

A successful run prints the model's output to the terminal. Part 2 scripts additionally print a **token-usage summary** at the end showing prompt tokens, completion tokens, and total tokens consumed.

> [!TIP]
> To switch providers without editing `.env`, you can override `LLM_PROVIDER` inline at the terminal:
> ```bash
> LLM_PROVIDER=ollama python examples/part_1/1_prompt_anatomy.py
> ```

---

## Verifying Your Connection

Before running the full examples, use the relevant snippet below to confirm your credentials work for your chosen provider.

### Azure OpenAI

```python
# test_azure.py
import os
from pathlib import Path
import openai
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / "keys" / ".env")

client = openai.AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

response = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_MODEL_NAME"],
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

```bash
python test_azure.py
```

> **401 Unauthorized** → API key is wrong or has extra whitespace.  
> **404 Resource Not Found** → Deployment name does not match what is configured in Azure Portal.

---

### OpenAI

```python
# test_openai.py
import os
from pathlib import Path
import openai
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / "keys" / ".env")

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model=os.environ["OPENAI_MODEL_NAME"],
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

```bash
python test_openai.py
```

> **401 Unauthorized** → Invalid or revoked API key.  
> **429 Rate Limit** → Your account has hit its quota; check usage at [platform.openai.com/usage](https://platform.openai.com/usage).

---

### Anthropic

```python
# test_anthropic.py
import os
from pathlib import Path
import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / "keys" / ".env")

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

message = client.messages.create(
    model=os.environ["ANTHROPIC_MODEL_NAME"],
    max_tokens=64,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(message.content[0].text)
```

```bash
python test_anthropic.py
```

> **AuthenticationError** → API key is incorrect.  
> **NotFoundError** → Model name is invalid; verify the model name at [docs.anthropic.com/models](https://docs.anthropic.com/en/docs/about-claude/models/all-models).

---

### Google Gemini

```python
# test_gemini.py
import os
from pathlib import Path
import openai
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / "keys" / ".env")

# Gemini exposes an OpenAI-compatible endpoint
client = openai.OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

response = client.chat.completions.create(
    model=os.environ["GEMINI_MODEL_NAME"],
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

```bash
python test_gemini.py
```

> **401 Unauthorized** → API key missing or invalid; re-check the value in `.env`.  
> **404 Not Found** → Model name is wrong; check current model names at [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models).

---

### Ollama

```python
# test_ollama.py
import os
from pathlib import Path
import openai
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / "keys" / ".env")

client = openai.OpenAI(
    api_key="ollama",   # Ollama does not require a real key
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
)

response = client.chat.completions.create(
    model=os.environ["OLLAMA_MODEL_NAME"],
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

```bash
python test_ollama.py
```

> **Connection refused** → Ollama is not running; start it with `ollama serve`.  
> **404 model not found** → The model has not been pulled; run `ollama pull qwen2.5-coder:7b`.

---

## Installing Ollama

Ollama lets you run large language models locally on your machine, with no data sent to an external API.

> [!NOTE]
> Ollama runs best with a dedicated GPU but works on CPU-only machines at reduced speed. Ensure you have at least **8 GB of RAM** and **8 GB of free disk space** before proceeding.

### macOS

Install via Homebrew (recommended):

```bash
brew install ollama
```

Start the Ollama service:

```bash
ollama serve
```

**Alternative — GUI installer:** Download the macOS `.dmg` from [ollama.com/download](https://ollama.com/download), open it, drag Ollama to Applications, and launch it. The service starts automatically.

### Windows

1. Go to [ollama.com/download](https://ollama.com/download) and click **Download for Windows**.
2. Run `OllamaSetup.exe`. No configuration is needed.
3. Ollama starts as a background service automatically and appears in the system tray.

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

The installer registers an `ollama` systemd service that starts automatically on boot.

### Verify Installation

```bash
ollama --version
```

You should see output like `ollama version 0.x.x`. If the command is not found, restart your terminal (or machine) and try again.

### Pull the Workshop Model

```bash
ollama pull qwen2.5-coder:7b
```

This model is approximately **4 GB** — ensure you have a stable connection and sufficient disk space. Once downloaded, start an interactive session to confirm it works:

```bash
ollama run qwen2.5-coder:7b
```

Type a message and press Enter. Type `/bye` to exit.

### Useful Ollama Commands

| Command | Description |
|---------|-------------|
| `ollama list` | List all locally downloaded models |
| `ollama pull qwen2.5-coder:7b` | Download the workshop model (~4 GB) |
| `ollama run qwen2.5-coder:7b` | Start an interactive chat session |
| `ollama rm <model>` | Delete a model and free disk space |
| `ollama serve` | Start the Ollama API server manually if not running |
| `ollama ps` | Show models currently loaded in memory |

---

## LLM Provider Configuration

All provider configuration is managed through environment variables in `keys/.env`. The table below is a complete reference for every supported variable.

| Variable | Provider | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | All | Active provider: `azure`, `openai`, `anthropic`, `gemini`, or `ollama` |
| `AZURE_OPENAI_API_KEY` | Azure | Your Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure | Azure resource endpoint (e.g. `https://your-resource.openai.azure.com/`) |
| `AZURE_OPENAI_MODEL_NAME` | Azure | Deployment name as configured in Azure Portal (e.g. `gpt-4o`) |
| `AZURE_OPENAI_API_VERSION` | Azure | API version string (e.g. `2024-02-15-preview`) |
| `OPENAI_API_KEY` | OpenAI | Your OpenAI secret key |
| `OPENAI_MODEL_NAME` | OpenAI | Model name (e.g. `gpt-4o`, `gpt-4o-mini`) |
| `ANTHROPIC_API_KEY` | Anthropic | Your Anthropic API key |
| `ANTHROPIC_MODEL_NAME` | Anthropic | Model name (e.g. `claude-sonnet-4-20250514`) |
| `GEMINI_API_KEY` | Gemini | Your Google AI Studio API key |
| `GEMINI_MODEL_NAME` | Gemini | Model name (e.g. `gemini-2.0-flash`, `gemini-1.5-pro`) |
| `OLLAMA_BASE_URL` | Ollama | Local API URL (default: `http://localhost:11434/v1`) |
| `OLLAMA_MODEL_NAME` | Ollama | Model name as returned by `ollama list` (e.g. `qwen2.5-coder:7b`) |

### Switching Providers

Only one provider is active at a time. To switch, change the `LLM_PROVIDER` value in `keys/.env` (or set it inline at the terminal) and ensure the matching credentials are populated. You do not need to remove other providers' credentials — unused entries are simply ignored.

### Structured Extraction Mode (Example 5)

`5_invoice_extractor.py` uses the `instructor` library, which requires knowing *how* the provider returns structured data. The mode is selected automatically per provider:

| Provider | Instructor Mode | Reason |
|----------|----------------|--------|
| `azure` / `openai` | `TOOLS` | Native function / tool calling — most reliable |
| `anthropic` | `JSON` | No native tool calling via the adapter |
| `gemini` | `JSON` | OpenAI-compatible JSON mode |
| `ollama` | `MD_JSON` | Markdown-fenced JSON, required for local models |

### Cost Considerations

| Provider | Pricing model | Notes |
|----------|--------------|-------|
| **Azure OpenAI** | Per 1 000 tokens (input + output priced separately) | Pricing depends on your Azure subscription tier and deployment region |
| **OpenAI** | Per 1 000 tokens | See [openai.com/pricing](https://openai.com/pricing); `gpt-4o-mini` is significantly cheaper than `gpt-4o` for experimentation |
| **Anthropic** | Per 1 000 tokens | See [anthropic.com/pricing](https://www.anthropic.com/pricing); Haiku models are the most cost-effective |
| **Google Gemini** | Per 1 000 tokens | Generous free tier on `gemini-2.0-flash`; see [ai.google.dev/pricing](https://ai.google.dev/pricing) |
| **Ollama** | Free (local compute) | No API costs; cost is electricity and hardware. Multi-step Part 2 scripts run many calls — Ollama is ideal for experimentation |

> [!TIP]
> Part 2 scripts print a **token-usage summary** at the end of every run. Use this to track consumption before running expensive patterns (e.g. `8_self_consistency.py` with a high `n_paths` value).

---

## Troubleshooting & Common Issues

### Python & Environment

| Issue | Fix |
|-------|-----|
| `python` command not found | Use `python3` instead. On macOS with Homebrew, `python3` is the correct command. |
| Python version too low (`<3.10`) | Install Python 3.10+ from [python.org](https://www.python.org/downloads/) or via `brew install python@3.12`. |
| VS Code cannot find interpreter | Press `Cmd+Shift+P` (macOS) / `Ctrl+Shift+P` (Windows/Linux), type **Python: Select Interpreter**, and choose the `.venv` path. |
| `brew: command not found` (macOS) | Homebrew was not added to PATH. Run `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` and follow the post-install PATH instructions printed to your terminal. |
| `ModuleNotFoundError` after install | Ensure your virtual environment is activated (`source .venv/bin/activate`) before running pip and Python commands. |

### Azure OpenAI

| Issue | Fix |
|-------|-----|
| **401 Unauthorized** | API key is wrong or has leading/trailing whitespace. Open `.env` and verify the key has no extra characters. |
| **404 Resource Not Found** | The deployment name in `AZURE_OPENAI_MODEL_NAME` does not match the deployment configured in Azure Portal. |
| **429 Too Many Requests** | You have hit your Azure quota. Wait a moment and retry, or request a quota increase in the Azure Portal. |
| **APIConnectionError** | Endpoint URL is malformed. It must end with a `/` and use `https://`. |

### OpenAI

| Issue | Fix |
|-------|-----|
| **401 Unauthorized** | API key is invalid or has been revoked. Generate a new key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). |
| **429 Rate Limit** | Free-tier quota exceeded. Check usage at [platform.openai.com/usage](https://platform.openai.com/usage) or upgrade your plan. |
| **Model not found** | The model name is invalid. Common values: `gpt-4o`, `gpt-4o-mini`. |

### Anthropic

| Issue | Fix |
|-------|-----|
| **AuthenticationError** | API key is incorrect. Verify it at [console.anthropic.com](https://console.anthropic.com). |
| **NotFoundError** | Model name is invalid. Check current model names at [docs.anthropic.com](https://docs.anthropic.com/en/docs/about-claude/models/all-models). |
| **OverloadedError** | Anthropic's API is temporarily overloaded. Wait a few seconds and retry. |

### Google Gemini

| Issue | Fix |
|-------|-----|
| **401 Unauthorized** | API key is missing or invalid. Re-create it at [aistudio.google.com](https://aistudio.google.com/app/apikey). |
| **404 Not Found** | Model name is incorrect. Check [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) for valid names. |
| **quota exceeded** | You have hit the free-tier request limit. Wait for the quota window to reset (typically 1 minute for RPM limits). |

### Ollama

| Issue | Fix |
|-------|-----|
| **Connection refused / Cannot connect** | Ollama is not running. Start it with `ollama serve` in a separate terminal. |
| **404 model not found** | The model has not been pulled. Run `ollama pull qwen2.5-coder:7b`. |
| **Address already in use** | A previous Ollama instance is still running. On macOS: `pkill ollama`. On Windows: kill the process in Task Manager. |
| **Model download interrupted** | Check available disk space (`df -h`). Cancel with `Ctrl+C` and re-run `ollama pull qwen2.5-coder:7b`. |
| **Very slow responses** | Running on CPU without a GPU. This is expected — smaller models (7B) are more practical on CPU-only machines. |

### General Script Issues

| Issue | Fix |
|-------|-----|
| `EnvironmentError: LLM_PROVIDER='...' requires env vars...` | One or more required variables for the active provider are missing from `.env`. Check the [Configuration Reference](#llm-provider-configuration). |
| `ValueError: Unknown LLM_PROVIDER '...'` | `LLM_PROVIDER` is set to an unrecognised value. Valid values: `azure`, `openai`, `anthropic`, `gemini`, `ollama`. |
| Script exits immediately with no output | Check that the virtual environment is active and all dependencies installed: `pip install -r requirements.txt`. |
| Part 2 script hangs indefinitely | An async LLM call may be waiting for a response that will not arrive (e.g. Ollama is not running). Press `Ctrl+C`, verify the provider, and retry. |

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
