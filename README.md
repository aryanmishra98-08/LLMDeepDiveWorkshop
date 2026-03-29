# LLMDeepDiveWorkshop

# Prompt Engineering Patterns

A hands-on collection of prompt-engineering patterns and advanced LLM techniques built with **Azure OpenAI** and **Ollama**. Each numbered script is a self-contained example that you can run, read, and adapt.

---

## Prompt Examples

This repository walks through **17 progressive examples** split across two parts:

| Part | Examples | Focus |
|------|----------|-------|
| **Part 1** (1–6) | Foundational prompt patterns | Prompt anatomy, few-shot learning, chain-of-thought, system-prompt configuration, structured extraction, prompt debugging |
| **Part 2** (7–17) | Advanced & agentic patterns | ReAct agents, self-consistency, research chains, branching, iterative refinement, role-based prompting, meta-prompt optimisation, tree-of-thoughts, self-critique, constitutional AI, least-to-most decomposition |

---

## Project Structure

```
CodebaseExamples/
├── README.md                  # You are here
├── LICENSE                    # MIT licence
├── requirements.txt           # Python dependencies
├── keys/
│   └── .env                   # Your API credentials (not committed)
└── examples/
    ├── part_1/                # Foundational prompt patterns (1-6)
    │   ├── shared_config.py   # Sync Azure OpenAI client & env loading
    │   ├── 1_prompt_anatomy.py
    │   ├── 2_few_shot_classifier.py
    │   ├── 3_cot_math_solver.py
    │   ├── 4_system_prompt_config.py
    │   ├── 5_invoice_extractor.py
    │   └── 6_prompt_debugger.py
    └── part_2/                # Advanced & agentic patterns (7-17)
        ├── shared_config.py   # Async Azure OpenAI client & TokenTracker
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

### How It Works

Each `shared_config.py` loads credentials from `keys/.env`, creates an OpenAI client, and exposes it to every script in its folder. Part 1 uses a **synchronous** client; Part 2 uses an **asynchronous** client with a built-in `TokenTracker` so you can monitor cost and usage.

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone <repo-url>
cd CodebaseExamples
git checkout prompt-examples
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` installs:

- `openai` — Azure OpenAI SDK
- `ollama` — Local model client
- `python-dotenv` — `.env` file loading

---

## Creating the `.env` File

Inside the `keys/` directory, create a file named `.env`:

```
keys/.env
```

Add your credentials (replace each placeholder with the value you received):

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_MODEL_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-15-preview
OLLAMA_MODEL_NAME=qwen2.5-coder:7b
```

Immediately add `.env` to your `.gitignore`:

```bash
echo ".env" >> .gitignore
```

> **IMPORTANT:** Never hard-code API keys in source files. Always load them from a `.env` file and add `.env` to `.gitignore` immediately.

---

## Running & Verifying

Run any example from the repository root:

```bash
# Part 1 — synchronous examples
python examples/part_1/1_prompt_anatomy.py

# Part 2 — async / agentic examples
python examples/part_2/7_react_agent.py
```

A successful run prints model output to the terminal. Part 2 scripts also print a token-usage summary at the end.

---

## Configuring Azure OpenAI

Azure OpenAI provides access to GPT models through a secure, enterprise-grade endpoint. You will receive credentials from the workshop organiser.

> **IMPORTANT:** Never hard-code API keys in source files. Always load them from a `.env` file and add `.env` to `.gitignore` immediately.

### Credentials You Will Receive

| Credential | Example |
|------------|---------|
| **API Key** | A 32-character hex string |
| **Endpoint URL** | `https://<your-resource>.openai.azure.com/` |
| **Deployment Name** | `gpt-4o` or `gpt-35-turbo` |
| **API Version** | `2024-02-15-preview` |

### Store Credentials in the `.env` File

In the `keys/` directory, create a file named `.env` and add:

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_MODEL_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### Verify the Connection

Create a file called `test_azure.py` in the project root with the following content:

```python
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

# Load credentials from keys/.env
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

Run the test:

```bash
python test_azure.py
```

A short reply from the model confirms that your key, endpoint, and deployment name are all correct.

> **TIP:** If you see a **401 Unauthorized** error, re-check the API key for extra whitespace. A **404 Resource Not Found** error means the deployment name does not match what was configured in Azure Portal.

---

## Installing Ollama

Ollama lets you run large language models locally on your machine. We will use it to run open-source models without sending data to an external API.

> **NOTE:** Ollama runs best with a dedicated GPU but works on CPU-only machines at reduced speed. Ensure you have at least **8 GB of RAM** and **8 GB of free disk space** before proceeding.

### Download & Install

#### macOS

Install via Homebrew (recommended):

```bash
brew install ollama
```

Then start the Ollama service:

```bash
ollama serve
```

**Alternative — GUI installer:** Go to <https://ollama.com/download>, download the macOS `.dmg`, open it, drag Ollama to Applications, and launch it.

#### Windows

1. Go to <https://ollama.com/download>
2. Click **Download for Windows**.
3. Run `OllamaSetup.exe`. No configuration needed.
4. Ollama starts as a background service automatically.
5. The Ollama icon will appear in the system tray.

### Verify Installation

```bash
ollama --version
```

You should see output such as `ollama version 0.x.x`. If the command is not found, restart your machine and try again.

### Pull the Workshop Model

We will use `qwen2.5-coder:7b` throughout the workshop. This model is approximately **4 GB** — ensure you have a stable internet connection and sufficient disk space before pulling.

```bash
ollama pull qwen2.5-coder:7b
```

Once complete, start an interactive session to confirm it works:

```bash
ollama run qwen2.5-coder:7b
```

Type a message and press Enter to chat. Type `/bye` to exit the session.

### Useful Ollama Commands

| Command | Description |
|---------|-------------|
| `ollama list` | List all locally downloaded models |
| `ollama pull qwen2.5-coder:7b` | Download the workshop model (~4 GB) |
| `ollama run qwen2.5-coder:7b` | Start an interactive session with the model |
| `ollama rm <model>` | Delete a model to free disk space |
| `ollama serve` | Start the Ollama API server manually if not running |
| `ollama ps` | Show models currently loaded in memory |

---

## Troubleshooting & Common Issues

| Issue | Fix |
|-------|-----|
| `python` not recognised | Restart terminal after install. On macOS with Homebrew, use `python3`. |
| VS Code cannot find interpreter | Press `Ctrl+Shift+P` / `Cmd+Shift+P`, type **Select Interpreter**, and choose the `.venv` path. |
| `brew: command not found` (macOS) | Homebrew was not added to PATH after install. Re-run the install script or follow the post-install instructions printed in the terminal. |
| Azure **401 Unauthorized** | Re-check the API key. Ensure there are no extra spaces or newlines in your `.env` file. |
| Azure **404 Resource Not Found** | Verify the deployment name matches exactly what was configured in Azure Portal. |
| Ollama: **address already in use** | A previous instance is still running. On macOS: `pkill ollama`. On Windows: stop the service in Task Manager. |
| Ollama model download interrupted | Check disk space (need ~4 GB free). Cancel with `Ctrl+C` and re-run: `ollama pull qwen2.5-coder:7b`. |

---

## Configuration Reference

All configuration is handled through environment variables in `keys/.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_API_KEY` | Yes | Your Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure resource endpoint URL |
| `AZURE_OPENAI_MODEL_NAME` | Yes | Deployment / model name (e.g. `gpt-4o`) |
| `AZURE_OPENAI_API_VERSION` | Yes | API version string (e.g. `2024-02-15-preview`) |
| `OLLAMA_MODEL_NAME` | No | Local Ollama model name (default: `qwen2.5-coder:7b`) |

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
