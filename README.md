# LLM Deep Dive Workshop

# AI Agents & Voice

A hands-on collection of AI agent examples that demonstrates the **ReAct (Reason + Act)** pattern using LangChain and a **real-time voice agent** powered by **Ultravox**. Supports **Azure OpenAI**, **OpenAI**, **Anthropic**, and a locally running **Ollama** instance for the LangChain example. Each script is self-contained, runnable from the repository root, and designed as a teaching resource.

---

> [!WARNING]
> **Testing Status:** This codebase has been fully tested with **Azure OpenAI** and **Ollama**. Support for **OpenAI** and **Anthropic** is implemented and should work, but has **not been independently verified**. If you encounter issues with those providers, check that credentials are correctly set in `keys/.env`.

---

## Table of Contents

- [Examples](#examples)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Verifying Your Connection](#verifying-your-connection)
- [Installing Ollama](#installing-ollama)
- [LLM Provider Configuration](#llm-provider-configuration)
- [Logging](#logging)
- [Troubleshooting & Common Issues](#troubleshooting--common-issues)
- [License](#license)

---

## Examples

This repository contains **two examples** covering agentic reasoning and real-time voice interaction.

---

### `examples/langchain/react_agent.py` — ReAct Agent (LangChain)

> **Key feature: live Thought → Action → Observation loop printed to the terminal.**

Uses LangChain's ReAct agent with two tools — **DuckDuckGo web search** and **Wikipedia** — to answer a multi-part research question. The agent reasons step-by-step and produces a synthesised final answer from live sources.

The agent is driven by a configurable `QUESTION` variable at the top of the script. The default question is:

> *"Who founded OpenAI and what are each of the founders doing now?"*

**How the ReAct loop works:**

| Step | What happens |
|------|-------------|
| **Think** | The LLM reasons about what it knows and what it needs to find out |
| **Act** | The agent selects and calls a tool (`web_search` or `wikipedia`) |
| **Observe** | The tool result is returned to the agent as context |
| **Repeat** | The loop continues until the agent has enough information to answer |
| **Finish** | The agent synthesises a final answer from all gathered observations |

**Run it to see** each tool call logged live, including elapsed time per tool and a final synthesised answer.

---

### `examples/ultravox/voice_agent_server.py` — Voice Agent Server (Ultravox / Nova)

A lightweight Flask server that proxies call creation to the Ultravox API. When the browser calls `POST /start-call`, the server authenticates with `ULTRAVOX_API_KEY` server-side and returns a `joinUrl`. The browser connects to the voice session via the Ultravox JS SDK using only that URL — the key is never exposed to the client.

Nova is pre-configured as a **BrightStart Coaching** receptionist and speaks first. The agent configuration includes:

| Setting | Value | Purpose |
|---------|-------|---------|
| Voice | Jessica | Ultravox voice ID |
| Temperature | 0.4 | Consistent but natural responses |
| Max duration | 300 s | Hard 5-minute call cap |
| Join timeout | 5 s | Abandons call if browser doesn't connect |
| First speaker | Agent (uninterruptible) | Nova greets the caller before they can interrupt |

**Run it to see** a voice agent answer questions about BrightStart Coaching, offer to book a free intro call, and respond naturally to follow-up questions — all in real time from the browser.

---

## Project Structure

```
LLMDeepDiveWorkshop/
├── README.md                  # You are here
├── LICENSE                    # MIT licence
├── requirements.txt           # Python dependencies
├── keys/
│   ├── .env.example           # Template — copy to keys/.env and fill in values
│   └── .env                   # Your credentials (not committed)
├── logs/                      # Runtime logs written here (git-ignored)
└── examples/
    ├── langchain/
    │   └── react_agent.py     # ReAct agent with DuckDuckGo + Wikipedia
    └── ultravox/
        ├── voice_agent_server.py   # Flask server — proxies Ultravox API
        └── nova_demo.html          # Browser UI served by voice_agent_server.py
```

---

## How It Works

### ReAct Agent Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         keys/.env                                   │
│  LLM_PROVIDER=azure   AZURE_OPENAI_API_KEY=...   AZURE_...          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  python-dotenv loads at import time
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    react_agent.py                                   │
│                                                                     │
│  build_llm()  →  LangChain chat model (provider-specific)           │
│                                                                     │
│  ├─ openai    → ChatOpenAI(...)                                     │
│  ├─ azure     → AzureChatOpenAI(...)                                │
│  ├─ anthropic → ChatAnthropic(...)                                  │
│  └─ ollama    → ChatOpenAI(base_url="localhost:11434/v1", ...)      │
│                                                                     │
│  Tools: web_search (DuckDuckGo) + wikipedia                         │
│  Agent: LangGraph ReAct — Think → Act → Observe loop                │
│  Logging: AgentLogger callback → console + logs/react_*.log         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │   LLM Provider   │
                    │  (Azure / OAI /  │
                    │  Anthropic /     │
                    │  Ollama)         │
                    └──────────────────┘
```

### Voice Agent Flow

```
┌──────────────┐    POST /start-call    ┌───────────────────────┐
│   Browser    │ ──────────────────────▶│  voice_agent_server   │
│ nova_demo.   │                        │  .py (Flask)          │
│ html         │                        │                       │
│              │ ◀── { joinUrl } ───────│  forwards AGENT_CONFIG│
│              │                        │  to Ultravox API      │
│              │                        │  using ULTRAVOX_API_  │
│              │                        │  KEY (server-side)    │
│              │                        └───────────────────────┘
│              │   WebSocket (joinUrl)   ┌──────────────────────┐
│              │ ◀──────────────────────▶│   Ultravox API       │
│              │    real-time audio      │   (Nova voice agent) │
└──────────────┘                         └──────────────────────┘
```

**How the system operates:**

1. **Single `.env` file, one variable to switch LLM providers.** Setting `LLM_PROVIDER=ollama` vs `LLM_PROVIDER=azure` redirects all traffic in the ReAct agent. Credentials for each provider are namespaced so multiple providers can coexist in `.env`.

2. **Provider-specific LangChain models.** Each provider is instantiated with its native LangChain chat model class. Ollama is accessed via its OpenAI-compatible local endpoint, so `ChatOpenAI` works with `base_url` pointing to localhost.

3. **API key never reaches the browser.** The Flask server holds `ULTRAVOX_API_KEY` and only returns a short-lived `joinUrl` to the browser. The browser connects directly to Ultravox over WebSocket using that URL.

4. **Structured logging to file and console.** Both scripts write timestamped log files to `logs/`. The `AgentLogger` callback in `react_agent.py` logs every LLM call, tool invocation, elapsed time, and token usage.

---

## Setup & Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11 or higher** — required for the `match`/`case` syntax and `asyncio` improvements used in the codebase. Check with:

  ```bash
  python3 --version
  ```

- **An API key for at least one provider:**

  | Provider | Where to obtain |
  |----------|----------------|
  | Azure OpenAI | From your workshop organiser or the [Azure Portal](https://portal.azure.com) |
  | OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
  | Anthropic | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
  | Ollama | No key needed — runs entirely locally. See [Installing Ollama](#installing-ollama) |
  | Ultravox | Required for the voice agent only — get one at [ultravox.ai](https://www.ultravox.ai) |

---

### 1. Clone the Repository

```bash
git clone <repository-url>
cd LLMDeepDiveWorkshop
git checkout Session-3
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

---

### 4. Configure API Keys

Copy the template and fill in your credentials:

```bash
cp keys/.env.example keys/.env
```

Open `keys/.env` in your editor and populate it. Set `LLM_PROVIDER` to activate the provider you want to use for the ReAct agent.

```env
# ── Active provider ─────────────────────────────────────────────────
# Set this to: azure | openai | anthropic | ollama
LLM_PROVIDER=azure

# ── Azure OpenAI ─────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_MODEL_NAME=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# ── OpenAI ───────────────────────────────────────────────────────────
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o

# ── Anthropic ────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL_NAME=claude-3-5-haiku-20241022

# ── Ollama (local — no API key required) ─────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL_NAME=qwen3:8b

# ── Ultravox (voice agent only) ──────────────────────────────────────
ULTRAVOX_API_KEY=your_ultravox_api_key_here
```

`keys/.env` is git-ignored and will never be committed.

---

## Running the App

All examples are run from the **repository root** (not from inside `examples/`), so that the relative path to `keys/.env` resolves correctly.

### ReAct Agent (LangChain)

```bash
python examples/langchain/react_agent.py
```

**Expected output:** structured logs showing each tool call, elapsed time per tool, and a final synthesised answer. Logs are also written to `logs/react_YYYYMMDD_HHMMSS.log`.

**Customise:** change `QUESTION` at the top of the script, or switch providers by setting `LLM_PROVIDER` in `keys/.env`.

---

### Voice Agent Server (Ultravox / Nova)

```bash
python examples/ultravox/voice_agent_server.py
```

The server starts on **port 5000** and opens `http://localhost:5000` automatically in Chrome (or your default browser). `nova_demo.html` must be in the same directory as the server script — it already is.

**Optional environment variables:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUTO_OPEN_BROWSER` | `true` | Set to `false` to disable auto-launch |
| `APP_URL` | `http://127.0.0.1:5000` | URL shown in the startup banner |
| `ULTRAVOX_MODEL` | `ultravox-v0.7` | Ultravox model version to use |

Logs are written to `logs/voice_YYYYMMDD_HHMMSS.log`.

---

## Verifying Your Connection

Before running the full examples, use the relevant snippet below to confirm your credentials work.

### Azure OpenAI

```python
# test_azure.py
import os
from pathlib import Path
from dotenv import load_dotenv
import openai

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
from dotenv import load_dotenv
import openai

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
from dotenv import load_dotenv
import anthropic

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
> **NotFoundError** → Model name is invalid; verify at [docs.anthropic.com](https://docs.anthropic.com/en/docs/about-claude/models/all-models).

---

### Ollama

```python
# test_ollama.py
import os
from pathlib import Path
from dotenv import load_dotenv
import openai

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
> **404 model not found** → The model has not been pulled; run `ollama pull qwen3:8b`.

---

### Ultravox

Test the Ultravox API key directly:

```python
# test_ultravox.py
import os
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parent / "keys" / ".env")

api_key = os.environ["ULTRAVOX_API_KEY"]
response = requests.get(
    "https://api.ultravox.ai/api/voices",
    headers={"X-API-Key": api_key},
    timeout=10,
)
print(response.status_code, response.json())
```

```bash
python test_ultravox.py
```

> **401 Unauthorized** → API key is missing or invalid.  
> **200 + list of voices** → Connection is working.

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

You should see output like `ollama version 0.x.x`. If the command is not found, restart your terminal and try again.

### Pull the Workshop Model

```bash
ollama pull qwen3:8b
```

This model is approximately **5 GB** — ensure you have a stable connection and sufficient disk space. Once downloaded, start an interactive session to confirm it works:

```bash
ollama run qwen3:8b
```

Type a message and press Enter. Type `/bye` to exit.

### Useful Ollama Commands

| Command | Description |
|---------|-------------|
| `ollama list` | List all locally downloaded models |
| `ollama pull qwen3:8b` | Download the workshop model (~5 GB) |
| `ollama run qwen3:8b` | Start an interactive chat session |
| `ollama rm <model>` | Delete a model and free disk space |
| `ollama serve` | Start the Ollama API server manually if not running |
| `ollama ps` | Show models currently loaded in memory |

---

## LLM Provider Configuration

All provider configuration is managed through environment variables in `keys/.env`. The table below is a complete reference for every supported variable.

| Variable | Provider | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | All | Active provider: `azure`, `openai`, `anthropic`, or `ollama` |
| `AZURE_OPENAI_API_KEY` | Azure | Your Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure | Azure resource endpoint (e.g. `https://your-resource.openai.azure.com/`) |
| `AZURE_OPENAI_MODEL_NAME` | Azure | Deployment name as configured in Azure Portal (e.g. `gpt-4.1-mini`) |
| `AZURE_OPENAI_API_VERSION` | Azure | API version string (e.g. `2024-12-01-preview`) |
| `OPENAI_API_KEY` | OpenAI | Your OpenAI secret key |
| `OPENAI_MODEL_NAME` | OpenAI | Model name (e.g. `gpt-4o`, `gpt-4o-mini`) |
| `ANTHROPIC_API_KEY` | Anthropic | Your Anthropic API key |
| `ANTHROPIC_MODEL_NAME` | Anthropic | Model name (e.g. `claude-3-5-haiku-20241022`) |
| `OLLAMA_BASE_URL` | Ollama | Local API URL (default: `http://localhost:11434/v1`) |
| `OLLAMA_MODEL_NAME` | Ollama | Model name as returned by `ollama list` (e.g. `qwen3:8b`) |
| `ULTRAVOX_API_KEY` | Ultravox | Your Ultravox API key (voice agent only) |
| `ULTRAVOX_MODEL` | Ultravox | Ultravox model version (default: `ultravox-v0.7`) |
| `AUTO_OPEN_BROWSER` | Ultravox | Auto-launch browser on server start (default: `true`) |
| `APP_URL` | Ultravox | URL shown in startup banner (default: `http://127.0.0.1:5000`) |

### Switching Providers

Only one LLM provider is active at a time for the ReAct agent. To switch, change the `LLM_PROVIDER` value in `keys/.env`. You do not need to remove other providers' credentials — unused entries are simply ignored.

### Cost Considerations

| Provider | Pricing model | Notes |
|----------|--------------|-------|
| **Azure OpenAI** | Per 1 000 tokens (input + output priced separately) | Pricing depends on your Azure subscription tier and region |
| **OpenAI** | Per 1 000 tokens | See [openai.com/pricing](https://openai.com/pricing); `gpt-4.1-mini` is significantly cheaper for experimentation |
| **Anthropic** | Per 1 000 tokens | See [anthropic.com/pricing](https://www.anthropic.com/pricing); Haiku models are most cost-effective |
| **Ollama** | Free (local compute) | No API costs; cost is electricity and hardware |
| **Ultravox** | Per minute of voice call | See [ultravox.ai](https://www.ultravox.ai) for current pricing |

---

## Logging

Both scripts write structured logs to the `logs/` directory:

- `logs/react_YYYYMMDD_HHMMSS.log` — every LLM call, tool invocation, elapsed time, and token usage
- `logs/voice_YYYYMMDD_HHMMSS.log` — server startup, call creation requests, Ultravox API responses, and errors

Console output mirrors the log at `INFO` level. The file handler captures `DEBUG`-level detail (including full request/response shapes) that is not printed to the console.

---

## Troubleshooting & Common Issues

### Python & Environment

| Issue | Fix |
|-------|-----|
| `python` command not found | Use `python3` instead. On macOS with Homebrew, `python3` is the correct command. |
| Python version too low (`<3.10`) | Install Python 3.10+ from [python.org](https://www.python.org/downloads/) or via `brew install python@3.12`. |
| `ModuleNotFoundError` after install | Ensure your virtual environment is activated (`source .venv/bin/activate`) before running pip and Python commands. |
| `brew: command not found` (macOS) | Run `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` and follow the post-install PATH instructions. |

### Azure OpenAI

| Issue | Fix |
|-------|-----|
| **401 Unauthorized** | API key is wrong or has leading/trailing whitespace. Verify the key in `.env`. |
| **404 Resource Not Found** | The deployment name in `AZURE_OPENAI_MODEL_NAME` does not match the deployment in Azure Portal. |
| **429 Too Many Requests** | You have hit your Azure quota. Wait and retry, or request a quota increase in the Azure Portal. |
| **APIConnectionError** | Endpoint URL is malformed. It must end with `/` and use `https://`. |

### OpenAI

| Issue | Fix |
|-------|-----|
| **401 Unauthorized** | API key is invalid or revoked. Generate a new key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). |
| **429 Rate Limit** | Free-tier quota exceeded. Check usage at [platform.openai.com/usage](https://platform.openai.com/usage). |
| **Model not found** | Invalid model name. Common values: `gpt-4o`, `gpt-4o-mini`. |

### Anthropic

| Issue | Fix |
|-------|-----|
| **AuthenticationError** | API key is incorrect. Verify it at [console.anthropic.com](https://console.anthropic.com). |
| **NotFoundError** | Model name is invalid. Check current names at [docs.anthropic.com](https://docs.anthropic.com/en/docs/about-claude/models/all-models). |
| **OverloadedError** | Anthropic's API is temporarily overloaded. Wait a few seconds and retry. |

### Ollama

| Issue | Fix |
|-------|-----|
| **Connection refused** | Ollama is not running. Start it with `ollama serve` in a separate terminal. |
| **404 model not found** | The model has not been pulled. Run `ollama pull qwen3:8b`. |
| **Address already in use** | A previous Ollama instance is still running. On macOS: `pkill ollama`. On Windows: kill the process in Task Manager. |
| **Very slow responses** | Running on CPU without a GPU. This is expected — 8B models are more practical on CPU-only machines. |
| **Agent answers without using tools** | The model does not correctly implement the tool-calling protocol. Use `qwen3:8b` or `gemma4` — Some other models outputs tool calls as plain text and is not compatible with LangChain's agent. |

### Ultravox / Voice Agent

| Issue | Fix |
|-------|-----|
| **`ULTRAVOX_API_KEY is not set`** | Add your Ultravox key to `keys/.env`. |
| **401 Unauthorized** | API key is wrong or revoked. Check your Ultravox account. |
| **429 Rate Limit** | Too many calls in a short period. Wait and retry. |
| **Port 5000 already in use** | Stop the other process or change the port in `voice_agent_server.py` (`app.run(port=…)`). |
| **Browser does not open automatically** | Navigate to `http://localhost:5000` manually, or check that `AUTO_OPEN_BROWSER=true` in `keys/.env`. |
| **Microphone permission blocked** | Enable microphone access for `localhost` in your browser's site settings and refresh. |

### ReAct Agent

| Issue | Fix |
|-------|-----|
| `Unknown LLM_PROVIDER='...'` | `LLM_PROVIDER` is set to an unrecognised value. Valid values: `azure`, `openai`, `anthropic`, `ollama`. |
| **DuckDuckGo / Wikipedia timeouts** | These tools require an internet connection. Retry or check network access. |
| **Agent loops without answering** | The model may be struggling with the question. Try a simpler `QUESTION` or switch to a more capable provider. |

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
