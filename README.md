# LLM Deep Dive Workshop

Hands-on code examples from the workshop sessions. Every script is self-contained and runnable from the repository root.

---

## Setup

### 1. Clone & install

```bash
git clone <repository-url>
cd LLMDeepDiveWorkshop
pip install -r requirements.txt
```

### 2. Configure credentials

Copy the template and fill in the values for the provider(s) you want to use:

```bash
cp keys/.env.example keys/.env
# then edit keys/.env
```

`keys/.env` is git-ignored and will never be committed. See [keys/.env.example](keys/.env.example) for the full list of supported variables.

### 3. Run an example

All scripts are run from the **repository root**:

```bash
# Example 1 — ReAct agent
python examples/langchain/react_agent.py

# Example 2 — Voice agent server
python examples/ultravox/voice_agent_server.py
```

---

## Repository Structure

```
LLMDeepDiveWorkshop/
├── README.md
├── LICENSE
├── requirements.txt
├── keys/
│   └── .env.example      # Template — copy to keys/.env and fill in values
│   └── .env              # Your credentials — git-ignored, never committed
├── logs/                 # Runtime logs written here (git-ignored)
└── examples/
    ├── langchain/
    │   └── react_agent.py          # ReAct agent with web search + Wikipedia
    └── ultravox/
        ├── voice_agent_server.py   # Flask server — proxies Ultravox API
        └── nova_demo.html          # Browser UI served by voice_agent_server.py
```

---

## Examples

### `examples/langchain/react_agent.py` — ReAct Agent (LangChain)

> **Key feature: live Thought → Action → Observation loop printed to the terminal.**

Uses LangChain's ReAct agent with two tools — **DuckDuckGo web search** and **Wikipedia** — to answer a multi-part research question. The agent reasons step-by-step and produces a synthesised final answer from live sources.

**Requires:** `LLM_PROVIDER` and its matching API key in `keys/.env`.

**Run:**

```bash
python examples/langchain/react_agent.py
```

**Expected output:** structured logs showing each tool call, elapsed time, and token usage, followed by the agent's final answer. Logs are also written to `logs/react_YYYYMMDD_HHMMSS.log`.

**Customise:** change `QUESTION` at the top of the script, or switch providers by setting `LLM_PROVIDER` in `keys/.env`.

---

### `examples/ultravox/voice_agent_server.py` — Voice Agent Server (Ultravox / Nova)

> **Key feature: API key stays server-side; the browser joins the call via a secure `joinUrl`.**

A lightweight Flask server that proxies call creation to the Ultravox API. The browser never touches the API key — it calls `POST /start-call`, receives a `joinUrl`, and connects via the Ultravox JS SDK. Nova is pre-configured as a BrightStart Coaching receptionist and speaks first.

**Requires:** `ULTRAVOX_API_KEY` in `keys/.env`.

**Run:**

```bash
python examples/ultravox/voice_agent_server.py
```

The server starts on **port 5000** and opens `http://localhost:5000` automatically. `nova_demo.html` must be in the same directory as the server script (it already is). Logs are written to `logs/voice_YYYYMMDD_HHMMSS.log`.

**Environment variables (optional):**

| Variable | Default | Purpose |
|---|---|---|
| `AUTO_OPEN_BROWSER` | `true` | Set to `false` to disable the auto-launch |
| `APP_URL` | `http://127.0.0.1:5000` | URL shown in the startup banner |
| `ULTRAVOX_MODEL` | `ultravox-v0.7` | Ultravox model version to use |

---

## Environment Variables Reference

| Variable | Required by | Notes |
|---|---|---|
| `LLM_PROVIDER` | `react_agent.py` | `openai` \| `azure` \| `anthropic` \| `ollama` |
| `OPENAI_API_KEY` | `react_agent.py` | When `LLM_PROVIDER=openai` |
| `OPENAI_MODEL_NAME` | `react_agent.py` | Default: `gpt-4o` |
| `AZURE_OPENAI_API_KEY` | `react_agent.py` | When `LLM_PROVIDER=azure` |
| `AZURE_OPENAI_ENDPOINT` | `react_agent.py` | When `LLM_PROVIDER=azure` |
| `AZURE_OPENAI_API_VERSION` | `react_agent.py` | Default: `2024-12-01-preview` |
| `AZURE_OPENAI_MODEL_NAME` | `react_agent.py` | Default: `gpt-4o` |
| `ANTHROPIC_API_KEY` | `react_agent.py` | When `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL_NAME` | `react_agent.py` | Default: `claude-3-5-haiku-20241022` |
| `OLLAMA_BASE_URL` | `react_agent.py` | Default: `http://localhost:11434/v1` |
| `OLLAMA_MODEL_NAME` | `react_agent.py` | Default: `llama3.2` |
| `ULTRAVOX_API_KEY` | `voice_agent_server.py` | Required — get one at ultravox.ai |
| `ULTRAVOX_MODEL` | `voice_agent_server.py` | Default: `ultravox-v0.7` |
| `AUTO_OPEN_BROWSER` | `voice_agent_server.py` | Default: `true` |
| `APP_URL` | `voice_agent_server.py` | Default: `http://127.0.0.1:5000` |

---

## Logging

Both scripts write structured logs to the `logs/` directory:

- `logs/react_YYYYMMDD_HHMMSS.log` — every LLM call, tool invocation, elapsed time, and token usage
- `logs/voice_YYYYMMDD_HHMMSS.log` — server startup, call creation requests, Ultravox API responses, and errors

Console output mirrors the log at `INFO` level. Set the handler level in the script to `DEBUG` to see verbose model-level details in the terminal as well.

---

## Troubleshooting

**`ModuleNotFoundError`** — run `pip install -r requirements.txt` from the repo root.

**`OPENAI_API_KEY is not set`** (or any provider key error) — confirm the variable is present in `keys/.env` and matches the `LLM_PROVIDER` value.

**`ULTRAVOX_API_KEY is not set`** — add your Ultravox key to `keys/.env`.

**Port 5000 already in use** — stop the other process or change the port in `voice_agent_server.py` (`app.run(port=…)`).

**Browser does not open automatically** — navigate to `http://localhost:5000` manually, or check that `AUTO_OPEN_BROWSER=true` in `keys/.env`.

**DuckDuckGo / Wikipedia timeouts** — these tools require an internet connection. Retry or check network access.

**Microphone permission blocked in browser** — enable microphone access for `localhost` in your browser's site settings and refresh.

---

## License

[MIT](LICENSE)
