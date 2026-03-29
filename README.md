# LLMDeepDiveWorkshop

# LLM Wrapper: Command-Line Chatbot with Azure OpenAI / OpenAI

A command-line chatbot application that connects to **Azure OpenAI** or **OpenAI** APIs with multi-user session management, pluggable conversation memory strategies, and built-in token usage tracking. It is designed as a teaching example to demonstrate how to structure an AI chat app in Python.

---

## Features

- **Multiple AI providers** — choose between Azure OpenAI and OpenAI at login.
- **Three conversation memory strategies:**
  | Strategy | Behaviour |
  |---|---|
  | **Simple** | Keeps the full chat history — nothing is ever removed. |
  | **Fixed Window** | Keeps only the last *N* messages; older ones are dropped. |
  | **Summarizing** | Asks the AI to compress older messages into a summary to save context space. |
- **Multi-user support** — switch between users without restarting the app; each user only sees their own sessions.
- **Session management** — create new sessions, resume previous ones, and view per-session message counts.
- **Token usage tracking** — view token consumption per session or per user from an admin overview menu.
- **Rotating log file** — all activity is logged to `logs/main_system_logs.log` (rotates daily, 30-day retention).

---

## Project Structure

```
CodebaseExamples/
├── app.py                        # CLI entry point — menus, login flow, chat loop
├── config.py                     # Settings, credentials, logger setup
├── requirements.txt              # Python dependencies
├── keys/
│   └── .env                      # API keys and model settings (not committed)
├── logs/
│   └── main_system_logs.log      # Auto-generated log file
└── modules/
    ├── __init__.py               # Package docstring
    ├── AIManagers.py             # Azure OpenAI / OpenAI client wrappers
    ├── ChatOrchestrator.py       # Central controller that ties everything together
    ├── ConversationManagers.py   # Simple, Fixed-Window, and Summarizing managers
    └── TokenTracker.py           # Per-session and per-user token accounting
```

---

## How It Works

```
┌──────────┐      ┌──────────────────┐      ┌──────────────┐
│  app.py  │─────▶│ ChatOrchestrator │─────▶│  AIManager   │──▶ OpenAI API
│  (CLI)   │      │   (controller)   │      │ (Azure/std)  │
└──────────┘      └──────────────────┘      └──────────────┘
                         │       │
                         ▼       ▼
              ┌────────────┐  ┌──────────────┐
              │Conversation│  │ TokenTracker │
              │ Manager    │  │              │
              └────────────┘  └──────────────┘
```

1. **`app.py`** presents the user interface (login → user menu → chat → admin menu).
2. **`ChatOrchestrator`** is the central controller. It receives a user message, passes it through the conversation manager, calls the AI, stores the reply, and records token usage.
3. **`AIManagers`** wrap the OpenAI SDK to send chat completion requests and return a `ChatResult`.
4. **`ConversationManagers`** decide which messages to include in each API call based on the chosen strategy.
5. **`TokenTracker`** keeps running totals of prompt/completion/total tokens per session and per user.

---

## Setup

### Prerequisites

- **Python 3.10+**
- An **Azure OpenAI** or **OpenAI** API key

### 1. Clone the repository

```bash
git clone <repo-url>
cd CodebaseExamples
git checkout llm-wrapper
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create (or edit) the file `keys/.env` with your credentials:

```dotenv
# Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_MODEL_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Standard OpenAI
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL_NAME=gpt-4o
```

> **Note:** Only fill in the section for the provider you plan to use.

---

## Running the App

```bash
python app.py
```

You will be guided through the following steps:

1. **Enter a username** — any string; used to group your sessions.
2. **Choose an AI provider** — Azure OpenAI or OpenAI.
3. **Choose a memory strategy** — Simple, Fixed Window, or Summarizing.
4. **User menu** — create sessions, resume, view token usage, or open the admin menu.
5. Inside a session, type a message and press Enter to chat with the AI.

---

## Verifying Everything Works

1. **Quick smoke test** — run the app and create a new session. Send a simple message like `Hello!`. If you receive a reply, the API connection and basic flow are working.

2. **Check conversation memory** — inside a session, select *View conversation history* (option 2) to confirm messages are being stored.

3. **Check token tracking** — select *Token usage for this session* (option 3) to verify that token counts are being recorded after each exchange.

4. **Multi-user test** — go back to the user menu and choose *Switch user*. Log in as a different user, create a session, and send a message. Then open *Main menu (admin view)* → *Token usage per user* to see both users listed.

5. **Logs** — inspect `logs/main_system_logs.log` to confirm that login events, AI calls, and token recordings are being logged.

---

## Configuration Reference

All tuneable settings live in `config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `MAX_COMPLETION_TOKENS` | `2048` | Max tokens the model can return per response |
| `CONVERSATION_MANAGER` | `"simple"` | Default memory strategy (`simple`, `fixed_window`, `summarizing`) |
| `MAX_MESSAGES` | `20` | Messages to keep in fixed-window mode |
| `WINDOW_SIZE` | `10` | Message count that triggers summarisation |
| `SUMMARIZE_COUNT` | `5` | How many old messages to summarise at once |
| `SYSTEM_PROMPT` | *"You are a helpful…"* | Instruction given to the AI at conversation start |
| `LOG_LEVEL` | `"INFO"` | Python logging level |
| `LOG_FILE` | `"logs/main_system_logs.log"` | Path to the log file |

---

## License

This project is licensed under the [MIT License](LICENSE).