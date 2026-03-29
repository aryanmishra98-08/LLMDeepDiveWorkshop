# LLMDeepDiveWorkshop

# CLI Chatbot Wrapper: Command-Line Chatbot with Azure OpenAI, OpenAI, Anthropic, Gemini & Ollama

A command-line chatbot application that connects to **Azure OpenAI**, **OpenAI**, **Anthropic**, **Google Gemini**, or a locally running **Ollama** instance, with multi-user session management, pluggable conversation memory strategies, built-in token usage tracking, and a unified provider interface. It is designed as a teaching resource to demonstrate how to structure an AI chat application in Python.

---

> [!WARNING]
> **Testing Status:** This codebase has been fully tested with **Ollama** and **Azure OpenAI** only. Support for **OpenAI**, **Anthropic**, and **Google Gemini** is implemented and should work, but has **not been independently verified**. If you encounter issues with those providers, check that credentials are correctly set and the provider adapter is behaving as expected.

---

## Table of Contents

* [Features](#features)
* [Project Structure](#project-structure)
* [How It Works](#how-it-works)
* [Setup & Installation](#setup--installation)
* [Running the App](#running-the-app)
* [Verifying Your Setup](#verifying-your-setup)
* [Configuration Reference](#configuration-reference)
* [License](#license)

---

## Features

This application demonstrates a production-style CLI chatbot with the following capabilities:

* **Multiple LLM providers** — switch between Azure OpenAI, OpenAI, Anthropic, Gemini, and Ollama
* **Unified provider interface** — all providers expose a consistent request/response structure
* **Multi-user support** — each user has isolated sessions
* **Session management** — create, resume, and inspect chat sessions
* **Conversation memory strategies:**

| Strategy         | Behaviour                                |
| ---------------- | ---------------------------------------- |
| **Simple**       | Retains the entire conversation history  |
| **Fixed Window** | Keeps only the most recent *N* messages  |
| **Summarizing**  | Compresses older messages into summaries |

* **Token usage tracking** — per-session and per-user accounting
* **Rotating logs** — system activity logged with retention

---

## Project Structure

```
CodebaseExamples/
├── app.py                        # CLI entry point — menus, login flow, chat loop
├── config.py                     # Settings, credentials, logger setup
├── requirements.txt              # Python dependencies
├── keys/
│   └── .env                      # API credentials (not committed)
├── logs/
│   └── main_system_logs.log      # Runtime logs (auto-generated)
└── modules/
    ├── AIManagers.py             # Provider wrappers (Azure/OpenAI/Anthropic/Gemini/Ollama)
    ├── ChatOrchestrator.py       # Core controller
    ├── ConversationManagers.py   # Memory strategies
    └── TokenTracker.py           # Token accounting
```

---

## How It Works

The system follows a layered architecture:

```
┌──────────┐      ┌──────────────────┐      ┌──────────────────────┐
│  app.py  │─────▶│ ChatOrchestrator │─────▶│      AIManager       │──▶ LLM API
│  (CLI)   │      │   (controller)   │      │ (Azure/OpenAI/       │
└──────────┘      └──────────────────┘      │  Anthropic/Gemini/   │
                                            │  Ollama)             │
                                            └──────────────────────┘
                         │       │
                         ▼       ▼
              ┌────────────┐  ┌──────────────┐
              │Conversation│  │ TokenTracker │
              │ Manager    │  │              │
              └────────────┘  └──────────────┘
```

### System Flow

1. **CLI Layer (`app.py`)**

   * Handles user interaction: login, menus, chat loop

2. **Orchestration Layer (`ChatOrchestrator`)**

   * Coordinates:

     * Conversation state
     * LLM calls
     * Token tracking

3. **Provider Layer (`AIManagers`)**

   * Wraps different SDKs
   * Normalizes all providers into a common interface

4. **Memory Layer (`ConversationManagers`)**

   * Controls context window behavior
   * Applies selected memory strategy

5. **Tracking Layer (`TokenTracker`)**

   * Records prompt/completion usage
   * Aggregates per session and per user

---

## Setup & Installation

### Prerequisites

* **Python 3.10 or higher**
* API key for at least one provider:

  * Azure OpenAI
  * OpenAI
  * Anthropic
  * Google Gemini
  * OR a local **Ollama** instance

---

### 1. Clone the Repository

```bash
git clone https://github.com/aryanmishra98-08/LLMDeepDiveWorkshop.git
cd LLMDeepDiveWorkshop
git checkout Session-1
```

---

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure API Keys

Create `keys/.env`:

```bash
mkdir -p keys
touch keys/.env
```

Populate it:

```env
# ── Azure OpenAI ────────────────────────────────
AZURE_OPENAI_API_KEY=your-azure-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_MODEL_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# ── OpenAI ──────────────────────────────────────
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL_NAME=gpt-4o

# ── Anthropic ───────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL_NAME=claude-3-5-sonnet-20241022

# ── Gemini ──────────────────────────────────────
GEMINI_API_KEY=your-gemini-key-here
GEMINI_MODEL_NAME=gemini-2.0-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

# ── Ollama ──────────────────────────────────────
OLLAMA_MODEL_NAME=llama3.2
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Immediately protect your secrets:

```bash
echo "keys/.env" >> .gitignore
```

> [!IMPORTANT]
> **Never hard-code API keys in source files.** Always load them from a `.env` file and ensure `.env` is listed in `.gitignore` before your first commit. Exposed keys can incur unexpected charges.

---

## Running the App

```bash
python app.py
```

### Runtime Flow

1. Enter username
2. Select provider
3. Select memory strategy
4. Create or resume session
5. Start chatting

---

## Verifying Your Setup

Use this checklist:

* **Basic response test**
  Send: `Hello` → expect a response

* **Memory validation**
  Check conversation history persists

* **Token tracking**
  View session usage after messages

* **Multi-user isolation**
  Switch users and verify separate sessions

* **Logging**
  Inspect:

  ```
  logs/main_system_logs.log
  ```

---

## Configuration Reference

All settings are defined in `config.py`:

| Setting                 | Purpose                       |
| ----------------------- | ----------------------------- |
| `MAX_COMPLETION_TOKENS` | Max response size             |
| `CONVERSATION_MANAGER`  | Default memory strategy       |
| `MAX_MESSAGES`          | Fixed window size             |
| `WINDOW_SIZE`           | Summarization trigger         |
| `SUMMARIZE_COUNT`       | Messages summarized per cycle |
| `SYSTEM_PROMPT`         | Base instruction to model     |
| `LOG_LEVEL`             | Logging verbosity             |
| `LOG_FILE`              | Log file path                 |

---

## License

This project is licensed under the [MIT License](LICENSE).