"""
config.py  --  Application settings and logger setup.

Loads credentials from a .env file, defines conversation
settings, and creates a shared logger that all other modules can import.
"""

import logging
import logging.handlers
import os

from dotenv import load_dotenv


# ── Load environment variables from the .env file ──────────────────────────

load_dotenv("keys/.env")


# ── Azure OpenAI Settings ─────────────────────────────────────────────────
# These values come from your .env file. They tell the app how to connect
# to the Azure OpenAI API.

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_MODEL_NAME  = os.getenv("AZURE_OPENAI_MODEL_NAME", "")


# ── OpenAI Settings ───────────────────────────────────────────────────────
# These values come from your .env file. They tell the app how to connect
# to the OpenAI API (non-Azure).

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "")


# ── Anthropic Settings ────────────────────────────────────────────────────
# These values come from your .env file. They tell the app how to connect
# to the Anthropic API.

ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "")


# ── Gemini Settings ───────────────────────────────────────────────────────
# Gemini exposes an OpenAI-compatible endpoint, so no extra SDK is needed.
# Set GEMINI_BASE_URL in keys/.env to point the OpenAI client at Google's API.

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "")
GEMINI_BASE_URL   = os.getenv("GEMINI_BASE_URL", "")


# ── Ollama Settings ───────────────────────────────────────────────────────
# Ollama runs locally and exposes an OpenAI-compatible endpoint.
# Set OLLAMA_BASE_URL in keys/.env to point the OpenAI client at your Ollama instance.

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "")


# Model behaviour defaults (can be tweaked here)
MAX_COMPLETION_TOKENS = 2048


# ── Conversation Settings ─────────────────────────────────────────────────
# Controls which conversation manager to use and how it behaves.
#   "simple"       -> keeps the full chat history, no trimming
#   "fixed_window" -> keeps only the last MAX_MESSAGES messages
#   "summarizing"  -> summarises older messages to save space

CONVERSATION_MANAGER = "simple"
MAX_MESSAGES         = 20    # how many messages to keep (fixed_window)
WINDOW_SIZE          = 10    # when to trigger summarisation (summarizing)
SUMMARIZE_COUNT      = 5     # how many old messages to summarise at once


# ── System Prompt ─────────────────────────────────────────────────────────
# This is the instruction given to the AI at the start of every conversation.

SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. Answer questions clearly "
    "and concisely."
)


# ── Logging Settings ──────────────────────────────────────────────────────

LOG_LEVEL = "INFO"
LOG_FILE  = "logs/main_system_logs.log"


# ── Logger Setup ──────────────────────────────────────────────────────────
# Creates a logger that writes messages to a log file.  Other modules
# import `logger` and call logger.info(...), logger.error(...), etc.

# Make sure the logs folder exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Set up the log message format
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Write logs to a file that rotates every midnight (keeps 30 days)
file_handler = logging.handlers.TimedRotatingFileHandler(
    filename=LOG_FILE,
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8",
)
file_handler.setFormatter(formatter)

# Build the logger object
logger = logging.getLogger("app")
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger.addHandler(file_handler)
logger.propagate = False  # avoid duplicate log output
