"""
Shared Setup: Lightweight sync config for examples 1-6.
Provides Azure OpenAI client, Ollama model name, and common imports.
"""

import os
from pathlib import Path

import openai
from dotenv import load_dotenv

# Load environment variables from keys/.env
load_dotenv(Path(__file__).resolve().parent.parent.parent / "keys" / ".env")

# Azure OpenAI client (sync)
azure_client = openai.AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

AZURE_MODEL = os.environ["AZURE_OPENAI_MODEL_NAME"]
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL_NAME")
