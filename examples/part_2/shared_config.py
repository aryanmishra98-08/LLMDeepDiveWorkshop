"""
Shared Setup: Async Client and Token Tracker.
All advanced examples (7-17) depend on this shared foundation.
Requirements: pip install openai tiktoken python-dotenv ollama
"""

import asyncio
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import openai
from dotenv import load_dotenv

# Load environment variables from keys/.env
load_dotenv(Path(__file__).resolve().parent.parent.parent / "keys" / ".env")

# Azure OpenAI async client
azure_client = openai.AsyncAzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

AZURE_MODEL = os.environ["AZURE_OPENAI_MODEL_NAME"]
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL_NAME")


@dataclass
class TokenTracker:
    """Tracks cumulative token usage and estimated cost across calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    # Azure OpenAI pricing varies by deployment — update to match your tier
    input_price: float = 0.15   # per 1M tokens (placeholder)
    output_price: float = 0.60  # per 1M tokens (placeholder)

    def update(self, usage):
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost(self) -> float:
        return (self.prompt_tokens / 1_000_000 * self.input_price +
                self.completion_tokens / 1_000_000 * self.output_price)

    def report(self) -> str:
        return (f"Calls: {self.calls} | Tokens: {self.total_tokens:,} "
                f"(in: {self.prompt_tokens:,}, out: {self.completion_tokens:,}) | "
                f"Est. cost: ${self.estimated_cost:.4f}")


async def chat(messages: list[dict], tracker: TokenTracker,
               model: str = None, temperature: float = None,
               max_tokens: int = 2048) -> str:
    """Async chat completion via Azure OpenAI with token tracking and error handling."""
    model = model or AZURE_MODEL
    try:
        kwargs = dict(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await azure_client.chat.completions.create(**kwargs)
        tracker.update(response.usage)
        return response.choices[0].message.content
    except Exception as e:
        print(f"[API Error] {type(e).__name__}: {e}")
        raise
