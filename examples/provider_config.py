"""
Provider Configuration — Single source of truth for LLM provider
registry, credential validation, client initialization, and helpers.

Supports: Azure OpenAI, OpenAI, Anthropic, Gemini, Ollama.
Adding a new provider requires:
  1. An entry in PROVIDER_REGISTRY
  2. A branch in _create_sync_client / _create_async_client
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv

# Load environment variables from keys/.env (idempotent if already loaded)
load_dotenv(Path(__file__).resolve().parent.parent / "keys" / ".env")


# ── Anthropic → OpenAI adapter ────────────────────────────────────────
# Anthropic uses its own SDK and API shape.  The adapter below wraps it
# into the same interface that the openai SDK exposes so that every
# example file can call  client.chat.completions.create(...)  unchanged.

@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int

@dataclass
class _Message:
    content: str
    role: str = "assistant"

@dataclass
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"

@dataclass
class _ChatCompletion:
    choices: list[_Choice]
    usage: Optional[_Usage]


class _AnthropicCompletions:
    """Sync wrapper: translates openai-style create() → anthropic messages API."""

    def __init__(self, anthropic_client):
        self._client = anthropic_client

    def create(self, *, model, messages, max_completion_tokens=4096,
               temperature=None, **kwargs):
        # Anthropic treats system messages separately
        system_text = None
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system_text = (system_text or "") + m["content"] + "\n"
            else:
                filtered.append(m)

        api_kwargs = dict(model=model, messages=filtered,
                          max_tokens=max_completion_tokens)
        if system_text:
            api_kwargs["system"] = system_text.strip()
        if temperature is not None:
            api_kwargs["temperature"] = temperature

        resp = self._client.messages.create(**api_kwargs)

        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = _Usage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
        return _ChatCompletion(
            choices=[_Choice(message=_Message(content=text))],
            usage=usage,
        )


class _AsyncAnthropicCompletions:
    """Async wrapper: same translation for the async Anthropic client."""

    def __init__(self, anthropic_client):
        self._client = anthropic_client

    async def create(self, *, model, messages, max_completion_tokens=4096,
                     temperature=None, **kwargs):
        system_text = None
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system_text = (system_text or "") + m["content"] + "\n"
            else:
                filtered.append(m)

        api_kwargs = dict(model=model, messages=filtered,
                          max_tokens=max_completion_tokens)
        if system_text:
            api_kwargs["system"] = system_text.strip()
        if temperature is not None:
            api_kwargs["temperature"] = temperature

        resp = await self._client.messages.create(**api_kwargs)

        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = _Usage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
        return _ChatCompletion(
            choices=[_Choice(message=_Message(content=text))],
            usage=usage,
        )


class _ChatNamespace:
    """Mimics  client.chat.completions  for the Anthropic adapters."""
    def __init__(self, completions):
        self.completions = completions

class _AnthropicAdapter:
    """Sync adapter:  client.chat.completions.create(...) works like openai."""
    def __init__(self, anthropic_client):
        self.chat = _ChatNamespace(_AnthropicCompletions(anthropic_client))

class _AsyncAnthropicAdapter:
    """Async adapter:  await client.chat.completions.create(...) works like openai."""
    def __init__(self, anthropic_client):
        self.chat = _ChatNamespace(_AsyncAnthropicCompletions(anthropic_client))

# ── Provider registry ─────────────────────────────────────────────────
# Each entry lists the env vars that MUST be set for that provider.

PROVIDER_REGISTRY: dict[str, dict] = {
    "azure": {
        "required_vars": [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_MODEL_NAME",
        ],
        "model_var": "AZURE_OPENAI_MODEL_NAME",
    },
    "openai": {
        "required_vars": ["OPENAI_API_KEY", "OPENAI_MODEL_NAME"],
        "model_var": "OPENAI_MODEL_NAME",
    },
    "anthropic": {
        "required_vars": ["ANTHROPIC_API_KEY", "ANTHROPIC_MODEL_NAME"],
        "model_var": "ANTHROPIC_MODEL_NAME",
    },
    "gemini": {
        "required_vars": ["GEMINI_API_KEY", "GEMINI_MODEL_NAME"],
        "model_var": "GEMINI_MODEL_NAME",
    },
    "ollama": {
        "required_vars": ["OLLAMA_BASE_URL", "OLLAMA_MODEL_NAME"],
        "model_var": "OLLAMA_MODEL_NAME",
    },
}


# ── Validation ────────────────────────────────────────────────────────

def validate_provider(provider: str) -> dict:
    """Validate that *provider* is known and all its credentials are set.

    Returns the registry entry on success.
    Raises on unknown provider or missing env vars (fail-fast).
    """
    if provider not in PROVIDER_REGISTRY:
        supported = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Supported providers: {supported}"
        )

    config = PROVIDER_REGISTRY[provider]
    missing = [v for v in config["required_vars"] if not os.environ.get(v)]

    if missing:
        raise EnvironmentError(
            f"LLM_PROVIDER='{provider}' requires the following env var(s) "
            f"which are missing or empty: {', '.join(missing)}"
        )

    return config


# ── Client factories ──────────────────────────────────────────────────

def _create_sync_client(provider: str) -> tuple[openai.OpenAI, str]:
    """Return (sync_client, model_name) for the given provider."""
    model = os.environ[PROVIDER_REGISTRY[provider]["model_var"]]

    if provider == "azure":
        client = openai.AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    elif provider == "openai":
        client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif provider == "anthropic":
        import anthropic as _anthropic
        client = _AnthropicAdapter(
            _anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        )
    elif provider == "gemini":
        base_url = os.environ.get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        client = openai.OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url=base_url,
        )
    elif provider == "ollama":
        client = openai.OpenAI(
            api_key="ollama",
            base_url=os.environ["OLLAMA_BASE_URL"],
        )
    else:
        # Should never reach here after validation, but just in case.
        raise ValueError(f"No client factory for provider '{provider}'")

    return client, model


def _create_async_client(provider: str):
    """Return (async_client, model_name) for the given provider."""
    model = os.environ[PROVIDER_REGISTRY[provider]["model_var"]]

    if provider == "azure":
        client = openai.AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    elif provider == "openai":
        client = openai.AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif provider == "anthropic":
        import anthropic as _anthropic
        client = _AsyncAnthropicAdapter(
            _anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        )
    elif provider == "gemini":
        base_url = os.environ.get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        client = openai.AsyncOpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url=base_url,
        )
    elif provider == "ollama":
        client = openai.AsyncOpenAI(
            api_key="ollama",
            base_url=os.environ["OLLAMA_BASE_URL"],
        )
    else:
        raise ValueError(f"No async client factory for provider '{provider}'")

    return client, model


# ── Public helpers ────────────────────────────────────────────────────

def resolve_provider() -> str:
    """Read and normalise LLM_PROVIDER from the environment."""
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        raise EnvironmentError(
            "LLM_PROVIDER env var must be set "
            f"(supported: {', '.join(sorted(PROVIDER_REGISTRY))})"
        )
    return provider


def _ping_sync(client, model: str, provider: str) -> None:
    """Send a lightweight test prompt to verify credentials and connectivity."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=5,
        )
        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("Empty response from provider")
    except Exception as e:
        raise ConnectionError(
            f"LLM_PROVIDER='{provider}' validation ping failed. "
            f"Credentials or endpoint may be misconfigured.\n"
            f"  {type(e).__name__}: {e}"
        ) from e


async def _ping_async(client, model: str, provider: str) -> None:
    """Async version of the validation ping."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=5,
        )
        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("Empty response from provider")
    except Exception as e:
        raise ConnectionError(
            f"LLM_PROVIDER='{provider}' validation ping failed. "
            f"Credentials or endpoint may be misconfigured.\n"
            f"  {type(e).__name__}: {e}"
        ) from e


def init_sync_client() -> tuple[openai.OpenAI, str, str]:
    """Validate env, create a sync client, verify with a test prompt.

    Returns (client, model, provider).
    """
    global _active_client, _active_model, _active_provider
    provider = resolve_provider()
    validate_provider(provider)
    client, model = _create_sync_client(provider)
    _ping_sync(client, model, provider)
    _active_client, _active_model, _active_provider = client, model, provider
    return client, model, provider


def init_async_client():
    """Validate env, create an async client.

    NOTE: The async validation ping is deferred — call
    ``await ping()`` once an event loop is running
    (e.g. at the top of your async main).

    Returns (client, model, provider).
    """
    global _active_client, _active_model, _active_provider
    provider = resolve_provider()
    validate_provider(provider)
    client, model = _create_async_client(provider)
    _active_client, _active_model, _active_provider = client, model, provider
    return client, model, provider


async def verify_async_client(client, model: str, provider: str) -> None:
    """Run the validation ping for an async client (call from async context)."""
    await _ping_async(client, model, provider)


# ── Token tracking & async chat helper ────────────────────────────────

@dataclass
class TokenTracker:
    """Tracks cumulative token usage and estimated cost across calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    # Pricing varies by provider/deployment — update to match your tier
    input_price: float = 0.15   # per 1M tokens (placeholder)
    output_price: float = 0.60  # per 1M tokens (placeholder)

    def update(self, usage):
        if usage is None:
            self.calls += 1
            return
        self.prompt_tokens += usage.prompt_tokens or 0
        self.completion_tokens += usage.completion_tokens or 0
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


# These module-level references are set by init_sync_client / init_async_client.
# They allow chat() and ping() to use the active client without extra args.
_active_client = None
_active_model = None
_active_provider = None


async def chat(messages: list[dict], tracker: TokenTracker,
               model: str = None, temperature: float = None,
               max_tokens: int = 2048) -> str:
    """Async chat completion with token tracking and error handling."""
    model = model or _active_model
    try:
        kwargs = dict(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await _active_client.chat.completions.create(**kwargs)
        tracker.update(response.usage)
        return response.choices[0].message.content
    except Exception as e:
        print(f"[API Error] {type(e).__name__}: {e}")
        raise


async def ping():
    """Verify provider connectivity. Call once at the start of async main."""
    await verify_async_client(_active_client, _active_model, _active_provider)
