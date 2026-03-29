"""
AIManagers.py  --  Handles communication with LLM providers.

Supports: Azure OpenAI, OpenAI, Anthropic, Google Gemini, Ollama.

All managers implement BaseAIManager and return a ChatResult with
a consistent shape, including normalised token counts, so the rest
of the app never needs to know which provider is in use.
"""

from abc import ABC, abstractmethod

import anthropic
from openai import AzureOpenAI, OpenAI

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL_NAME,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_MODEL_NAME,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL_NAME,
    MAX_COMPLETION_TOKENS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME,
    logger,
)


# ── ChatResult ─────────────────────────────────────────────────────────────
# A simple container for the result of an AI chat completion call.
# token counts are always populated on success so TokenTracker never has
# to inspect the raw response object.

class ChatResult:

    def __init__(
        self,
        status,
        reply=None,
        error=None,
        raw=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    ):
        self.status = status                    # True if the call succeeded
        self.reply = reply                      # the AI's text reply
        self.error = error                      # error details (None on success)
        self.raw = raw                          # full API response object
        self.prompt_tokens = prompt_tokens      # tokens sent to the model
        self.completion_tokens = completion_tokens  # tokens the model sent back
        self.total_tokens = total_tokens        # prompt + completion


# ── BaseAIManager ──────────────────────────────────────────────────────────
# Abstract base class that every provider manager must implement.
# The single required method keeps the contract simple and explicit.

class BaseAIManager(ABC):

    @abstractmethod
    def chat_completion(self, messages):
        """
        Send a list of messages to the model and return a ChatResult.

        Parameters:
            messages -- list of dicts:
                [{"role": "system"|"user"|"assistant", "content": "..."}]

        Returns:
            ChatResult
        """


# ── AzureOpenAIManager ─────────────────────────────────────────────────────

class AzureOpenAIManager(BaseAIManager):
    """
    Sends messages to Azure OpenAI and returns the response.

    Usage:
        manager = AzureOpenAIManager()
        result  = manager.chat_completion(messages)

        if result.status:
            print(result.reply)
        else:
            print(result.error)
    """

    def __init__(self):
        self.model_name            = AZURE_OPENAI_MODEL_NAME
        self.max_completion_tokens = MAX_COMPLETION_TOKENS

        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

    def chat_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
            )

            if response and response.choices and len(response.choices) > 0:
                reply_text = response.choices[0].message.content
                usage = response.usage
                logger.info("Azure OpenAI chat completion successful.")
                return ChatResult(
                    status=True,
                    reply=reply_text,
                    raw=response,
                    prompt_tokens=usage.prompt_tokens or 0,
                    completion_tokens=usage.completion_tokens or 0,
                    total_tokens=usage.total_tokens or 0,
                )
            else:
                return ChatResult(status=False)

        except Exception as e:
            logger.error("Azure OpenAI chat completion error: %s", str(e))
            return ChatResult(status=False, error=str(e))


# ── OpenAIManager ──────────────────────────────────────────────────────────

class OpenAIManager(BaseAIManager):
    """
    Sends messages to OpenAI (non-Azure) and returns the response.

    Usage:
        manager = OpenAIManager()
        result  = manager.chat_completion(messages)

        if result.status:
            print(result.reply)
        else:
            print(result.error)
    """

    def __init__(self):
        self.model_name            = OPENAI_MODEL_NAME
        self.max_completion_tokens = MAX_COMPLETION_TOKENS

        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def chat_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
            )

            if response and response.choices and len(response.choices) > 0:
                reply_text = response.choices[0].message.content
                usage = response.usage
                logger.info("OpenAI chat completion successful.")
                return ChatResult(
                    status=True,
                    reply=reply_text,
                    raw=response,
                    prompt_tokens=usage.prompt_tokens or 0,
                    completion_tokens=usage.completion_tokens or 0,
                    total_tokens=usage.total_tokens or 0,
                )
            else:
                return ChatResult(status=False)

        except Exception as e:
            logger.error("OpenAI chat completion error: %s", str(e))
            return ChatResult(status=False, error=str(e))


# ── AnthropicManager ───────────────────────────────────────────────────────

class AnthropicManager(BaseAIManager):
    """
    Sends messages to Anthropic (Claude) and returns the response.

    Anthropic's API separates the system prompt from the message list,
    so this manager extracts any system-role messages automatically.

    Usage:
        manager = AnthropicManager()
        result  = manager.chat_completion(messages)

        if result.status:
            print(result.reply)
        else:
            print(result.error)
    """

    def __init__(self):
        self.model_name  = ANTHROPIC_MODEL_NAME
        self.max_tokens  = MAX_COMPLETION_TOKENS

        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def chat_completion(self, messages):
        try:
            # Anthropic requires the system prompt as a separate kwarg.
            # Extract all system messages and join them; pass the rest as
            # the conversational message list.
            system_parts = [
                m["content"] for m in messages if m["role"] == "system"
            ]
            chat_messages = [m for m in messages if m["role"] != "system"]

            system_text = "\n\n".join(system_parts) if system_parts else None

            kwargs = dict(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=chat_messages,
            )
            if system_text:
                kwargs["system"] = system_text

            response = self.client.messages.create(**kwargs)

            if response and response.content and len(response.content) > 0:
                reply_text = response.content[0].text
                prompt_tokens     = response.usage.input_tokens or 0
                completion_tokens = response.usage.output_tokens or 0
                logger.info("Anthropic chat completion successful.")
                return ChatResult(
                    status=True,
                    reply=reply_text,
                    raw=response,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
            else:
                return ChatResult(status=False)

        except Exception as e:
            logger.error("Anthropic chat completion error: %s", str(e))
            return ChatResult(status=False, error=str(e))


# ── GeminiManager ──────────────────────────────────────────────────────────

class GeminiManager(BaseAIManager):
    """
    Sends messages to Google Gemini via its OpenAI-compatible endpoint.
    No extra SDK is required -- the standard openai package is used with
    a custom base_url pointing to Google's API.

    Usage:
        manager = GeminiManager()
        result  = manager.chat_completion(messages)

        if result.status:
            print(result.reply)
        else:
            print(result.error)
    """

    def __init__(self):
        self.model_name            = GEMINI_MODEL_NAME
        self.max_completion_tokens = MAX_COMPLETION_TOKENS

        self.client = OpenAI(
            base_url=GEMINI_BASE_URL,
            api_key=GEMINI_API_KEY,
        )

    def chat_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
            )

            if response and response.choices and len(response.choices) > 0:
                reply_text = response.choices[0].message.content
                usage = response.usage
                logger.info("Gemini chat completion successful.")
                return ChatResult(
                    status=True,
                    reply=reply_text,
                    raw=response,
                    prompt_tokens=usage.prompt_tokens or 0,
                    completion_tokens=usage.completion_tokens or 0,
                    total_tokens=usage.total_tokens or 0,
                )
            else:
                return ChatResult(status=False)

        except Exception as e:
            logger.error("Gemini chat completion error: %s", str(e))
            return ChatResult(status=False, error=str(e))


# ── OllamaManager ─────────────────────────────────────────────────────────

class OllamaManager(BaseAIManager):
    """
    Sends messages to a locally-running Ollama instance via its
    OpenAI-compatible endpoint.  No extra SDK is required.

    Ollama must be running on the machine (or reachable at OLLAMA_BASE_URL).
    The API key is set to a dummy value because the OpenAI client requires
    a non-empty string -- Ollama itself ignores it.

    Usage:
        manager = OllamaManager()
        result  = manager.chat_completion(messages)

        if result.status:
            print(result.reply)
        else:
            print(result.error)
    """

    def __init__(self):
        self.model_name            = OLLAMA_MODEL_NAME
        self.max_completion_tokens = MAX_COMPLETION_TOKENS

        self.client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama ignores the key; non-empty string required
        )

    def chat_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
            )

            if response and response.choices and len(response.choices) > 0:
                reply_text = response.choices[0].message.content
                usage = response.usage
                logger.info("Ollama chat completion successful.")
                return ChatResult(
                    status=True,
                    reply=reply_text,
                    raw=response,
                    prompt_tokens=usage.prompt_tokens or 0,
                    completion_tokens=usage.completion_tokens or 0,
                    total_tokens=usage.total_tokens or 0,
                )
            else:
                return ChatResult(status=False)

        except Exception as e:
            logger.error("Ollama chat completion error: %s", str(e))
            return ChatResult(status=False, error=str(e))
