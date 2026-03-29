"""
AIManagers.py  --  Handles communication with the OpenAI APIs.

This module creates OpenAI clients (Azure and standard) using your
credentials and sends chat messages to the model.  It returns a simple
dictionary with the result so the rest of the app can use it easily.
"""

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_MODEL_NAME,
    MAX_COMPLETION_TOKENS,
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME,
    logger,
)
from openai import AzureOpenAI, OpenAI


# ── ChatResult ─────────────────────────────────────────────────────────────
# A simple container for the result of an AI chat completion call.
# Works just like TokenUsage -- plain attributes, no fuss.

class ChatResult:

    def __init__(self, status, reply=None, error=None, raw=None):
        self.status = status           # True if the call succeeded
        self.reply = reply             # the AI's text reply
        self.error = error             # error details (None on success)
        self.raw = raw                 # full API response object


class AzureOpenAIManager:
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
        # Store model settings so we can use them in every request
        self.model_name            = AZURE_OPENAI_MODEL_NAME
        self.max_completion_tokens  = MAX_COMPLETION_TOKENS

        # Create the Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

    def chat_completion(self, messages):
        """
        Send a list of messages to the AI model and get a response.

        Parameters:
            messages -- a list of dicts like:
                [{"role": "system", "content": "You are helpful."},
                 {"role": "user",   "content": "Hello!"}]

        Returns:
            A ChatResult with these attributes:
                status  -- True if the call succeeded, False otherwise
                reply   -- the AI's text reply (None on failure)
                error   -- error details if something went wrong
                raw     -- the full API response object (for token tracking)
        """
        try:
            # Call the Azure OpenAI chat completion endpoint
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
            )

            # Check that we actually got a reply
            if response and response.choices and len(response.choices) > 0:
                reply_text = response.choices[0].message.content
                logger.info("Azure OpenAI chat completion successful.")
                return ChatResult(status=True, reply=reply_text, raw=response)
            else:
                return ChatResult(status=False)

        except Exception as e:
            logger.error("Azure OpenAI chat completion error: %s", str(e))
            return ChatResult(status=False, error=str(e))


class OpenAIManager:
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
        """
        Send a list of messages to the AI model and get a response.

        Parameters:
            messages -- a list of dicts like:
                [{"role": "system", "content": "You are helpful."},
                 {"role": "user",   "content": "Hello!"}]

        Returns:
            A ChatResult with these attributes:
                status  -- True if the call succeeded, False otherwise
                reply   -- the AI's text reply (None on failure)
                error   -- error details if something went wrong
                raw     -- the full API response object (for token tracking)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
            )

            if response and response.choices and len(response.choices) > 0:
                reply_text = response.choices[0].message.content
                logger.info("OpenAI chat completion successful.")
                return ChatResult(status=True, reply=reply_text, raw=response)
            else:
                return ChatResult(status=False)

        except Exception as e:
            logger.error("OpenAI chat completion error: %s", str(e))
            return ChatResult(status=False, error=str(e))
