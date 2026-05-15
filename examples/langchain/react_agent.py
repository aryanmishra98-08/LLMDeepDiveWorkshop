"""
ReAct Agent (LangChain):
Demonstrates the See → Think → Do → Check loop using LangChain's ReAct agent.
Every Thought, Action, and Observation prints live as the agent reasons step-by-step.

Provider selection is driven by LLM_PROVIDER in keys/.env.
Supported values: openai | azure | anthropic | ollama
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Union

# Load credentials from keys/.env
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / "keys" / ".env"
if _ENV_PATH.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import BaseMessage
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent

# ── Logging setup ─────────────────────────────────────────────────────────────
_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_RUN_TS = time.strftime("%Y%m%d_%H%M%S")
_LOG_FILE = _LOGS_DIR / f"react_{_RUN_TS}.log"

log = logging.getLogger("react_agent")
log.setLevel(logging.DEBUG)
log.handlers.clear()
log.propagate = False

_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_formatter)

_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_formatter)

log.addHandler(_console_handler)
log.addHandler(_file_handler)
log.info("Logging initialized | file: %s", _LOG_FILE)

# ── Config ────────────────────────────────────────────────────────────────────

QUESTION = "Who founded OpenAI and what are each of the founders doing now?"
PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# ── LLM Factory ───────────────────────────────────────────────────────────────

def build_llm() -> tuple[Any, str]:
    """
    Initialise the correct LangChain chat model based on LLM_PROVIDER.

    Returns
    -------
    (llm, model_label) — the instantiated chat model and a human-readable
    identifier used only for display/logging purposes.
    """
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log.error("OPENAI_API_KEY is not set — add it to keys/.env")
            sys.exit(1)
        log.info("Provider: OpenAI")
        return ChatOpenAI(model=model, temperature=0, api_key=api_key), model

    if PROVIDER == "azure":
        from langchain_openai import AzureChatOpenAI
        model      = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o")
        endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key    = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        missing = [k for k, v in {
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_API_KEY":  api_key,
        }.items() if not v]
        if missing:
            log.error("Missing Azure env vars: %s", ", ".join(missing))
            sys.exit(1)
        log.info("Provider: Azure OpenAI")
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=model,
            api_version=api_version,
            api_key=api_key,
            temperature=0,
        ), f"azure/{model}"

    if PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model   = os.getenv("ANTHROPIC_MODEL_NAME", "claude-sonnet-4-6")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            log.error("ANTHROPIC_API_KEY is not set — add it to keys/.env")
            sys.exit(1)
        log.info("Provider: Anthropic")
        return ChatAnthropic(model=model, temperature=0, api_key=api_key), model

    if PROVIDER == "ollama":
        from langchain_openai import ChatOpenAI  # Ollama exposes an OpenAI-compatible API
        model    = os.getenv("OLLAMA_MODEL_NAME", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        log.info("Provider: Ollama")
        return ChatOpenAI(
            model=model,
            temperature=0,
            base_url=base_url,
            api_key="ollama",  # Ollama ignores the key; value is required by the client
        ), f"ollama/{model}"

    log.error(
        "Unknown LLM_PROVIDER='%s'. Supported values: openai | azure | anthropic | ollama",
        PROVIDER,
    )
    sys.exit(1)


# ── Callback handler (structured logging) ─────────────────────────────────────

class AgentLogger(BaseCallbackHandler):
    """Logs every meaningful event in the agent's execution lifecycle."""

    def __init__(self) -> None:
        super().__init__()
        self._chain_start_time: float = 0.0
        self._tool_start_time: float  = 0.0
        self._llm_start_time: float   = 0.0

    # ── LLM ──────────────────────────────────────────────────────────────────

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        """Called when a plain (non-chat) LLM call begins.

        Args:
            serialized: Serialised LLM configuration dict, including model name
                        under ``serialized["kwargs"]["model_name"]``.
            prompts: List of raw string prompts sent to the LLM.
        """
        self._llm_start_time = time.perf_counter()
        model_name = serialized.get("kwargs", {}).get("model_name") or serialized.get("name", "unknown")
        log.debug("LLM call started | model: %s | prompt chars: %d",
                  model_name, sum(len(p) for p in prompts))

    def on_chat_model_start(
        self, serialized: dict, messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        """Called when a chat-based LLM call begins (used by ChatOpenAI, ChatAnthropic, etc.).

        Args:
            serialized: Serialised chat model configuration.
            messages: Batches of ``BaseMessage`` objects representing the conversation
                      history passed to the model.
        """
        self._llm_start_time = time.perf_counter()
        model_name = serialized.get("kwargs", {}).get("model_name") or serialized.get("name", "unknown")
        total_chars = sum(len(str(m)) for batch in messages for m in batch)
        log.debug("Chat model call started | model: %s | message chars: %d", model_name, total_chars)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when an LLM call completes successfully.

        Args:
            response: ``LLMResult`` containing generations and optional token-usage
                      metadata under ``response.llm_output["token_usage"]``.
        """
        elapsed = time.perf_counter() - self._llm_start_time
        generations = response.generations
        token_usage = (response.llm_output or {}).get("token_usage")
        log.debug("LLM call finished | elapsed: %.2fs | generations: %d | tokens: %s",
                  elapsed, sum(len(g) for g in generations), token_usage)

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when an LLM call raises an exception.

        Args:
            error: The exception raised by the LLM client.
        """
        log.error("LLM error: %s", error)

    # ── Tools ─────────────────────────────────────────────────────────────────

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        """Called immediately before a tool is invoked by the agent.

        Args:
            serialized: Serialised tool configuration, including the tool name.
            input_str: The raw string query the agent is passing to the tool.
        """
        self._tool_start_time = time.perf_counter()
        tool_name = serialized.get("name", "unknown")
        preview   = input_str[:200] + ("…" if len(input_str) > 200 else "")
        log.info("  → Tool call  | %-15s | input: %s", tool_name, preview)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool returns its result to the agent.

        Args:
            output: The string result returned by the tool. Logged at INFO
                    level with a 300-character preview to keep output readable.
        """
        elapsed  = time.perf_counter() - self._tool_start_time
        chars    = len(str(output))
        preview  = str(output)[:300] + ("…" if chars > 300 else "")
        log.info("  ← Tool result| elapsed: %.2fs | %d chars | preview: %s",
                 elapsed, chars, preview)

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when a tool raises an exception instead of returning a result.

        Args:
            error: The exception raised inside the tool function. The tool
                   wrappers below catch most exceptions and return error strings
                   instead, so this fires only for truly unexpected failures.
        """
        log.error("  ✗ Tool error : %s", error)

    # ── Agent ─────────────────────────────────────────────────────────────────

    # on_agent_action / on_agent_finish / on_chain_* are AgentExecutor hooks
    # and are not emitted by LangGraph nodes — omitted intentionally.


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate and run the ReAct agent end-to-end.

    Steps:
        1. Initialise the two search tools (DuckDuckGo + Wikipedia).
        2. Build the LLM from the active provider config (``LLM_PROVIDER`` env var).
        3. Construct a LangGraph ReAct agent with a research-focused system prompt.
        4. Invoke the agent with ``QUESTION`` and stream structured logs via
           ``AgentLogger`` callbacks.
        5. Print the final synthesised answer to stdout.

    Side effects:
        - Writes a timestamped log file to ``logs/react_YYYYMMDD_HHMMSS.log``.
        - Makes live HTTP requests to DuckDuckGo and Wikipedia.
    """
    # ── Tools ─────────────────────────────────────────────────────────────────

    log.info("Initialising tools: web_search, wikipedia")
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    _ddg_api = DuckDuckGoSearchAPIWrapper(region="us-en")

    def _safe_web_search(query: str) -> str:
        """Search the web for current information about a topic.

        A 1-second sleep before the request helps avoid DuckDuckGo rate-limiting
        when the agent fires multiple searches in quick succession.
        Returns an error string on failure instead of raising, so the agent can
        decide whether to retry or fall back to Wikipedia.
        """
        time.sleep(1)  # polite delay — prevents DuckDuckGo rate-limit (429) errors
        try:
            return _ddg_api.run(query)
        except Exception as e:
            return f"Web search failed for '{query}': {e}"

    search = StructuredTool.from_function(
        func=_safe_web_search,
        name="web_search",
        description="Search the web for current information about a topic.",
    )

    _wiki_api = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)

    def _safe_wikipedia(query: str) -> str:
        """Search Wikipedia for factual information about a person, place, or topic.

        Returns the top-2 Wikipedia article summaries (up to 2000 characters each).
        A 1-second sleep mirrors the DuckDuckGo wrapper to keep inter-call spacing
        consistent. Returns an error string on failure rather than raising.
        """
        time.sleep(1)  # consistent polite delay matching web_search
        try:
            return _wiki_api.run(query)
        except Exception as e:
            return f"Wikipedia lookup failed for '{query}': {e}"

    wiki = StructuredTool.from_function(
        func=_safe_wikipedia,
        name="wikipedia",
        description="Search Wikipedia for factual information about a person, place, or topic.",
    )
    tools = [search, wiki]
    log.info("Tools ready: %s", [t.name for t in tools])

    # ── LLM ───────────────────────────────────────────────────────────────────

    llm, model_label = build_llm()
    log.info("LLM ready | %s", model_label)

    # ── Agent ─────────────────────────────────────────────────────────────────

    log.info("Constructing agent")
    # System prompt shapes the agent's reasoning style: it instructs the model
    # to use tools for factual claims and remain concise. Adjust this text to
    # change how the agent reasons or formats its final answer.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a research assistant. Use available tools to find accurate, "
            "up-to-date information. Always cite sources from tool results. "
            "Be concise and factual in your final answer."
        ),
    )

    # ── Run ───────────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("REACT AGENT — LangGraph Tool-Use Demo")
    print("=" * 60)
    print(f"\n📋 Question : {QUESTION}")
    print(f"⚙️  Provider : {PROVIDER}  |  Model: {model_label}")
    print(f"🔧 Tools     : {', '.join(t.name for t in tools)}")
    print(f"\n{'─' * 50}")
    print("Watch for agent tool calls and responses in the log output above.")
    print("─" * 50 + "\n")

    log.info("Invoking agent | question: %s", QUESTION)
    _start = time.perf_counter()
    # recursion_limit caps the number of Think→Act→Observe cycles (default 25).
    # max_concurrency=1 ensures tools run sequentially, which is important for
    # rate-sensitive APIs like DuckDuckGo that reject burst requests.
    result = agent.invoke(
        {"messages": [("human", QUESTION)]},
        config={"callbacks": [AgentLogger()], "recursion_limit": 25, "max_concurrency": 1},
    )
    _elapsed = time.perf_counter() - _start
    final_answer = result["messages"][-1].content
    log.info("Agent completed | total elapsed: %.2fs | answer chars: %d",
             _elapsed, len(final_answer))

    print("\n" + "═" * 60)
    print("RESULT")
    print("═" * 60)
    print(f"\n{final_answer}\n")


if __name__ == "__main__":
    main()
