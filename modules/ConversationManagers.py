"""
ConversationManagers.py  --  Manages chat history for each session.

There are three conversation managers to choose from (set in config.py):

  1. SimpleConversationManager
     Keeps the entire chat history.  Nothing is ever removed.

  2. FixedWindowConversationManager
     Keeps only the most recent N messages.  Older ones are dropped.

  3. SummarizingConversationManager
     When the history gets too long, older messages are summarised
     into a short paragraph so context is preserved without sending
     the full history to the model every time.

The function `get_conversation_manager()` at the bottom of this file
reads the CONVERSATION_MANAGER setting from config and returns the
right manager automatically.
"""

from uuid import uuid4

from config import (
    SYSTEM_PROMPT,
    CONVERSATION_MANAGER,
    MAX_MESSAGES,
    WINDOW_SIZE,
    SUMMARIZE_COUNT,
    logger,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Simple Conversation Manager  (keeps everything)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleConversationManager:
    """
    The simplest approach: store every message forever.
    Good for short conversations where memory isn't a concern.
    """

    def __init__(self):
        # sessions is a dict:  session_id -> list of message dicts
        self.sessions = {}

    def start_session(self):
        """Create a new chat session and return its unique id."""
        session_id = str(uuid4())
        # Every session starts with the system prompt
        self.sessions[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        return session_id

    def get_messages(self, session_id):
        """Return all messages for a session."""
        return self.sessions.get(session_id, [])

    def add_user_message(self, session_id, content):
        """Add a user message to the session."""
        self.sessions[session_id].append(
            {"role": "user", "content": content}
        )

    def add_assistant_message(self, session_id, content):
        """Add an assistant (AI) message to the session."""
        self.sessions[session_id].append(
            {"role": "assistant", "content": content}
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Fixed-Window Conversation Manager  (keeps last N messages)
# ═══════════════════════════════════════════════════════════════════════════

class FixedWindowConversationManager:
    """
    Keeps only the last `max_messages` user/assistant messages.
    The system prompt at position 0 is always preserved.
    """

    def __init__(self, max_messages=20):
        self.max_messages = max_messages
        self.sessions = {}

    def start_session(self):
        """Create a new chat session and return its unique id."""
        session_id = str(uuid4())
        self.sessions[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        return session_id

    def _trim(self, session_id):
        """Remove the oldest non-system messages if we're over the limit."""
        messages = self.sessions[session_id]

        # Everything after the system prompt (index 0) is a chat message
        chat_messages = messages[1:]

        if len(chat_messages) > self.max_messages:
            # Keep the system prompt + only the newest messages
            self.sessions[session_id] = (
                [messages[0]] + chat_messages[-self.max_messages:]
            )

    def get_messages(self, session_id):
        """Return messages for a session (trimmed to the window size)."""
        self._trim(session_id)
        return self.sessions.get(session_id, [])

    def add_user_message(self, session_id, content):
        """Add a user message to the session."""
        self.sessions[session_id].append(
            {"role": "user", "content": content}
        )

    def add_assistant_message(self, session_id, content):
        """Add an assistant (AI) message to the session."""
        self.sessions[session_id].append(
            {"role": "assistant", "content": content}
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Summarizing Conversation Manager  (compresses old messages)
# ═══════════════════════════════════════════════════════════════════════════

class SummarizingConversationManager:
    """
    When the chat window fills up, the oldest messages are sent to the
    AI to be compressed into a short summary.  This keeps the message
    list small while preserving important context.

    How it works (with default settings window_size=10, summarize_count=5):
      - Chat normally until there are 10 non-system messages.
      - Take the 5 oldest messages and ask the AI to summarise them.
      - Store that summary separately.
      - Remove those 5 messages from the active window.
      - Next time get_messages() is called, the summary is injected
        right after the system prompt so the AI still has context.
    """

    # This prompt tells the AI how to write a good summary
    SUMMARIZE_PROMPT = (
        "You are a concise summariser. Condense the following conversation "
        "messages into a short, factual summary that preserves all key "
        "information, decisions, and context needed to continue the "
        "conversation. Respond with the summary only."
    )

    def __init__(self, window_size=10, summarize_count=5, ai_manager=None):
        self.window_size = window_size
        self.summarize_count = summarize_count
        self.ai_manager = ai_manager   # used to call the AI for summaries
        self.sessions = {}             # session_id -> list of messages
        self.summaries = {}            # session_id -> list of summary strings

    def start_session(self):
        """Create a new chat session and return its unique id."""
        session_id = str(uuid4())
        self.sessions[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.summaries[session_id] = []
        return session_id

    def get_messages(self, session_id):
        """
        Return the messages to send to the model.

        The result is always:
          [system prompt, (summary if any), ...recent chat messages]
        """
        # First, check if we need to summarise old messages
        self._maybe_summarize(session_id)

        system_msg = self.sessions[session_id][0]
        chat_msgs = self.sessions[session_id][1:]

        result = [system_msg]

        # If we have summaries, add them as a system message
        if self.summaries.get(session_id):
            combined = "\n---\n".join(self.summaries[session_id])
            result.append({
                "role": "system",
                "content": "Summary of earlier conversation:\n" + combined,
            })

        # Add the recent chat messages
        result.extend(chat_msgs)
        return result

    def add_user_message(self, session_id, content):
        """Add a user message to the session."""
        self.sessions[session_id].append(
            {"role": "user", "content": content}
        )

    def add_assistant_message(self, session_id, content):
        """Add an assistant (AI) message to the session."""
        self.sessions[session_id].append(
            {"role": "assistant", "content": content}
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _maybe_summarize(self, session_id):
        """If the window is full, summarise the oldest messages."""
        messages = self.sessions[session_id]
        chat_msgs = messages[1:]  # everything except the system prompt

        # Not full yet -- nothing to do
        if len(chat_msgs) < self.window_size:
            return

        # Split into "old messages to summarise" and "recent messages to keep"
        old_msgs = chat_msgs[: self.summarize_count]
        remaining = chat_msgs[self.summarize_count:]

        logger.info(
            "Session %s: summarising %d messages (%d remain in window)",
            session_id, len(old_msgs), len(remaining),
        )

        # Get the summary text
        summary_text = self._summarize(old_msgs)
        self.summaries[session_id].append(summary_text)

        # Rebuild the session: system prompt + remaining messages only
        self.sessions[session_id] = [messages[0]] + remaining

    def _summarize(self, messages):
        """
        Ask the AI to summarise a list of messages.

        If no AI manager is available (or the call fails), fall back
        to simply joining the messages together as plain text.
        """
        # Try using the AI if we have a manager
        if self.ai_manager is not None:
            prompt = [
                {"role": "system", "content": self.SUMMARIZE_PROMPT},
            ] + messages

            result = self.ai_manager.chat_completion(prompt)

            if result.status:
                logger.info("LLM summarisation successful.")
                return result.reply

            logger.warning("LLM summarisation failed -- using fallback.")

        # Fallback: just concatenate the messages as text
        return "\n".join(
            m["role"] + ": " + m["content"] for m in messages
        )


# ═══════════════════════════════════════════════════════════════════════════
# Manager selector  --  picks the right manager based on config
# ═══════════════════════════════════════════════════════════════════════════

def get_conversation_manager(ai_manager=None):
    """
    Read the CONVERSATION_MANAGER setting from config.py and return
    the matching conversation manager object.

    Parameters:
        ai_manager -- (optional) the AI manager, passed to the
                      summarizing manager so it can call the AI.
    """
    manager_type = CONVERSATION_MANAGER.lower()

    if manager_type == "simple":
        return SimpleConversationManager()

    elif manager_type == "fixed_window":
        return FixedWindowConversationManager(max_messages=MAX_MESSAGES)

    elif manager_type == "summarizing":
        return SummarizingConversationManager(
            window_size=WINDOW_SIZE,
            summarize_count=SUMMARIZE_COUNT,
            ai_manager=ai_manager,
        )

    else:
        raise ValueError(
            "Unknown conversation manager: '" + CONVERSATION_MANAGER + "'. "
            "Choose one of: simple, fixed_window, summarizing."
        )

