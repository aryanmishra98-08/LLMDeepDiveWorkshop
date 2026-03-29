"""
ChatOrchestrator.py  --  The main brain of the application.

This module ties everything together:
  - Creates and manages chat sessions
  - Sends messages to the AI and gets replies
  - Tracks token usage

The CLI (app.py) talks to ChatOrchestrator, and ChatOrchestrator
talks to the AI manager, conversation manager, and token tracker.
"""

from config import logger, MAX_MESSAGES, WINDOW_SIZE, SUMMARIZE_COUNT
from modules.AIManagers import AzureOpenAIManager, ChatResult, OpenAIManager
from modules.ConversationManagers import (
    SimpleConversationManager,
    FixedWindowConversationManager,
    SummarizingConversationManager,
)
from modules.TokenTracker import TokenTracker, TokenUsage


# ── Available options the user can pick from ───────────────────────────────

# Each key is what the user sees; the value is used internally.
AVAILABLE_AI_MANAGERS = {
    "Azure OpenAI": "azure",
    "OpenAI": "openai",
}

AVAILABLE_MEMORY_MANAGERS = {
    "Simple (keep full history)": "simple",
    "Fixed Window (keep last N messages)": "fixed_window",
    "Summarizing (compress old messages)": "summarizing",
}


# ═══════════════════════════════════════════════════════════════════════════
# ChatOrchestrator
# ═══════════════════════════════════════════════════════════════════════════

class ChatOrchestrator:
    """
    The central controller that the rest of the app uses.

    It holds:
      - An AI manager        (talks to Azure OpenAI)
      - A conversation manager (stores chat history)
      - A token tracker      (counts how many tokens are used)
      - A user-sessions map  (which user owns which sessions)

    You pass in which AI provider and memory strategy to use
    when creating the orchestrator.
    """

    def __init__(self, ai_choice="azure", memory_choice="simple", shared_tracker=None):
        # ── Create the AI manager based on the user's choice ───────────────
        self.provider = ai_choice

        if ai_choice == "azure":
            self.ai_manager = AzureOpenAIManager()
        elif ai_choice == "openai":
            self.ai_manager = OpenAIManager()
        else:
            # Default to Azure if something unexpected is passed
            self.ai_manager = AzureOpenAIManager()

        # ── Create the conversation (memory) manager ───────────────────────
        self.manager_label = memory_choice

        if memory_choice == "fixed_window":
            self.conv_manager = FixedWindowConversationManager(max_messages=MAX_MESSAGES)
        elif memory_choice == "summarizing":
            self.conv_manager = SummarizingConversationManager(
                window_size=WINDOW_SIZE,
                summarize_count=SUMMARIZE_COUNT,
                ai_manager=self.ai_manager,
            )
        else:
            # Default to simple
            self.conv_manager = SimpleConversationManager()

        # ── Token tracker ──────────────────────────────────────────────────
        # Use a shared tracker if provided (so the admin view can see all
        # users' data across user switches), otherwise create a new one.
        if shared_tracker is not None:
            self.token_tracker = shared_tracker
        else:
            self.token_tracker = TokenTracker()

        # ── Track which sessions belong to which user ──────────────────────
        # Format:  { "alice": ["session-id-1", "session-id-2"], ... }
        self.user_sessions = {}

        logger.info(
            "ChatOrchestrator ready -- AI: %s, memory: %s",
            self.provider,
            self.manager_label,
        )

    # ── Session Management ─────────────────────────────────────────────────

    def create_session(self, username):
        """
        Start a new chat session for a user.
        Returns the new session id.
        """
        session_id = self.conv_manager.start_session()
        self.token_tracker.register_session(session_id, username)

        # Add this session to the user's list
        if username not in self.user_sessions:
            self.user_sessions[username] = []
        self.user_sessions[username].append(session_id)

        logger.info("New session %s for user '%s'", session_id, username)
        return session_id

    def get_user_sessions(self, username):
        """Return all session ids for a user."""
        return list(self.user_sessions.get(username, []))

    def get_session_count(self, username):
        """Return how many sessions a user has."""
        return len(self.user_sessions.get(username, []))

    # ── Sending Messages ───────────────────────────────────────────────────

    def send_message(self, session_id, user_message):
        """
        Send a message to the AI and get a reply.

        Steps:
          1. Save the user's message in the conversation history
          2. Get the full message list (may be trimmed or summarised)
          3. Send the messages to the AI
          4. Save the AI's reply in the conversation history
          5. Record how many tokens were used

        Returns a ChatResult with:
            status  -- True or False
            reply   -- the AI's response text (or None on failure)
            error   -- error details (or None if no error)
        """
        # Step 1: Save the user's message
        self.conv_manager.add_user_message(session_id, user_message)

        # Step 2: Get the conversation history to send
        messages = self.conv_manager.get_messages(session_id)

        # Step 3: Call the AI
        logger.info("Sending message to AI...")
        response = self.ai_manager.chat_completion(messages)

        # Step 4 & 5: If successful, save the reply and track tokens
        if response.status:
            self.conv_manager.add_assistant_message(session_id, response.reply)
            self.token_tracker.record(session_id, response.raw)

            return ChatResult(status=True, reply=response.reply)

        # If the AI call failed, return the error info
        return ChatResult(status=False, error=response.error)

    # ── Conversation History ───────────────────────────────────────────────

    def get_history(self, session_id):
        """Return the current message list for a session."""
        return self.conv_manager.get_messages(session_id)

    def get_message_count(self, session_id):
        """Return how many chat messages are in a session (excluding system prompt)."""
        msgs = self.conv_manager.get_messages(session_id)
        return max(len(msgs) - 1, 0)

    # ── Token Usage ────────────────────────────────────────────────────────

    def get_session_token_usage(self, session_id):
        """Return token usage for a specific session."""
        return self.token_tracker.get_session_usage(session_id)

    def get_user_token_usage(self, username):
        """Return total token usage for a user (across all sessions)."""
        return self.token_tracker.get_user_usage(username)
