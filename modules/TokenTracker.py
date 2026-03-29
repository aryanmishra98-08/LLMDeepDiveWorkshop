"""
TokenTracker.py  --  Keeps track of how many tokens each API call uses.

Tracks token usage at two levels:
  - Per session  (each chat session has its own counter)
  - Per user     (totals across all of a user's sessions)

After every successful API call, call tracker.record(...) to update counts.
"""

from config import logger


# ── TokenUsage ─────────────────────────────────────────────────────────────
# A simple container that holds the running totals for one scope
# (either one session or one user).

class TokenUsage:

    def __init__(self):
        self.prompt_tokens = 0       # tokens sent to the model
        self.completion_tokens = 0   # tokens the model sent back
        self.total_tokens = 0        # prompt + completion
        self.request_count = 0       # how many API calls were made

    def add(self, prompt, completion, total):
        """Add the token counts from one API response."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.request_count += 1


# ── TokenTracker ───────────────────────────────────────────────────────────
# The main tracker that the rest of the app uses.

class TokenTracker:

    def __init__(self):
        # Stores a TokenUsage object for each session id
        self.session_usage = {}

        # Stores a TokenUsage object for each username
        self.user_usage = {}

        # Maps session_id -> username so we know which user owns a session
        self.session_to_user = {}

    # ── Setup ──────────────────────────────────────────────────────────────

    def register_session(self, session_id, username):
        """
        Link a session to a user.  Call this once when a new session
        is created so the tracker knows who owns it.
        """
        self.session_to_user[session_id] = username

        # Create empty usage objects if they don't exist yet
        if session_id not in self.session_usage:
            self.session_usage[session_id] = TokenUsage()

        if username not in self.user_usage:
            self.user_usage[username] = TokenUsage()

    # ── Recording ──────────────────────────────────────────────────────────

    def record(self, session_id, prompt_tokens, completion_tokens, total_tokens):
        """
        Add token counts from one API call to both the session and user totals.

        Token counts are passed directly as integers so this tracker stays
        decoupled from any SDK's response shape -- each AI manager extracts
        the counts itself and passes them here.
        """
        # Update session-level totals
        if session_id in self.session_usage:
            self.session_usage[session_id].add(prompt_tokens, completion_tokens, total_tokens)

        # Update user-level totals
        username = self.session_to_user.get(session_id)
        if username and username in self.user_usage:
            self.user_usage[username].add(prompt_tokens, completion_tokens, total_tokens)

        logger.debug(
            "Tokens recorded -- session=%s user=%s prompt=%d completion=%d total=%d",
            session_id, username, prompt_tokens, completion_tokens, total_tokens,
        )

    # ── Queries ────────────────────────────────────────────────────────────

    def get_session_usage(self, session_id):
        """Return the TokenUsage for a session (or an empty one)."""
        return self.session_usage.get(session_id, TokenUsage())

    def get_user_usage(self, username):
        """Return the TokenUsage for a user (or an empty one)."""
        return self.user_usage.get(username, TokenUsage())

    def get_all_usernames(self):
        """Return a list of all usernames that have been tracked."""
        return list(self.user_usage.keys())

    def get_all_session_ids(self):
        """Return a list of all session ids that have been tracked."""
        return list(self.session_usage.keys())

    def get_username_for_session(self, session_id):
        """Return the username that owns a session (or None)."""
        return self.session_to_user.get(session_id)
