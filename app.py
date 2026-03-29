"""
app.py  --  Command-line chat application.

This is the main file you run:   python app.py

Flow:
  1. Enter a username.
  2. Pick an AI provider  (e.g. Azure OpenAI).
  3. Pick a memory strategy  (simple / fixed window / summarizing).
  4. User menu  --  session controls for that user:
       new session, resume, session count, per-session tokens,
       switch user, back to main menu, quit.
  5. Main menu  --  admin-level overview:
       token usage per user, token usage per session,
       back to user menu, switch user, quit.

Switching users re-prompts for username + LLM + memory choices.
Each user only sees their own sessions in the user menu.
The main menu shows holistic usage across ALL users.
"""

import sys
import os
import textwrap
from config import logger
from modules.ChatOrchestrator import (
    ChatOrchestrator,
    AVAILABLE_AI_MANAGERS,
    AVAILABLE_MEMORY_MANAGERS,
)
from modules.TokenTracker import TokenTracker


# ── Display Helpers ────────────────────────────────────────────────────────

DIVIDER = "-" * 60


def print_header(title):
    """Print a section header with lines above and below."""
    print("\n" + DIVIDER)
    print("  " + title)
    print(DIVIDER)


def print_menu(options):
    """Print a numbered list of menu options."""
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    print()


def get_input(prompt="Choice: "):
    """Safely read user input.  Returns empty string on Ctrl+C."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def print_token_usage(label, usage):
    """Print token usage stats in a readable format."""
    print(f"  {label}")
    print(f"    Requests   : {usage.request_count}")
    print(f"    Prompt     : {usage.prompt_tokens:,} tokens")
    print(f"    Completion : {usage.completion_tokens:,} tokens")
    print(f"    Total      : {usage.total_tokens:,} tokens")


def print_history(messages):
    """Print the conversation history nicely formatted."""
    print_header("Conversation History")

    if not messages:
        print("  (empty)")
        return

    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]

        # Truncate very long messages for display
        if len(content) > 300:
            content = content[:300] + " ..."

        # Wrap text to fit the terminal
        wrapped = textwrap.fill(
            content, width=72,
            initial_indent="    ",
            subsequent_indent="    ",
        )
        print(f"\n  [{role}]")
        print(wrapped)

    print()


def choose_from_list(title, options_dict):
    """
    Show a numbered list built from a dictionary and return the
    selected value.  Returns None if the user cancels.
    """
    print_header(title)

    labels = list(options_dict.keys())
    values = list(options_dict.values())

    print_menu(labels)

    choice = get_input()

    if choice.isdigit() and 1 <= int(choice) <= len(labels):
        picked_label = labels[int(choice) - 1]
        picked_value = values[int(choice) - 1]
        print(f"  Selected: {picked_label}\n")
        return picked_value

    # If only one option exists, auto-select it
    if len(labels) == 1:
        print(f"  Auto-selected: {labels[0]}\n")
        return values[0]

    print("  Invalid choice.\n")
    return None


# ── Login Flow ─────────────────────────────────────────────────────────────
# Asks for username, LLM provider, and memory strategy.
# Returns (username, orchestrator) or (None, None) if cancelled.

def login_flow(shared_tracker):
    """
    Prompt the user for:
      1. Username
      2. AI provider
      3. Memory strategy

    Returns (username, orchestrator) on success, or (None, None) on cancel.
    """

    # Step 1: Username
    username = get_input("\n  Enter your username: ")
    if not username:
        print("  Username is required.")
        return None, None

    # Step 2: Pick AI provider
    ai_choice = choose_from_list(
        "Choose your AI provider",
        AVAILABLE_AI_MANAGERS,
    )
    if ai_choice is None:
        return None, None

    # Step 3: Pick memory strategy
    memory_choice = choose_from_list(
        "Choose how to manage conversation memory",
        AVAILABLE_MEMORY_MANAGERS,
    )
    if memory_choice is None:
        return None, None

    # Create the orchestrator with the shared tracker so admin view
    # can see data from all users across the lifetime of the app
    orch = ChatOrchestrator(
        ai_choice=ai_choice,
        memory_choice=memory_choice,
        shared_tracker=shared_tracker,
    )

    logger.info(
        "User '%s' logged in -- AI: %s, memory: %s",
        username, ai_choice, memory_choice,
    )

    return username, orch


# ── Session Loop ──────────────────────────────────────────────────────────
# Runs while the user is chatting inside one session.

def session_loop(username, session_id, orch):
    """Interactive chat loop for one session."""

    print_header("Session active  --  " + session_id[:8] + "...")
    print(f"  AI Provider : {orch.provider}")
    print(f"  Memory      : {orch.manager_label}")
    print(f"  User        : {username}\n")
    print("  Type a message to chat, or pick an option below.\n")

    session_menu = [
        "Chat with AI (just type your message)",
        "View conversation history",
        "Token usage for this session",
        "Back to user menu",
    ]

    while True:
        print_menu(session_menu)
        choice = get_input("Enter choice or message: ")

        if not choice:
            continue

        # Option 4: Back to user menu
        if choice == "4":
            print("\n  Returning to user menu...\n")
            return

        # Option 2: View history
        if choice == "2":
            print_history(orch.get_history(session_id))
            continue

        # Option 3: Token usage for this session
        if choice == "3":
            print_header("Session Token Usage")
            print_token_usage(
                "Session " + session_id[:8] + "...",
                orch.get_session_token_usage(session_id),
            )
            print()
            continue

        # Anything else is treated as a chat message
        if choice == "1":
            user_message = get_input("You: ")
        else:
            user_message = choice

        if not user_message:
            continue

        print("\n  Thinking...\n")
        result = orch.send_message(session_id, user_message)

        if result.status:
            wrapped = textwrap.fill(
                result.reply, width=72,
                initial_indent="  ",
                subsequent_indent="  ",
            )
            print(f"  [ASSISTANT]\n{wrapped}\n")
        else:
            print(f"  Error: {result.error or 'No response from AI.'}")
            print()


# ── User Menu ──────────────────────────────────────────────────────────────
# Per-user session controls.  Only shows THIS user's sessions and
# per-session token usage (no aggregated per-user totals).
#
# Returns: "main_menu", "switch_user", or "quit"

def user_menu(username, orch):
    """User-level menu.  Returns 'main_menu', 'switch_user', or 'quit'."""

    menu_options = [
        "New session",
        "Resume session",
        "Session count",
        "Token usage per session",
        "Switch user",
        "Main menu (admin view)",
        "Quit app",
    ]

    while True:
        print_header("User Menu  --  " + username)
        print(f"  AI Provider : {orch.provider}")
        print(f"  Memory      : {orch.manager_label}\n")
        print_menu(menu_options)
        choice = get_input()

        # ── 1. New session ─────────────────────────────────────────────────
        if choice == "1":
            session_id = orch.create_session(username)
            session_loop(username, session_id, orch)

        # ── 2. Resume session ──────────────────────────────────────────────
        elif choice == "2":
            sessions = orch.get_user_sessions(username)

            if not sessions:
                print("\n  No sessions yet -- start a new one first.\n")
                continue

            print_header("Your Sessions")
            for i, sid in enumerate(sessions, 1):
                msg_count = orch.get_message_count(sid)
                usage = orch.get_session_token_usage(sid)
                print(
                    f"  [{i}] {sid[:8]}...  "
                    f"({msg_count} messages, {usage.total_tokens:,} tokens)"
                )
            print()

            idx = get_input("Select session number: ")
            if idx.isdigit() and 1 <= int(idx) <= len(sessions):
                session_loop(username, sessions[int(idx) - 1], orch)
            else:
                print("  Invalid selection.\n")

        # ── 3. Session count ───────────────────────────────────────────────
        elif choice == "3":
            print_header("Session Count")
            count = orch.get_session_count(username)
            print(f"  Active sessions for {username}: {count}")
            print()

        # ── 4. Token usage per session (this user only) ────────────────────
        elif choice == "4":
            sessions = orch.get_user_sessions(username)
            print_header("Token Usage -- Your Sessions")

            if not sessions:
                print("  No sessions yet.\n")
                continue

            for sid in sessions:
                print_token_usage(
                    "Session " + sid[:8] + "...",
                    orch.get_session_token_usage(sid),
                )
            print()

        # ── 5. Switch user ─────────────────────────────────────────────────
        elif choice == "5":
            logger.info("User '%s' switching to another user.", username)
            print("\n  Switching user...\n")
            return "switch_user"

        # ── 6. Main menu (admin view) ──────────────────────────────────────
        elif choice == "6":
            return "main_menu"

        # ── 7. Quit app ───────────────────────────────────────────────────
        elif choice == "7":
            return "quit"

        else:
            print("  Invalid choice -- please enter a number from the menu.\n")


# ── Main Menu (Admin View) ────────────────────────────────────────────────
# Shows holistic token usage across ALL users and ALL sessions.
#
# Returns: "user_menu", "switch_user", or "quit"

def main_menu(shared_tracker):
    """Admin-level overview.  Returns 'user_menu', 'switch_user', or 'quit'."""

    menu_options = [
        "Token usage per user",
        "Token usage per session",
        "Back to user menu",
        "Switch user",
        "Quit app",
    ]

    while True:
        print_header("Main Menu  --  Admin Overview")
        print_menu(menu_options)
        choice = get_input()

        # ── 1. Token usage per user ────────────────────────────────────────
        if choice == "1":
            usernames = shared_tracker.get_all_usernames()
            print_header("Token Usage -- All Users")

            if not usernames:
                print("  No usage data yet.\n")
                continue

            for name in usernames:
                print_token_usage(name, shared_tracker.get_user_usage(name))
            print()

        # ── 2. Token usage per session ─────────────────────────────────────
        elif choice == "2":
            session_ids = shared_tracker.get_all_session_ids()
            print_header("Token Usage -- All Sessions")

            if not session_ids:
                print("  No sessions yet.\n")
                continue

            for sid in session_ids:
                owner = shared_tracker.get_username_for_session(sid)
                label = sid[:8] + "...  (" + (owner or "unknown") + ")"
                print_token_usage(label, shared_tracker.get_session_usage(sid))
            print()

        # ── 3. Back to user menu ───────────────────────────────────────────
        elif choice == "3":
            return "user_menu"

        # ── 4. Switch user ─────────────────────────────────────────────────
        elif choice == "4":
            print("\n  Switching user...\n")
            return "switch_user"

        # ── 5. Quit ───────────────────────────────────────────────────────
        elif choice == "5":
            return "quit"

        else:
            print("  Invalid choice -- please enter a number from the menu.\n")


# ── App Entry Point ───────────────────────────────────────────────────────

def main():
    """
    Entry point.

    Flow:
      1. Welcome screen.
      2. Login  (username + LLM + memory)  →  creates orchestrator.
      3. User menu  (session controls, per-session tokens).
         From here the user can:
           - "main_menu"   →  go to admin view (step 4)
           - "switch_user" →  go back to login (step 2)
           - "quit"        →  exit
      4. Main menu / admin view  (per-user + per-session tokens).
         From here the user can:
           - "user_menu"   →  go back to user menu (step 3)
           - "switch_user" →  go back to login (step 2)
           - "quit"        →  exit
    """

    print_header("Welcome to CodebaseExamples!")

    # One shared tracker lives for the entire app lifetime.
    # Every orchestrator (even across user switches) feeds into it,
    # so the admin view always has the full picture.
    shared_tracker = TokenTracker()

    # ── Login loop ─────────────────────────────────────────────────────────
    # We re-enter this loop whenever the user picks "switch user".

    while True:

        # Step 1: Login (username + LLM + memory)
        username, orch = login_flow(shared_tracker)
        if username is None:
            # Login was cancelled -- try again
            continue

        # Step 2: Navigation loop between user menu and main menu
        current_view = "user_menu"

        while True:

            # ── User Menu ──────────────────────────────────────────────────
            if current_view == "user_menu":
                result = user_menu(username, orch)

                if result == "main_menu":
                    current_view = "main_menu"
                elif result == "switch_user":
                    break  # break inner loop → re-enter login
                elif result == "quit":
                    print_header("Goodbye!")
                    return

            # ── Main Menu (Admin) ──────────────────────────────────────────
            elif current_view == "main_menu":
                result = main_menu(shared_tracker)

                if result == "user_menu":
                    current_view = "user_menu"
                elif result == "switch_user":
                    break  # break inner loop → re-enter login
                elif result == "quit":
                    print_header("Goodbye!")
                    return


# ── Run the app ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
