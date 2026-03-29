"""
modules  --  Core building blocks for the chat application.

This package contains the following modules:

  AIManagers.py          - Clients for Azure OpenAI and OpenAI APIs.
  ChatOrchestrator.py    - Central controller that wires AI, memory, and
                           token tracking together.
  ConversationManagers.py - Three strategies for managing chat history
                           (simple, fixed-window, summarizing).
  TokenTracker.py        - Tracks token usage per session and per user.
"""
