"""
Branching Chain — Classify → Route → Handle:
Classify a message, then route to a specialized handler based on category.
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from provider_config import init_async_client, TokenTracker, chat, ping  # noqa: E402
_, MODEL, _ = init_async_client()


HANDLERS = {
    "billing": "You are a billing specialist. Be precise about amounts and dates. "
               "Offer specific solutions for billing disputes.",
    "technical": "You are a senior technical support engineer. Ask diagnostic "
                 "questions, provide step-by-step troubleshooting instructions.",
    "general": "You are a friendly customer service agent. Answer questions "
               "clearly and offer to escalate if needed.",
}

async def branching_chain(message: str, model: str = None) -> dict:
    """Classify a customer message, then route to specialized handler."""
    model = model or MODEL
    tracker = TokenTracker()

    # Step 1: Classify
    print(f"\n{'─' * 50}")
    print("⟳  Step 1/2 — Classification: determining message category...")
    classification = await chat([
        {"role": "system", "content":
         "Classify the customer message into exactly one category. "
         "Reply with ONLY the category name, nothing else.\n"
         "Categories: billing, technical, general"},
        {"role": "user", "content": message}
    ], tracker, model=model, max_tokens=20)

    category = classification.strip().lower()
    if category not in HANDLERS:
        print(f"⚠️  Unexpected category '{category}' — falling back to 'general'")
        category = "general"  # Fallback for unexpected classifications
    else:
        print(f"🏷️  Classified as: {category!r}")

    # Step 2: Route to specialized handler
    print(f"\n{'─' * 50}")
    print(f"⟳  Step 2/2 — Routing: handing off to {category} specialist...")
    print(f"   System prompt: \"{HANDLERS[category][:80]}...\"")
    response = await chat([
        {"role": "system", "content": HANDLERS[category]},
        {"role": "user", "content": message}
    ], tracker, model=model, max_tokens=512)
    print("✅ Handler response received")

    return {
        "category": category,
        "response": response,
        "usage": tracker.report(),
    }

async def main():
    await ping()
    message = "I was charged twice for my subscription last month and I want a refund."

    print("=" * 60)
    print("BRANCHING CHAIN — Classify → Route → Handle")
    print("=" * 60)
    print(f"\n📨 Customer message: \"{message}\"")
    print(f"\nAvailable handlers: {list(HANDLERS.keys())}")

    result = await branching_chain(message)

    print(f"\n{'═' * 60}")
    print("RESULT")
    print(f"{'═' * 60}")
    print(f"🏷️  Routed to : {result['category']} handler")
    print(f"\n💬 Response:\n{result['response']}")
    print(f"\n📊 {result['usage']}")

asyncio.run(main())
