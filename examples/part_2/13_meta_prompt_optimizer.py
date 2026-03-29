"""
Meta-Prompt Optimizer:
Iteratively improve a prompt over N rounds using failure examples as context.
"""

import asyncio
from typing import Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from provider_config import init_async_client, TokenTracker, chat, ping  # noqa: E402
_, MODEL, _ = init_async_client()


async def optimize_prompt(original_prompt: str, task_description: str,
                          example_failures: Optional[list[str]] = None,
                          rounds: int = 3,
                          model: str = None) -> dict:
    """Use an LLM to iteratively improve a prompt through N rounds."""
    model = model or MODEL
    tracker = TokenTracker()
    history = [{"round": 0, "prompt": original_prompt}]
    current = original_prompt

    print(f"\n{'─' * 50}")
    print(f"📋 Starting prompt: \"{current}\"")
    if example_failures:
        print(f"⚠️  Failure examples provided ({len(example_failures)}):")
        for f in example_failures:
            print(f"   - {f}")

    for r in range(1, rounds + 1):
        print(f"\n{'─' * 50}")
        print(f"⟳  Round {r}/{rounds} — LLM analyzing prompt weaknesses and improving...")
        failure_ctx = ""
        if example_failures:
            failure_ctx = (
                "\n\nExamples where the current prompt fails or underperforms:\n" +
                "\n".join(f"- {f}" for f in example_failures)
            )

        improved = await chat([
            {"role": "system", "content":
             "You are an expert prompt engineer. Your job is to improve prompts "
             "for LLM tasks. Analyze the current prompt, identify weaknesses, "
             "and produce a strictly better version. Output ONLY the improved "
             "prompt text, nothing else."},
            {"role": "user", "content":
             f"Task description: {task_description}\n\n"
             f"Current prompt (round {r}/{rounds}):\n"
             f"---\n{current}\n---\n"
             f"{failure_ctx}\n\n"
             f"Improve this prompt. Make it clearer, more specific, and more "
             f"likely to produce correct outputs. Add constraints, examples, "
             f"or structure as needed."}
        ], tracker, model=model, max_tokens=800)

        current = improved.strip()
        history.append({"round": r, "prompt": current})
        preview = current[:100].replace("\n", " ")
        print(f"✅ Round {r} improved prompt: \"{preview}{'...' if len(current) > 100 else ''}\"")
        print(f"   ({len(current.split())} words)")

    return {"original": original_prompt, "optimized": current,
            "rounds": rounds, "history": history, "usage": tracker.report()}

async def main():
    await ping()
    original = "Summarize the text."
    task_desc = ("Summarize technical documents into 2-3 bullet points "
                 "for executive stakeholders who need actionable insights.")
    failures = [
        "Output was 10 paragraphs long instead of bullet points",
        "Summary included jargon the executive wouldn't understand",
        "Missed the key financial impact mentioned in the document",
    ]
    rounds = 3

    print("=" * 60)
    print("META-PROMPT OPTIMIZER — Iterative Prompt Improvement")
    print("=" * 60)
    print(f"\n🎯 Task: {task_desc}")
    print(f"\nStrategy: Run {rounds} improvement rounds, each using "
          f"known failure examples to guide the LLM prompt engineer.")

    result = await optimize_prompt(
        original_prompt=original,
        task_description=task_desc,
        example_failures=failures,
        rounds=rounds,
    )

    print(f"\n{'═' * 60}")
    print("OPTIMIZATION RESULT")
    print(f"{'═' * 60}")
    print(f"\n📋 ORIGINAL prompt:\n   {result['original']}")
    print(f"\n{'─' * 60}")
    print(f"\n✅ OPTIMIZED prompt (after {result['rounds']} rounds):")
    print(f"   {result['optimized']}")
    print(f"\n📊 {result['usage']}")

asyncio.run(main())
