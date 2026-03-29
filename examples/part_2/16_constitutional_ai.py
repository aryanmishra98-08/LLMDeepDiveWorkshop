"""
Constitutional AI — Principles-Based Revision:
Generate → check output against a set of principles → revise to comply with any violations.
"""

import asyncio

from shared_config import AZURE_MODEL, TokenTracker, chat


async def constitutional_generate(task: str, principles: list[str],
                                   model: str = None) -> dict:
    """Generate → check against constitution → revise."""
    model = model or AZURE_MODEL
    tracker = TokenTracker()

    # Generate initial response
    print(f"\n{'─' * 50}")
    print("⟳  Step 1/3 — Generating initial response (unconstrained)...")
    draft = await chat([
        {"role": "user", "content": task}
    ], tracker, model=model, max_tokens=512)
    print(f"📝 Draft generated ({len(draft.split())} words)")

    # Check against each principle
    print(f"\n{'─' * 50}")
    print(f"⟳  Step 2/3 — Checking draft against {len(principles)} constitutional principles...")
    for i, p in enumerate(principles, 1):
        print(f"   {i}. {p}")
    principle_block = "\n".join(f"{i+1}. {p}" for i, p in enumerate(principles))
    critique = await chat([
        {"role": "system", "content":
         f"Check the response against these principles:\n{principle_block}\n\n"
         f"For each principle, state PASS or FAIL with a brief explanation. "
         f"If all pass, start with 'ALL PASS'."},
        {"role": "user", "content": f"Response to evaluate:\n{draft}"}
    ], tracker, model=model, max_tokens=400)

    all_passed = "ALL PASS" in critique.upper()
    if all_passed:
        print("✅ All principles passed — no revision needed")
        return {"output": draft, "revised": False, "usage": tracker.report()}

    # Show which principles were flagged
    print("⚠️  One or more principles were violated:")
    for line in critique.split("\n"):
        if "FAIL" in line.upper():
            print(f"   ❌ {line.strip()}")
        elif "PASS" in line.upper():
            print(f"   ✅ {line.strip()}")

    # Revise to comply with constitution
    print(f"\n{'─' * 50}")
    print("⟳  Step 3/3 — Revising response to comply with all principles...")
    revised = await chat([
        {"role": "system", "content":
         f"Revise this response to comply with ALL principles:\n{principle_block}"},
        {"role": "user", "content":
         f"Original: {draft}\n\nViolations found: {critique}\n\n"
         f"Rewrite to fix all violations while preserving useful content."}
    ], tracker, model=model, max_tokens=512)
    print(f"✅ Revision complete ({len(revised.split())} words)")

    return {"output": revised, "revised": True, "critique": critique,
            "usage": tracker.report()}

# Usage: politically neutral news summarizer
async def main():
    task = "Summarize the debate around AI regulation in the US."
    principles = [
        "Present all political perspectives fairly without favoring any side",
        "Use neutral, non-inflammatory language throughout",
        "Include specific facts and dates, not vague generalizations",
        "Do not editorialize or include personal opinions",
    ]

    print("=" * 60)
    print("CONSTITUTIONAL AI — Generate → Audit → Revise")
    print("=" * 60)
    print(f"\n📋 Task: {task}")
    print(f"\n📜 Constitutional principles ({len(principles)}):")
    for i, p in enumerate(principles, 1):
        print(f"   {i}. {p}")
    print("\nStrategy: Generate a draft freely, then audit it against each "
          "principle, and revise only if violations are found.")

    result = await constitutional_generate(task=task, principles=principles)

    print(f"\n{'═' * 60}")
    print("RESULT")
    print(f"{'═' * 60}")
    print(f"✏️  Revised: {'Yes — violations were corrected' if result['revised'] else 'No — draft passed all principles'}")
    if result.get("critique") and result["revised"]:
        print(f"\n🔍 Audit findings:\n{result['critique']}")
    print(f"\n📝 Final output:\n{result['output']}")
    print(f"\n📊 {result['usage']}")

asyncio.run(main())
