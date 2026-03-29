"""
Iterative Refinement Chain:
Draft → critique → revise loop with quality score-based early stopping.
"""

import asyncio

from shared_config import AZURE_MODEL, TokenTracker, chat


async def iterative_refine(task: str, max_iterations: int = 3,
                           model: str = None) -> dict:
    """Draft → Critique → Revise loop with quality-based termination."""
    model = model or AZURE_MODEL
    tracker = TokenTracker()
    history = []

    # Step 1: Initial draft
    print(f"\n{'─' * 50}")
    print("⟳  Step 1 — Generating initial draft...")
    draft = await chat([
        {"role": "system", "content": "You are a skilled writer."},
        {"role": "user", "content": f"Write a first draft:\n{task}"}
    ], tracker, model=model, max_tokens=512)
    history.append({"iteration": 0, "type": "draft", "content": draft})
    print(f"📝 Draft complete ({len(draft.split())} words)")

    for i in range(1, max_iterations + 1):
        print(f"\n{'─' * 50}")
        print(f"⟳  Iteration {i}/{max_iterations} — Critiquing draft...")

        # Step 2: Critique
        critique = await chat([
            {"role": "system", "content":
             "You are a tough but constructive editor. Critique the draft below. "
             "List 2-3 specific improvements. Rate overall quality 1-10. "
             "If quality >= 8, say 'APPROVED' on the first line."},
            {"role": "user", "content": f"Draft to critique:\n\n{draft}"}
        ], tracker, model=model, max_tokens=300)
        history.append({"iteration": i, "type": "critique", "content": critique})

        # Check if quality threshold met
        if "APPROVED" in critique.upper().split("\n")[0]:
            print(f"✅ Draft APPROVED by editor on iteration {i}!")
            return {"final": draft, "iterations": i, "approved": True,
                    "history": history, "usage": tracker.report()}

        first_line = critique.split("\n")[0][:80]
        print(f"📋 Critique (not approved): \"{first_line}...\"")
        print(f"⟳  Revising draft based on editor feedback...")

        # Step 3: Revise based on critique
        draft = await chat([
            {"role": "system", "content": "You are a skilled writer revising your work."},
            {"role": "user", "content":
             f"Original draft:\n{draft}\n\n"
             f"Editor feedback:\n{critique}\n\n"
             f"Write an improved version addressing ALL feedback points."}
        ], tracker, model=model, max_tokens=512)
        history.append({"iteration": i, "type": "revision", "content": draft})
        print(f"📝 Revision {i} complete ({len(draft.split())} words)")

    print(f"\n⚠️  Max iterations ({max_iterations}) reached without approval")
    return {"final": draft, "iterations": max_iterations, "approved": False,
            "history": history, "usage": tracker.report()}

async def main():
    task = (
        "Write a 100-word product description for a noise-cancelling "
        "headphone aimed at remote workers. Emphasize comfort and battery life."
    )

    print("=" * 60)
    print("ITERATIVE REFINEMENT CHAIN — Draft → Critique → Revise")
    print("=" * 60)
    print(f"\n📋 Task: {task}")
    print(f"\nStrategy: Generate a draft, have an editor critique it, "
          f"then revise. Repeat up to {3} times or until APPROVED.")

    result = await iterative_refine(task, max_iterations=3)

    print(f"\n{'═' * 60}")
    print("RESULT")
    print(f"{'═' * 60}")
    print(f"📊 Iterations : {result['iterations']} | "
          f"Approved: {'✅ Yes' if result['approved'] else '❌ No (max reached)'}")
    history_summary = [(h["type"], len(h["content"].split())) for h in result["history"]]
    print(f"📜 History    : {history_summary}")
    print(f"\n📝 Final output:\n{result['final']}")
    print(f"\n📊 {result['usage']}")

asyncio.run(main())
