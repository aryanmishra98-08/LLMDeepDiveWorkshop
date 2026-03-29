"""
Simplified Tree of Thoughts:
Generate multiple next-step approaches → evaluate which is most promising
→ continue down the best branch.
"""

import asyncio
import re

from shared_config import AZURE_MODEL, TokenTracker, chat


async def tree_of_thoughts(problem: str, breadth: int = 3, depth: int = 3,
                           model: str = None) -> dict:
    """Simplified ToT: generate → evaluate → select best → continue."""
    model = model or AZURE_MODEL
    tracker = TokenTracker()

    async def generate_thoughts(context: str, n: int) -> list[str]:
        result = await chat([
            {"role": "system", "content":
             "Generate exactly {n} distinct next-step approaches to solve this "
             "problem. Number them 1-{n}. Each should be a different strategy.".format(n=n)},
            {"role": "user", "content": context}
        ], tracker, model=model, max_tokens=600)
        thoughts = re.split(r'\n\d+[\.\)]\s*', "\n" + result)[1:]
        return [t.strip() for t in thoughts if t.strip()][:n]

    async def evaluate_thoughts(problem: str, thoughts: list[str]) -> int:
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(thoughts))
        result = await chat([
            {"role": "system", "content":
             "Evaluate which approach is most promising for solving the problem. "
             "Reply with ONLY the number (1, 2, or 3) of the best approach."},
            {"role": "user", "content":
             f"Problem: {problem}\n\nApproaches:\n{numbered}"}
        ], tracker, model=model, max_tokens=10)
        try:
            idx = int(re.search(r'\d+', result).group()) - 1
            return max(0, min(idx, len(thoughts) - 1))
        except (ValueError, AttributeError):
            return 0

    path = [f"Problem: {problem}"]
    for step in range(depth):
        print(f"\n{'─' * 50}")
        print(f"⟳  Depth {step + 1}/{depth} — Generating {breadth} candidate approaches...")
        context = "\n".join(path)
        thoughts = await generate_thoughts(context, breadth)
        if not thoughts:
            print("⚠️  No thoughts generated — stopping early")
            break
        for i, t in enumerate(thoughts, 1):
            preview = t[:80].replace("\n", " ")
            print(f"   {i}. {preview}{'...' if len(t) > 80 else ''}")
        print(f"\n⚙️  Evaluating {len(thoughts)} approaches — selecting best...")
        best_idx = await evaluate_thoughts(problem, thoughts)
        chosen = thoughts[best_idx]
        print(f"✅ Selected approach {best_idx + 1}: "
              f"\"{chosen[:70].replace(chr(10), ' ')}{'...' if len(chosen) > 70 else ''}\"")
        path.append(f"Step {step + 1}: {chosen}")

    # Final synthesis
    print(f"\n{'─' * 50}")
    print(f"⟳  Synthesizing reasoning path ({len(path)} steps) into final answer...")
    final = await chat([
        {"role": "system", "content": "Given the reasoning path, provide the final answer."},
        {"role": "user", "content": "\n".join(path)}
    ], tracker, model=model, max_tokens=256)
    print("✅ Final answer synthesized")

    return {"answer": final, "reasoning_path": path, "usage": tracker.report()}


async def main():
    problem = (
        "Design a system to detect and prevent credit card fraud in real-time "
        "for an e-commerce platform processing 10,000 transactions per minute."
    )
    breadth = 3
    depth = 3

    print("=" * 60)
    print("TREE OF THOUGHTS — Generate → Evaluate → Expand Best Branch")
    print("=" * 60)
    print(f"\n❓ Problem: {problem}")
    print(f"\nStrategy: At each depth level, generate {breadth} candidate "
          f"approaches, evaluate them, and expand the most promising one. "
          f"Repeat for {depth} levels.")

    result = await tree_of_thoughts(problem, breadth=breadth, depth=depth)

    print(f"\n{'═' * 60}")
    print("REASONING PATH")
    print(f"{'═' * 60}")
    for i, step in enumerate(result["reasoning_path"]):
        prefix = "🌱" if i == 0 else f"🌿 Step {i}"
        print(f"\n{prefix}:\n   {step}")
    print(f"\n{'─' * 60}")
    print(f"\n✅ FINAL ANSWER:\n{result['answer']}")
    print(f"\n📊 {result['usage']}")

asyncio.run(main())
