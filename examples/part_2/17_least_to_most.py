"""
Least-to-Most Prompting:
Decompose a problem into sub-problems from simplest to hardest,
solve each incrementally using prior solutions as context.
"""

import asyncio
import json

from shared_config import AZURE_MODEL, TokenTracker, chat


async def least_to_most(problem: str, model: str = None) -> dict:
    """Decompose into sub-problems (simple→complex), solve incrementally."""
    model = model or AZURE_MODEL
    tracker = TokenTracker()

    # Step 1: Decompose
    print(f"\n{'─' * 50}")
    print("⟳  Step 1/2 — Decomposing problem into ordered sub-problems "
          "(simplest → most complex)...")
    decomposition = await chat([
        {"role": "system", "content":
         "Break this problem into 2-4 sub-problems, ordered from simplest to "
         "most complex. Each sub-problem should build on previous ones. "
         "Return as JSON list: [\"sub1\", \"sub2\", ...]"},
        {"role": "user", "content": problem}
    ], tracker, model=model, max_tokens=2048)

    try:
        subproblems = json.loads(
            decomposition.strip().strip("```json").strip("```"))
        print(f"✅ Decomposed into {len(subproblems)} sub-problems:")
        for i, s in enumerate(subproblems, 1):
            print(f"   {i}. {s}")
    except json.JSONDecodeError:
        print("⚠️  JSON parse failed — treating full problem as single sub-problem")
        subproblems = [problem]

    # Step 2: Solve incrementally
    print(f"\n{'─' * 50}")
    print(f"⟳  Step 2/2 — Solving sub-problems incrementally "
          f"(each builds on prior solutions)...")
    solutions = []
    for idx, sub in enumerate(subproblems, 1):
        print(f"\n   🔧 Sub-problem {idx}/{len(subproblems)}: \"{sub[:80]}"
              f"{'...' if len(sub) > 80 else ''}\"")
        if solutions:
            print(f"      ↳ Using {len(solutions)} prior solution(s) as context")
        prior = "\n".join(
            f"Q: {s['question']}\nA: {s['answer']}" for s in solutions
        )
        context = f"Previously solved:\n{prior}\n\n" if prior else ""
        answer = await chat([
            {"role": "system", "content":
             "Solve this sub-problem. Use solutions to previous sub-problems "
             "if they're relevant. Be concise."},
            {"role": "user", "content": f"{context}Now solve: {sub}"}
        ], tracker, model=model, max_tokens=2048)
        solutions.append({"question": sub, "answer": answer.strip()})
        print(f"   ✅ Sub-problem {idx} solved ({len(answer.split())} words)")

    return {"subproblems": subproblems, "solutions": solutions,
            "final_answer": solutions[-1]["answer"] if solutions else None,
            "usage": tracker.report()}


async def main():
    problem = (
        "Design a recommendation system for an e-commerce platform that "
        "handles 1 million users and 500,000 products, supports real-time "
        "updates, and provides personalized recommendations."
    )

    print("=" * 60)
    print("LEAST-TO-MOST PROMPTING — Decompose → Solve Incrementally")
    print("=" * 60)
    print(f"\n❓ Problem: {problem}")
    print("\nStrategy: Break the problem into sub-problems ordered easiest→hardest, "
          "then solve each one using all prior solutions as context.")

    result = await least_to_most(problem)

    print(f"\n{'═' * 60}")
    print("SUB-PROBLEMS & SOLUTIONS")
    print(f"{'═' * 60}")
    for i, sol in enumerate(result["solutions"], 1):
        print(f"\n{'─' * 50}")
        print(f"Sub-problem {i}: {sol['question']}")
        print(f"\nSolution:\n{sol['answer']}")
    print(f"\n{'═' * 60}")
    print(f"✅ FINAL ANSWER (Sub-problem {len(result['solutions'])}):\n"
          f"{result['final_answer']}")
    print(f"\n📊 {result['usage']}")

asyncio.run(main())
