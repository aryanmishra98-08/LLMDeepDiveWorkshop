"""
Self-Consistency Solver with Majority Voting:
Parallel sampling across N reasoning paths with majority vote aggregation.
"""

import asyncio
import re
from collections import Counter

from shared_config import AZURE_MODEL, TokenTracker, chat


async def self_consistent_solve(question: str, n_paths: int = 5,
                                 model: str = None) -> dict:
    """Solve a math/logic problem using self-consistency (N paths + vote)."""
    model = model or AZURE_MODEL
    tracker = TokenTracker()
    system = (
        "Solve this math problem step by step. "
        "End your solution with EXACTLY these two lines:\n"
        "Meeting time: <HH:MM AM or PM>\n"
        "Distance from City A: <number> km"
    )

    path_num = 0

    async def single_path() -> str | None:
        nonlocal path_num
        path_num += 1
        current = path_num
        print(f"   🔀 Path {current}/{n_paths} — sampling independent reasoning...")
        result = await chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": question}],
            tracker, model=model, max_tokens=512
        )
        # Show the full reasoning for this path
        print(f"\n   📄 Path {current} reasoning:")
        for line in result.strip().split("\n"):
            print(f"      {line}")

        # Extract both parts of the answer
        time_m = re.search(r"Meeting time:\s*(\d+:\d+\s*[AP]M)", result, re.IGNORECASE)
        dist_m = re.search(r"Distance from City A:\s*(\d+\.?\d*)\s*km", result, re.IGNORECASE)
        if time_m and dist_m:
            answer = f"{time_m.group(1).strip()} | {dist_m.group(1).strip()} km from City A"
            print(f"   ✅ Path {current} extracted answer: {answer}")
        else:
            answer = None
            missing = []
            if not time_m: missing.append("meeting time")
            if not dist_m: missing.append("distance")
            print(f"   ⚠️  Path {current} — missing {' and '.join(missing)}, discarding")
        return answer

    # Sample N paths in parallel
    print(f"\n{'─' * 50}")
    print(f"⟳  Sampling {n_paths} independent reasoning paths in parallel...")
    tasks = [single_path() for _ in range(n_paths)]
    answers = await asyncio.gather(*tasks)
    valid = [a for a in answers if a is not None]

    print(f"\n📊 {len(valid)}/{n_paths} paths produced a valid answer")

    if not valid:
        print("❌ No valid answers extracted — unable to determine result")
        return {"answer": None, "error": "no_valid_answers",
                "usage": tracker.report()}

    # Majority vote
    vote_counts = Counter(valid)
    winner, win_count = vote_counts.most_common(1)[0]
    confidence = win_count / len(valid)

    print(f"\n🗳️  Majority vote across {len(valid)} valid paths:")
    for ans, count in sorted(vote_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"   {ans:>10}  {bar}  ({count} vote{'s' if count != 1 else ''})")
    print(f"\n🏆 Winner: {winner!r} — {win_count}/{len(valid)} votes "
          f"({confidence:.0%} confidence)")

    return {
        "answer": winner,
        "confidence": confidence,
        "vote_distribution": dict(vote_counts),
        "valid_paths": len(valid),
        "total_paths": n_paths,
        "usage": tracker.report(),
    }

async def main():
    question = (
        "A train leaves City A at 6:00 AM travelling at 90 km/h toward City B. "
        "A second train leaves City B at 7:30 AM travelling toward City A at 120 km/h. "
        "The distance between the two cities is 450 km. "
        "At what time do the two trains meet, and how far from City A is the meeting point?"
    )
    n_paths = 7

    print("=" * 60)
    print("SELF-CONSISTENCY SOLVER — Majority Voting")
    print("=" * 60)
    print(f"\n❓ Question: {question}")
    print(f"\nStrategy: Sample {n_paths} independent reasoning paths, "
          f"then take the majority vote.")

    result = await self_consistent_solve(question, n_paths=n_paths)

    print(f"\n{'═' * 60}")
    print("RESULT")
    print(f"{'═' * 60}")
    print(f"✅ Final answer : {result['answer']}")
    print(f"📈 Confidence   : {result['confidence']:.0%} "
          f"({result['valid_paths']}/{result['total_paths']} valid paths)")
    print(f"🗳️  Vote spread  : {result['vote_distribution']}")
    print(f"\n📊 {result['usage']}")

asyncio.run(main())
