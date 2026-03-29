"""
Self-Critique Loop (Reflexion):
Generate → critique → reflect on failure → regenerate with verbal reflection memory.
"""

import asyncio

from shared_config import AZURE_MODEL, TokenTracker, chat


async def self_critique_loop(task: str, max_attempts: int = 3,
                             model: str = None) -> dict:
    """Generate → Critique → Regenerate with verbal reflection memory."""
    model = model or AZURE_MODEL
    tracker = TokenTracker()
    reflections = []

    for attempt in range(1, max_attempts + 1):
        print(f"\n{'─' * 50}")
        print(f"⟳  Attempt {attempt}/{max_attempts} — Generating output...")
        if reflections:
            print(f"   💡 Applying {len(reflections)} lesson(s) from previous attempts:")
            for r in reflections:
                print(f"      • Attempt {r['attempt']}: {r['reflection']}")

        # Build context with previous reflections
        reflection_ctx = ""
        if reflections:
            reflection_ctx = (
                "\n\nPrevious attempts failed. Lessons learned:\n" +
                "\n".join(f"- Attempt {r['attempt']}: {r['reflection']}"
                          for r in reflections)
            )

        # Generate
        output = await chat([
            {"role": "system", "content":
             "You are a skilled writer." + reflection_ctx},
            {"role": "user", "content": task}
        ], tracker, model=model, max_tokens=512)
        print(f"📝 Output generated ({len(output.split())} words)")

        # Critique
        print(f"⟳  Critiquing output (scoring on accuracy, completeness, clarity, tone)...")
        critique = await chat([
            {"role": "system", "content":
             "Evaluate this output ruthlessly on: accuracy, completeness, "
             "clarity, and tone. Score 1-10. If score >= 8, start with 'PASS'. "
             "Otherwise start with 'FAIL' and explain what must improve."},
            {"role": "user", "content": f"Task: {task}\n\nOutput:\n{output}"}
        ], tracker, model=model, max_tokens=300)

        passed = critique.strip().upper().startswith("PASS")
        verdict = "✅ PASS" if passed else "❌ FAIL"
        first_line = critique.split("\n")[0][:80]
        print(f"{verdict} — Critique: \"{first_line}\"")

        if passed:
            return {"output": output, "attempts": attempt, "passed": True,
                    "final_critique": critique, "reflections": reflections,
                    "usage": tracker.report()}

        # Reflect on failure
        print(f"⟳  Extracting lesson from failure for next attempt...")
        reflection = await chat([
            {"role": "system", "content":
             "Based on the critique, write a ONE-SENTENCE lesson about what "
             "to do differently next time. Be specific and actionable."},
            {"role": "user", "content": critique}
        ], tracker, model=model, max_tokens=100)

        lesson = reflection.strip()
        reflections.append({"attempt": attempt, "reflection": lesson})
        print(f"💡 Lesson learned: \"{lesson}\"")

    print(f"\n⚠️  Max attempts ({max_attempts}) reached without passing critique")
    return {"output": output, "attempts": max_attempts, "passed": False,
            "reflections": reflections, "usage": tracker.report()}


async def main():
    task = (
        "Write a professional email to a client explaining a 2-week project delay "
        "due to unexpected technical challenges. Be honest but reassuring."
    )

    print("=" * 60)
    print("SELF-CRITIQUE LOOP (REFLEXION) — Generate → Critique → Reflect → Retry")
    print("=" * 60)
    print(f"\n📋 Task: {task}")
    print(f"\nStrategy: Generate output, have a ruthless critic score it (1-10). "
          f"If score < 8, extract a lesson and regenerate. Up to {3} attempts.")

    result = await self_critique_loop(task, max_attempts=3)

    print(f"\n{'═' * 60}")
    print("RESULT")
    print(f"{'═' * 60}")
    print(f"📊 Passed: {'✅ Yes' if result['passed'] else '❌ No'} | "
          f"Attempts: {result['attempts']}")
    if result['reflections']:
        print(f"\n💡 Reflections accumulated ({len(result['reflections'])}):")
        for r in result['reflections']:
            print(f"   Attempt {r['attempt']}: {r['reflection']}")
    print(f"\n📝 Final output:\n{result['output']}")
    print(f"\n📊 {result['usage']}")

asyncio.run(main())
