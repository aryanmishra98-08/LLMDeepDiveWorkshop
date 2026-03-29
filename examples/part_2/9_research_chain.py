"""
Sequential 3-Step Research Chain:
Plan subtopics → research each in parallel → synthesize.
Demonstrates context-lean handoffs between steps.
"""

import asyncio
import json

from shared_config import AZURE_MODEL, TokenTracker, chat


async def research_chain(topic: str, model: str = None) -> dict:
    """3-step chain: generate subtopics → research each → synthesize."""
    model = model or AZURE_MODEL
    tracker = TokenTracker()
    results = {"steps": [], "errors": []}

    # Step 1: Identify key subtopics
    print(f"\n{'─' * 50}")
    print("⟳  Step 1/3 — Planning: generating key subtopics...")
    step1 = await chat([
        {"role": "system", "content": "You are a research planner."},
        {"role": "user", "content":
         f"List exactly 3 key subtopics for researching '{topic}'. "
         f"Return as JSON: [\"subtopic1\", \"subtopic2\", \"subtopic3\"]"}
    ], tracker, model=model)

    try:
        subtopics = json.loads(step1.strip().strip("```json").strip("```"))
        results["steps"].append({"step": "plan", "subtopics": subtopics})
        print(f"✅ Subtopics identified:")
        for i, s in enumerate(subtopics, 1):
            print(f"   {i}. {s}")
    except json.JSONDecodeError:
        print("⚠️  JSON parse failed — falling back to full topic as single subtopic")
        results["errors"].append({"step": "plan", "error": "JSON parse failed",
                                   "raw": step1})
        subtopics = [topic]
        results["steps"].append({"step": "plan", "subtopics": subtopics,
                                  "fallback": True})

    # Step 2: Research each subtopic (parallel)
    print(f"\n{'─' * 50}")
    print(f"⟳  Step 2/3 — Research: investigating {len(subtopics)} subtopics in parallel...")

    async def research_subtopic(sub: str) -> dict:
        print(f"   🔍 Researching: \"{sub}\"")
        try:
            content = await chat([
                {"role": "system", "content":
                 "Write a concise 2-3 sentence factual summary on this topic."},
                {"role": "user", "content": sub}
            ], tracker, model=model, max_tokens=256)
            print(f"   ✅ Done: \"{sub}\"")
            return {"subtopic": sub, "summary": content}
        except Exception as e:
            print(f"   ❌ Failed: \"{sub}\" — {e}")
            return {"subtopic": sub, "error": str(e)}

    research = await asyncio.gather(*[research_subtopic(s) for s in subtopics])
    results["steps"].append({"step": "research", "findings": research})
    successes = sum(1 for r in research if "summary" in r)
    print(f"📥 Research complete — {successes}/{len(subtopics)} subtopics succeeded")

    # Step 3: Synthesize — pass ONLY summaries, not full context
    print(f"\n{'─' * 50}")
    print("⟳  Step 3/3 — Synthesis: combining findings into a unified briefing...")
    context = "\n\n".join(
        f"## {r['subtopic']}\n{r.get('summary', 'Research failed.')}"
        for r in research
    )
    synthesis = await chat([
        {"role": "system", "content":
         "Synthesize the research below into a coherent 1-paragraph briefing."},
        {"role": "user", "content": context}
    ], tracker, model=model, max_tokens=512)

    results["steps"].append({"step": "synthesize", "output": synthesis})
    results["usage"] = tracker.report()
    print("✅ Synthesis complete")
    return results

async def main():
    topic = "Impact of transformer architecture on NLP"

    print("=" * 60)
    print("SEQUENTIAL RESEARCH CHAIN — Plan → Research → Synthesize")
    print("=" * 60)
    print(f"\n📌 Topic: {topic}")
    print("\nApproach: 3-step chain — generate subtopics, research each in "
          "parallel, then synthesize into a briefing.")

    result = await research_chain(topic)

    print(f"\n{'═' * 60}")
    print("CHAIN OUTPUT")
    print(f"{'═' * 60}")
    for step in result["steps"]:
        print(f"\n[{step['step'].upper()}]")
        if "output" in step:
            print(step["output"])
        elif "subtopics" in step:
            print(step["subtopics"])
    if result.get("errors"):
        print(f"\n⚠️  Errors encountered: {result['errors']}")
    print(f"\n📊 {result['usage']}")

asyncio.run(main())
