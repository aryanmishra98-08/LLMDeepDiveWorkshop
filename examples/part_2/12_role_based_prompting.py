"""
Role-Based Prompting Patterns:
Three production-ready role patterns: expert consultant,
audience-adaptive explainer, and multi-persona debate.
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from provider_config import init_async_client, TokenTracker, chat, ping  # noqa: E402
_, MODEL, _ = init_async_client()


# Pattern 1: Expert consultant (domain behavior shaping)
SECURITY_REVIEWER = {
    "role": "system",
    "content": (
        "You are a senior application security engineer performing code review. "
        "For each code snippet: (1) identify vulnerabilities with CWE numbers, "
        "(2) rate severity as Critical/High/Medium/Low, (3) provide the exact "
        "fix with corrected code. Never say code is 'fine' — always find at "
        "least one improvement, even if it's a hardening suggestion."
    )
}

# Pattern 2: Audience-adaptive explainer
async def explain_for_audience(concept: str, audience: str,
                                model: str = None) -> str:
    model = model or MODEL
    tracker = TokenTracker()
    print(f"\n{'─' * 50}")
    print(f"⟳  Generating explanation for audience: {audience!r}...")
    result = await chat([
        {"role": "system", "content":
         f"Explain concepts for a {audience}. Match their vocabulary level, "
         f"use analogies they'd understand, and focus on what matters to them. "
         f"Keep explanations under 150 words."},
        {"role": "user", "content": f"Explain: {concept}"}
    ], tracker, model=model)
    print(f"✅ Explanation ready ({len(result.split())} words)")
    return result

# Pattern 3: Multi-persona debate for decision analysis
async def multi_persona_debate(question: str,
                                model: str = None) -> dict:
    """Three personas argue different angles, then a synthesizer concludes."""
    model = model or MODEL
    tracker = TokenTracker()
    personas = [
        ("Optimist", "You argue FOR the proposal. Find every benefit and upside. "
                     "Be specific with evidence and projected outcomes."),
        ("Critic", "You argue AGAINST the proposal. Identify risks, costs, "
                   "and failure modes. Be specific about what could go wrong."),
        ("Pragmatist", "You evaluate trade-offs neutrally. Identify conditions "
                       "under which the proposal succeeds or fails."),
    ]

    print(f"\n{'─' * 50}")
    print(f"⟳  Launching {len(personas)} personas in parallel: "
          f"{[p[0] for p in personas]}...")

    # Run all personas in parallel
    async def get_perspective(name: str, prompt: str) -> tuple[str, str]:
        print(f"   🎭 Persona '{name}' is formulating its argument...")
        result = await chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ], tracker, model=model, max_tokens=2048)
        print(f"   ✅ Persona '{name}' argument complete ({len(result.split())} words)")
        return name, result

    results = await asyncio.gather(
        *[get_perspective(n, p) for n, p in personas]
    )
    arguments = {name: text for name, text in results}

    # Synthesize
    print(f"\n{'─' * 50}")
    print("⟳  Synthesizing all perspectives into a balanced recommendation...")
    debate_context = "\n\n".join(
        f"**{name}**: {text}" for name, text in arguments.items()
    )
    synthesis = await chat([
        {"role": "system", "content":
         "You are a decision analyst. Given the debate below, provide a "
         "balanced 3-sentence recommendation with clear conditions."},
        {"role": "user", "content": debate_context}
    ], tracker, model=model, max_tokens=256)
    print("✅ Synthesis complete")

    return {"arguments": arguments, "synthesis": synthesis,
            "usage": tracker.report()}


async def main():
    await ping()
    concept = "transformer architecture"
    audiences = ["5-year-old child", "college CS student", "CTO"]

    # Demo Pattern 2: Audience-adaptive explainer
    print("=" * 70)
    print("PATTERN 2: Audience-Adaptive Explainer")
    print("=" * 70)
    print(f"\n📌 Concept: \"{concept}\"")
    print(f"🎯 Audiences: {audiences}")
    print("\nApproach: Tailor vocabulary, analogies, and focus to each audience.")
    for audience in audiences:
        print(f"\n{'─' * 50}")
        print(f"Audience: {audience}")
        print(f"{'─' * 50}")
        result = await explain_for_audience(concept, audience)
        print(result)

    # Demo Pattern 3: Multi-persona debate
    question = "Should our startup adopt a fully remote work policy?"
    print(f"\n{'=' * 70}")
    print("PATTERN 3: Multi-Persona Debate")
    print("=" * 70)
    print(f"\n❓ Question: \"{question}\"")
    print("\nApproach: Three personas (Optimist, Critic, Pragmatist) argue "
          "in parallel, then a decision analyst synthesizes a recommendation.")
    result = await multi_persona_debate(question)

    print(f"\n{'═' * 70}")
    print("DEBATE RESULTS")
    print(f"{'═' * 70}")
    for name, argument in result["arguments"].items():
        print(f"\n[{name.upper()}]:\n{argument}")
    print(f"\n{'─' * 70}")
    print(f"[SYNTHESIS]:\n{result['synthesis']}")
    print(f"\n📊 {result['usage']}")

asyncio.run(main())
