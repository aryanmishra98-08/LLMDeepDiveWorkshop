"""
Prompt Debugger: Isolate which component of a prompt causes failure.
Implements ablation testing by systematically removing components.
"""

import json
from dataclasses import dataclass

from shared_config import azure_client, AZURE_MODEL


@dataclass
class PromptComponent:
    name: str
    content: str


def ablation_test(
    components: list[PromptComponent],
    test_input: str,
) -> dict:
    """
    Test a prompt by removing one component at a time.
    Returns analysis of which components are essential.
    """
    results = {}

    # Test 1: Full prompt (all components)
    full_prompt = "\n\n".join(c.content for c in components) + f"\n\n{test_input}"
    full_response = azure_client.chat.completions.create(
        model=AZURE_MODEL,
        messages=[{"role": "user", "content": full_prompt}],
        max_completion_tokens=500,
    )
    results["FULL PROMPT"] = {
        "output": full_response.choices[0].message.content,
        "tokens": full_response.usage.completion_tokens,
    }

    # Test 2: Remove each component one at a time
    for i, removed in enumerate(components):
        remaining = [c for j, c in enumerate(components) if j != i]
        partial_prompt = "\n\n".join(c.content for c in remaining) + f"\n\n{test_input}"

        response = azure_client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[{"role": "user", "content": partial_prompt}],
            max_completion_tokens=500,
        )
        results[f"WITHOUT: {removed.name}"] = {
            "output": response.choices[0].message.content,
            "tokens": response.usage.completion_tokens,
        }

    # Test 3: Minimal prompt (just the input, no components)
    minimal_response = azure_client.chat.completions.create(
        model=AZURE_MODEL,
        messages=[{"role": "user", "content": test_input}],
        max_completion_tokens=500,
    )
    results["MINIMAL (no components)"] = {
        "output": minimal_response.choices[0].message.content,
        "tokens": minimal_response.usage.completion_tokens,
    }

    return results


def validate(output: str) -> tuple[bool, str]:
    """
    Three independent checks, each guarded by exactly one component:

    CONSTRAINTS  → all string values must be lowercase
                   (models naturally capitalise proper nouns without this rule)
    OUTPUT_FORMAT → exact opaque field names: nm, age_yrs, sev_tier, reply_addr
                   (unguessable without being told; EXAMPLE deliberately shows no JSON)
    EXAMPLE      → sev_tier uses inverted numbering: 1=critical, 3=minor
                   (models assume 1=low, 3=high without the mapping being explained)
    """
    text = output.strip()

    # Basic JSON check
    if text.startswith("```"):
        return False, "wrapped in markdown fences"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"

    # OUTPUT_FORMAT guard — opaque field names
    required_fields = {"nm", "age_yrs", "sev_tier", "reply_addr"}
    missing = required_fields - set(data.keys())
    if missing:
        return False, f"wrong/missing fields {missing} — OUTPUT_FORMAT is essential"

    # CONSTRAINTS guard — all string values must be lowercase
    for field in ("nm", "reply_addr"):
        val = data[field]
        if val is not None and val != val.lower():
            return False, f"{field}={val!r} must be lowercase — CONSTRAINTS is essential"

    # EXAMPLE guard — inverted tier: 1=critical, 2=moderate, 3=minor (counterintuitive)
    sev = data["sev_tier"]
    if sev is not None:
        if not isinstance(sev, int) or sev not in (1, 2, 3):
            return False, (
                f"sev_tier={sev!r} must be int 1/2/3 "
                f"(1=critical, 3=minor) — EXAMPLE is essential"
            )

    return True, "ok"


# --- Component design: each guards a structurally impossible-to-infer failure ---
#
# CONSTRAINTS   → requires ALL string values lowercase.
#                 Models always capitalise proper nouns ("Carlos", not "carlos").
#                 Removing this causes nm="Carlos" → fails lowercase check.
#
# OUTPUT_FORMAT → uses opaque abbreviated field names: nm, age_yrs, sev_tier, reply_addr.
#                 The EXAMPLE intentionally shows NO JSON output, so field names
#                 cannot be learned from any other component.
#                 Removing this causes model to use natural names like
#                 "name"/"severity"/"email" → missing fields → fails.
#
# EXAMPLE       → defines an INVERTED severity tier: 1=critical, 3=minor.
#                 Every model's prior is that 1=lowest, 3=highest.
#                 Removing this causes model to output sev_tier=3 for "CRITICAL"
#                 instead of the correct 1 → fails int-range check.
#
# INSTRUCTION / CONTEXT → expected to be redundant (removing them should still pass).

components = [
    PromptComponent(
        name="INSTRUCTION",
        content="Extract structured contact and issue data from the support ticket below.",
    ),
    PromptComponent(
        name="CONTEXT",
        content=(
            "You are processing incoming tickets for a SaaS customer support triage system. "
            "Tickets are typed informally by agents from phone calls."
        ),
    ),
    PromptComponent(
        name="CONSTRAINTS",
        content=(
            "Formatting rules:\n"
            "- Output ONLY a raw JSON object — no markdown, no code fences, no explanation\n"
            "- Normalise ALL string values to lowercase (required for our database)\n"
            "- Use null for any field that cannot be determined from the text — never guess"
        ),
    ),
    PromptComponent(
        name="OUTPUT_FORMAT",
        content=(
            "Use these exact field names — do not rename them:\n"
            '{"nm": string|null, "age_yrs": number|null, '
            '"sev_tier": 1|2|3|null, "reply_addr": string|null}'
        ),
    ),
    PromptComponent(
        name="EXAMPLE",
        content=(
            "Severity tier reference (NOTE: scale is inverted — 1 is most urgent):\n"
            "  1 = production outage, all users blocked, CRITICAL, URGENT, system down\n"
            "  2 = degraded performance, partial outage, some users affected\n"
            "  3 = cosmetic bug, low priority, minor issue, feature request\n"
            "  null = no severity mentioned in the ticket"
        ),
    ),
]

test_input = (
    'Ticket: "Hi, this is Carlos. We have a production outage — our entire team '
    "can't access the API. This is CRITICAL, need someone ASAP. "
    'No age on file. Best email is carlos.dev@startup.io"'
)

# Correct output: lowercase strings, opaque field names, inverted tier (1 = critical)
expected = '{"nm": "carlos", "age_yrs": null, "sev_tier": 1, "reply_addr": "carlos.dev@startup.io"}'

print("=" * 70)
print("PROMPT ABLATION DEBUGGER — Strict Validation")
print("=" * 70)
print(f"\nTest input: {test_input}")
print(f"Expected:   {expected}")
print()

results = ablation_test(components, test_input)

for config, data in results.items():
    ok, reason = validate(data["output"])
    status = "✅" if ok else "❌"
    output_preview = data["output"].replace("\n", " ")
    print(f"{status} {config}")
    print(f"   Output:  {output_preview}")
    if not ok:
        print(f"   Failure: {reason}")
    print(f"   Tokens:  {data['tokens']}")
    print()

print("─" * 70)
print("INTERPRETATION GUIDE:")
print("  ✅ → output was correct even without this component (possibly redundant)")
print("  ❌ → this component is ESSENTIAL — removing it breaks the output")
print()
print("Expected failure pattern:")
print("  • WITHOUT CONSTRAINTS   → nm='Carlos' instead of 'carlos' — lowercase rule violated")
print("  • WITHOUT OUTPUT_FORMAT → uses 'name'/'severity'/'email' instead of nm/sev_tier/reply_addr")
