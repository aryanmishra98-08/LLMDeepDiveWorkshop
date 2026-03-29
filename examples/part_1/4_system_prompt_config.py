"""
System Prompt Configurator: Same task, different system/user prompt strategies.
Shows how moving instructions between system and user prompts affects behavior.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from provider_config import init_sync_client  # noqa: E402
client, MODEL, _ = init_sync_client()

TASK = "Explain what a Python decorator is and provide one example."

# --- Strategy 1: Everything in user prompt (no system prompt) ---
def strategy_no_system():
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                "You are a senior Python developer teaching a junior colleague. "
                "Be concise. Use code examples. Keep it under 150 words.\n\n"
                f"Task: {TASK}"
            ),
        }],
        max_completion_tokens=500,
    )
    return response.choices[0].message.content

# --- Strategy 2: Persona in system, task in user (recommended) ---
def strategy_split():
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior Python developer teaching a junior colleague. "
                    "Be concise. Use code examples. Keep responses under 150 words."
                ),
            },
            {"role": "user", "content": TASK},
        ],
        max_completion_tokens=500,
    )
    return response.choices[0].message.content

# --- Strategy 3: Heavy system prompt with constraints ---
def strategy_heavy_system():
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior Python developer teaching a junior colleague.\n\n"
                    "# Rules\n"
                    "- Be concise: 150 words max\n"
                    "- Always include a runnable code example\n"
                    "- Explain WHY, not just WHAT\n"
                    "- Use analogies when helpful\n"
                    "- End with 'Try it yourself: [exercise suggestion]'\n"
                ),
            },
            {"role": "user", "content": TASK},
        ],
        max_completion_tokens=500,
    )
    return response.choices[0].message.content

# --- Strategy 4: XML-style system prompt ---
def strategy_xml_system():
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior Python developer teaching a junior colleague.\n"
                    "<rules>\n"
                    "- Be concise: 150 words max\n"
                    "- Always include a runnable code example\n"
                    "- Explain WHY, not just WHAT\n"
                    "- End with a 'Try it yourself' exercise\n"
                    "</rules>"
                ),
            },
            {"role": "user", "content": TASK},
        ],
        max_completion_tokens=500,
    )
    return response.choices[0].message.content


strategies = {
    "1. No system prompt (all in user)": strategy_no_system,
    "2. Split: persona in system, task in user": strategy_split,
    "3. Heavy system prompt with constraints": strategy_heavy_system,
    "4. XML-style system prompt": strategy_xml_system,
}

print("=" * 70)
print("SYSTEM PROMPT STRATEGY COMPARISON")
print("=" * 70)

for name, fn in strategies.items():
    print(f"\n{'─' * 60}")
    print(f"📋 {name}")
    print(f"{'─' * 60}")
    print(fn())
    print()
