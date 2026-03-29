"""
Prompt Anatomy Demo: Same task, 5 different prompt structures.
Compare how structure affects output quality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from provider_config import init_sync_client  # noqa: E402
client, MODEL, _ = init_sync_client()

TASK_TEXT = """
Meeting Notes - Feb 10, 2026
Attendees: Sarah (PM), Jake (Eng), Maria (Design), Tom (QA)
- Sarah: Launch date moved to March 15. Need to cut scope.
- Jake: Auth module done. Payment integration needs 2 more weeks.
- Maria: New onboarding flow mockups ready for review.
- Tom: Found 3 critical bugs in checkout. Blocking release.
- Sarah: Let's prioritize: fix checkout bugs first, then payments.
- Action items: Jake fixes bugs by Feb 14. Maria shares mockups by Feb 12.
  Tom re-tests checkout after fixes. Sarah updates stakeholders Friday.
"""

def call_llm(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=500,
    )
    return response.choices[0].message.content


# --- Structure 1: No structure (vague) ---
prompt_1 = f"Summarize this:\n{TASK_TEXT}"

# --- Structure 2: Instruction only ---
prompt_2 = f"Summarize the following meeting notes into action items.\n\n{TASK_TEXT}"

# --- Structure 3: Instruction + constraints ---
prompt_3 = f"""Summarize the following meeting notes into action items.
Rules:
- One bullet per action item
- Include owner and deadline
- Maximum 5 items

{TASK_TEXT}"""

# --- Structure 4: All 5 components (markdown delimiters) ---
prompt_4 = f"""# Instruction
Extract action items from the meeting notes below.

# Context
These are weekly engineering standup notes. The team is preparing for a product launch.

# Constraints
- One bullet per action item
- Format: "- [OWNER] ACTION by DEADLINE"
- Maximum 5 items
- Only include items with a clear owner

# Output Format
Return a markdown bulleted list. Nothing else.

# Meeting Notes
{TASK_TEXT}"""

# --- Structure 5: All 5 components (XML delimiters) ---
prompt_5 = f"""<instructions>
Extract action items from the meeting notes below.
</instructions>

<context>
These are weekly engineering standup notes. The team is preparing for a product launch.
</context>

<constraints>
- One bullet per action item
- Format: "- [OWNER] ACTION by DEADLINE"
- Maximum 5 items
- Only include items with a clear owner
</constraints>

<output_format>
Return a markdown bulleted list. Nothing else.
</output_format>

<meeting_notes>
{TASK_TEXT}
</meeting_notes>"""


# Run all 5 and compare
structures = {
    "1. No structure (vague)": prompt_1,
    "2. Instruction only": prompt_2,
    "3. Instruction + constraints": prompt_3,
    "4. Full 5-component (Markdown)": prompt_4,
    "5. Full 5-component (XML)": prompt_5,
}

print("=" * 70)
print("PROMPT ANATOMY COMPARISON — Same task, different structures")
print("=" * 70)

for name, prompt in structures.items():
    print(f"\n{'─' * 60}")
    print(f"📝 {name} — model: {MODEL}")
    print(f"   Input token estimate: ~{len(prompt.split()) * 1.3:.0f} tokens")
    print(f"{'─' * 60}")
    result = call_llm(prompt)
    print(result)
    print()
