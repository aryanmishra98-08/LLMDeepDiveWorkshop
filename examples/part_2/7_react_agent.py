"""
ReAct Agent from Scratch: Full ReAct implementation with tool execution,
parse error handling, and token tracking. No frameworks.
"""

import re
import math
import asyncio
from dataclasses import dataclass

from shared_config import AZURE_MODEL, TokenTracker, chat

# ── Tool definitions ──────────────────────────────────────────────────
TOOLS = {
    "calculator": {
        "desc": "Evaluate a math expression. Input: valid Python math expression string.",
        "func": lambda expr: str(eval(expr, {"__builtins__": {}}, {"math": math})),
    },
    "web_search": {
        "desc": "Search the web for a query. Input: search query string.",
        "func": lambda q: next(
            (result for keywords, result in [
                (["population", "france"],       "France has a population of approximately 68.4 million (2025)."),
                (["gdp", "france"],              "France GDP is approximately $3.1 trillion USD (2024)."),
                (["capital", "france"],          "The capital of France is Paris."),
            ] if all(kw in q.lower() for kw in keywords)),
            f"No results found for: {q}"
        ),
    },
}

REACT_SYSTEM = """You solve tasks by interleaving Thought, Action, and Observation steps.

Available tools:
{tool_block}

Output format — follow this EXACTLY on each turn:
Thought: <your reasoning about what to do next>
Action: <tool_name>[<input>]

When you have the final answer:
Thought: <reasoning why you're done>
Action: finish[<your final answer>]

Rules:
- Always emit exactly one Thought then one Action per turn.
- Never fabricate observations. Wait for real tool output.
- If a tool errors, reason about it and try a different approach."""

def build_tool_block() -> str:
    return "\n".join(f"- {name}: {t['desc']}" for name, t in TOOLS.items())

def parse_action(text: str) -> tuple[str, str]:
    """Extract tool name and input from 'Action: tool_name[input]'."""
    m = re.search(r"Action:\s*(\w+)\[(.+?)\]\s*$", text, re.MULTILINE | re.DOTALL)
    if not m:
        raise ValueError(f"Could not parse action from:\n{text}")
    return m.group(1).strip(), m.group(2).strip()

async def react_agent(question: str, max_steps: int = 7,
                      model: str = None) -> dict:
    """Run a ReAct loop: Thought → Action → Observation → ... → finish."""
    model = model or AZURE_MODEL
    tracker = TokenTracker()
    system = REACT_SYSTEM.format(tool_block=build_tool_block())
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}"},
    ]
    trajectory = []

    for step in range(1, max_steps + 1):
        print(f"\n{'─' * 50}")
        print(f"⟳  Step {step} — calling LLM...")

        # 1. LLM generates Thought + Action
        llm_output = await chat(messages, tracker, model=model)
        print(f"\n🧠 LLM output:")
        print(f"   {llm_output.strip().replace(chr(10), chr(10) + '   ')}")

        # 2. Parse action
        try:
            tool_name, tool_input = parse_action(llm_output)
        except ValueError:
            print("⚠️  Parse error — injecting format correction and retrying")
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content":
                "Observation: ERROR — invalid format. Use Action: tool_name[input]"})
            trajectory.append({"step": step, "error": "parse_failure"})
            continue

        print(f"\n🔧 Parsed action: tool={tool_name!r}  input={tool_input!r}")

        # 3. Check termination
        if tool_name == "finish":
            print(f"\n✅ Agent called finish — loop complete")
            trajectory.append({"step": step, "thought": llm_output, "answer": tool_input})
            return {"answer": tool_input, "steps": step,
                    "trajectory": trajectory, "usage": tracker.report()}

        # 4. Execute tool
        if tool_name not in TOOLS:
            observation = f"ERROR: Unknown tool '{tool_name}'. Available: {list(TOOLS)}"
            print(f"\n❌ Unknown tool '{tool_name}'")
        else:
            print(f"\n⚙️  Executing tool: {tool_name}[{tool_input}]")
            try:
                observation = TOOLS[tool_name]["func"](tool_input)
                print(f"📥 Observation: {observation}")
            except Exception as e:
                observation = f"ERROR: {type(e).__name__}: {e}"
                print(f"❌ Tool error: {observation}")

        # 5. Inject observation and loop
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": f"Observation: {observation}"})
        trajectory.append({"step": step, "thought": llm_output,
                           "action": tool_name, "input": tool_input,
                           "observation": observation})
        print(f"\n💬 Observation injected into context — continuing loop")

    return {"answer": None, "steps": max_steps,
            "trajectory": trajectory, "usage": tracker.report(),
            "error": "max_steps_exceeded"}

# Run it
async def main():
    question = "What is the per-capita GDP of France? Use the population and GDP tools."

    print("=" * 60)
    print("ReAct AGENT — Reasoning + Acting Loop")
    print("=" * 60)
    print(f"\n❓ Question: {question}")
    print(f"\nAvailable tools: {list(TOOLS.keys())}")
    print(f"Max steps: 7")

    result = await react_agent(question)

    print(f"\n{'═' * 60}")
    print("TRAJECTORY SUMMARY")
    print(f"{'═' * 60}")
    for t in result["trajectory"]:
        action = t.get("action", "finish")
        detail = t.get("observation", t.get("answer", "(parse error)"))
        print(f"  Step {t['step']}: {action:15} → {detail}")

    print(f"\n{'─' * 60}")
    print(f"✅ Final answer: {result['answer']}")
    print(f"📊 {result['usage']}")

asyncio.run(main())
