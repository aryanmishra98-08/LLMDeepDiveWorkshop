"""
Chain-of-Thought Math Solver: Compare direct answer vs CoT approaches.
Shows the dramatic difference in accuracy for multi-step problems.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from provider_config import init_sync_client  # noqa: E402
client, MODEL, _ = init_sync_client()

PROBLEMS = [
    "A store had 52 apples. They sold 18 in the morning and 12 in the afternoon, "
    "then received a shipment of 30. How many apples does the store have now?",

    "A train travels at 60 mph for 2.5 hours, then at 40 mph for 1.5 hours. "
    "What is the total distance traveled?",

    "Sarah has 3 times as many stickers as Tom. Tom has 8 more stickers than "
    "Maria. If Maria has 12 stickers, how many stickers does Sarah have?",
]

EXPECTED = ["52", "210", "60"]


def solve_direct(problem: str) -> dict:
    """Zero-shot: just ask for the answer."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"{problem}\n\nProvide ONLY the numerical answer."
        }],
        max_completion_tokens=500,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "tokens": response.usage.completion_tokens,
    }


def solve_zero_shot_cot(problem: str) -> dict:
    """Zero-shot CoT: 'Let's think step by step'."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"{problem}\n\nLet's think step by step."
        }],
        max_completion_tokens=300,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "tokens": response.usage.completion_tokens,
    }


def solve_structured_cot(problem: str) -> dict:
    """Few-shot CoT with structured reasoning template."""
    prompt = """Solve math word problems step by step using this format:

Problem: A shop had 40 oranges. They sold 15 and received 10 more. How many remain?
Step 1: Start with 40 oranges.
Step 2: Sold 15 → 40 - 15 = 25
Step 3: Received 10 → 25 + 10 = 35
Answer: 35

Problem: A bakery made 60 cookies and sold 22, then made 15 more. How many cookies?
Step 1: Start with 60 cookies.
Step 2: Sold 22 → 60 - 22 = 38
Step 3: Made 15 more → 38 + 15 = 53
Answer: 53

Problem: """ + problem + """
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=300,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "tokens": response.usage.completion_tokens,
    }


print("=" * 70)
print("CHAIN-OF-THOUGHT COMPARISON: Direct vs Zero-shot CoT vs Structured CoT")
print("=" * 70)

for i, (problem, expected) in enumerate(zip(PROBLEMS, EXPECTED)):
    print(f"\n{'─' * 60}")
    print(f"Problem {i+1}: {problem}")
    print(f"Expected answer: {expected}")
    print(f"{'─' * 60}")

    direct = solve_direct(problem)
    zs_cot = solve_zero_shot_cot(problem)
    st_cot = solve_structured_cot(problem)

    print(f"\n  Direct answer ({direct['tokens']} tokens):\n{direct['answer']}")
    print(f"\n  Zero-shot CoT ({zs_cot['tokens']} tokens):\n{zs_cot['answer']}")
    print(f"\n  Structured CoT ({st_cot['tokens']} tokens):\n{st_cot['answer']}")
