"""
Few-Shot Classifier: Sentiment analysis with configurable example count.
Demonstrates the impact of 0-shot, 1-shot, 3-shot, and 5-shot.
"""

from shared_config import azure_client, AZURE_MODEL

# Bank of labeled examples for few-shot
EXAMPLES = [
    ("This laptop is absolutely amazing, best purchase I've ever made!", "POSITIVE"),
    ("Terrible quality. Broke after two days. Want my money back.", "NEGATIVE"),
    ("It works fine. Nothing special but gets the job done.", "NEUTRAL"),
    ("I love the battery life but the screen is disappointing.", "MIXED"),
    ("Worst customer service experience of my life. Never buying again.", "NEGATIVE"),
]

# Test inputs
TEST_REVIEWS = [
    "The camera quality is decent but the price feels a bit steep for what you get.",
    "DO NOT BUY THIS. Complete waste of money. Returned immediately.",
    "Pretty happy with it overall. The setup was easy and it works as advertised.",
]

LABELS = ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]


def build_prompt(n_examples: int, review: str) -> str:
    """Build a classification prompt with n few-shot examples."""
    label_str = ", ".join(LABELS)

    if n_examples == 0:
        return (
            f"Classify the following product review as one of: {label_str}.\n"
            f"Respond with ONLY the label.\n\n"
            f"Review: \"{review}\"\n"
            f"Label:"
        )

    # Build few-shot examples
    example_block = ""
    for text, label in EXAMPLES[:n_examples]:
        example_block += f'Review: "{text}"\nLabel: {label}\n\n'

    return (
        f"Classify product reviews as one of: {label_str}.\n"
        f"Respond with ONLY the label.\n\n"
        f"{example_block}"
        f'Review: "{review}"\n'
        f"Label:"
    )


def classify(n_examples: int, review: str) -> str:
    prompt = build_prompt(n_examples, review)
    response = azure_client.chat.completions.create(
        model=AZURE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=500,
    )
    return response.choices[0].message.content.strip()


# Run comparison
print("=" * 70)
print("FEW-SHOT CLASSIFICATION COMPARISON")
print("=" * 70)

for n in [0, 1, 3, 5]:
    print(f"\n{'─' * 50}")
    print(f"📊 {n}-shot classification")
    print(f"{'─' * 50}")
    for review in TEST_REVIEWS:
        label = classify(n, review)
        print(f"  {label:<10} ← \"{review}\"")
