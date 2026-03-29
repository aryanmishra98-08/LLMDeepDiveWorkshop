"""
Structured Invoice Extractor: Extract typed data from messy text.
Uses Instructor + Pydantic with validation and retry.
"""

import instructor
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date
from enum import Enum
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from provider_config import init_sync_client  # noqa: E402
llm_client, MODEL, PROVIDER = init_sync_client()

# Map each provider to the best instructor extraction mode.
# provider_config normalizes all providers to an OpenAI-compatible interface,
# so instructor.from_openai works universally. The mode controls how the
# model returns structured data:
#   TOOLS — native function/tool calling (OpenAI, Azure): most reliable
#   JSON  — plain JSON in the message body (Ollama, Gemini, Anthropic adapter):
#            required for providers that don't support function calling
_INSTRUCTOR_MODES: dict[str, instructor.Mode] = {
    "openai":     instructor.Mode.TOOLS,
    "azure":      instructor.Mode.TOOLS,
    "anthropic":  instructor.Mode.JSON,
    "gemini":     instructor.Mode.JSON,
    "ollama":     instructor.Mode.MD_JSON,
}
_mode = _INSTRUCTOR_MODES.get(PROVIDER, instructor.Mode.JSON)
instructor_client = instructor.from_openai(llm_client, mode=_mode)


# --- Define the extraction schema ---

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"

class LineItem(BaseModel):
    description: str = Field(description="Product or service description")
    quantity: int = Field(ge=1, description="Number of units")
    unit_price: float = Field(ge=0, description="Price per unit as stated (e.g. monthly rate, per-unit rate)")
    total: float = Field(ge=0, description="Actual billed amount for this line item as stated in the invoice")

class Invoice(BaseModel):
    vendor_name: str = Field(description="Company that issued the invoice")
    invoice_number: Optional[str] = Field(
        default=None, description="Invoice ID/number if present"
    )
    invoice_date: date = Field(description="Date the invoice was issued")
    due_date: Optional[date] = Field(
        default=None, description="Payment due date if mentioned"
    )
    currency: Currency = Field(
        default=Currency.USD, description="Currency of the amounts"
    )
    line_items: list[LineItem] = Field(
        min_length=1, description="Individual items/charges"
    )
    subtotal: float = Field(ge=0, description="Sum before tax")
    tax_rate: Optional[float] = Field(
        default=None, ge=0, le=1, description="Tax rate as decimal (e.g., 0.085)"
    )
    total_amount: float = Field(ge=0, description="Final total including tax")

    @field_validator("total_amount")
    @classmethod
    def total_must_exceed_subtotal(cls, v, info):
        if "subtotal" in info.data and v < info.data["subtotal"]:
            raise ValueError(
                f"Total ({v}) cannot be less than subtotal ({info.data['subtotal']})"
            )
        return v


# --- Extract from messy text ---

INVOICE_TEXT = """
Hey, attaching the invoice from Acme Cloud Services.

Invoice #ACM-2026-0142
Date: February 3, 2026

Items:
- Pro Plan subscription (annual), 5 seats @ $29/mo each = $1,740.00
- Additional storage 500GB, $0.10/GB/mo x 12 months = $600.00
- One-time setup & migration fee: $350.00
- Priority support add-on: 5 seats x $5/mo x 12 = $300.00

Subtotal: $2,990.00
Tax (8.5%): $254.15
Total Due: $3,244.15

Payment due within 30 days (by March 5, 2026).
Wire to: Acme Cloud Services LLC, routing 021000021, acct 1234567890
"""

# Extract with automatic retry on validation failure
invoice = instructor_client.chat.completions.create(
    model=MODEL,
    response_model=Invoice,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a precise data extraction assistant. "
                "Output only valid JSON — no inline comments, no annotations, no explanatory text inside the JSON. "
                "Always use ISO 8601 date format with zero-padded month and day (YYYY-MM-DD), e.g. 2026-02-03."
            ),
        },
        {
            "role": "user",
            "content": (
                "Extract all invoice data from the following text. "
                "Calculate quantities and unit prices for annual items.\n\n"
                f"{INVOICE_TEXT}"
            ),
        },
    ],
    max_retries=3,  # Retry up to 3 times if validation fails
)

# --- Display results ---
print("=" * 60)
print("EXTRACTED INVOICE DATA")
print("=" * 60)
print(f"Vendor:     {invoice.vendor_name}")
print(f"Invoice #:  {invoice.invoice_number}")
print(f"Date:       {invoice.invoice_date}")
print(f"Due:        {invoice.due_date}")
print(f"Currency:   {invoice.currency.value}")
print(f"\nLine Items:")
for item in invoice.line_items:
    print(f"  • {item.description}")
    print(f"    qty={item.quantity}, unit price=${item.unit_price:.2f} → ${item.total:.2f}")
print(f"\nSubtotal:   ${invoice.subtotal:,.2f}")
print(f"Tax rate:   {invoice.tax_rate:.1%}" if invoice.tax_rate else "Tax rate:   N/A")
print(f"Total:      ${invoice.total_amount:,.2f}")

# Validate the extraction
print(f"\n{'─' * 40}")
print("VALIDATION CHECK:")
computed_total = sum(item.total for item in invoice.line_items)
print(f"  Sum of line items: ${computed_total:,.2f}")
print(f"  Stated subtotal:   ${invoice.subtotal:,.2f}")
print(f"  Match: {'✅' if abs(computed_total - invoice.subtotal) < 0.01 else '❌'}")
