
import csv
import re

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


# Store leads in the project's data directory.
LEADS_FILE = Path("data/leads.csv")

LEAD_FIELDS = ["lead_id","conversation_id",
    "created_at","name",
    "contact","preferred_vehicle",
    "budget_aed","monthly_payment_aed",
    "requirements","purchase_timeline",
    "status",]

# Temporary qualification state, separate from conversation memory.
lead_sessions: dict[str, dict] = {}


def detect_purchase_intent(message: str) -> bool:
    """Check whether the user expresses an interest in buying a car."""
    text = message.lower()
    purchase_phrases = [
        "i want to buy","i'd like to buy",
        "i would like to buy","i'm interested in buying",
        "i am interested in buying","i want to purchase",
        "i'd like to purchase","i would like to purchase",
        "i'm ready to buy", "i am ready to buy",
        "contact me","sales team",
        "speak to a salesperson","speak to someone",]

    return any(phrase in text for phrase in purchase_phrases)


def get_or_create_lead_session(conversation_id: str) -> dict:
    """Retrieve an existing qualification session or create one."""

    if conversation_id not in lead_sessions:
        lead_sessions[conversation_id] = {
            "lead_id": str(uuid4()),
            "conversation_id": conversation_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "name": None,
            "contact": None,
            "preferred_vehicle": None,
            "budget_aed": None,
            "monthly_payment_aed": None,
            "requirements": None,
            "purchase_timeline": None,
            "status": "collecting",
            "awaiting": None,
            "saved": False,}

    return lead_sessions[conversation_id]


def extract_contact(message: str) -> str | None:
    """Extract an email address or phone number from a message."""

    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    email_match = re.search(email_pattern, message)

    if email_match:
        return email_match.group(0)

    # Simple phone-number pattern for common international formats.
    phone_pattern = r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"

    phone_match = re.search(phone_pattern, message)

    if phone_match:
        phone = phone_match.group(0).strip()

        # Require at least nine digits to avoid capturing short numbers.
        if sum(character.isdigit() for character in phone) >= 9:
            return phone

    return None


def extract_name(message: str) -> str | None:
    """
    Extract a name only when the customer explicitly introduces it.
    For example: 'My name is Sarah Ahmed'.
    """

    patterns = [
        r"\bmy name is\s+(.+)",
        r"\bthis is\s+(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)

        if match:
            name = match.group(1).strip().strip(" .,! ")

            # Reject empty, excessively long, or contact-like values.
            if (
                1 <= len(name.split()) <= 4
                and len(name) <= 80
                and not extract_contact(name)
            ):
                return name

    return None
def extract_budget(message: str) -> tuple[str, float] | None:
    """
    Extract a budget or monthly payment from a direct user response.

    Returns:
        ("budget_aed", amount) or ("monthly_payment_aed", amount)
    """
    text = message.lower().replace(",", "")

    # Identify whether the user is specifying a monthly payment.
    is_monthly = any(
        phrase in text
        for phrase in (
            "monthly",
            "per month",
            "a month",
            "monthly payment",
            "installment",
            "instalment",
        )
    )

    # Support common shorthand such as 100k.
    match = re.search(
        r"(?<!\w)(\d+(?:\.\d+)?)\s*(k|m)?(?!\w)",
        text,
    )

    if not match:
        return None

    amount = float(match.group(1))
    suffix = match.group(2)

    if suffix == "k":
        amount *= 1_000
    elif suffix == "m":
        amount *= 1_000_000

    if amount <= 0:
        return None

    field = "monthly_payment_aed" if is_monthly else "budget_aed"

    return field, amount
def is_unrelated_to_pending_field(
    message: str,
    awaiting: str | None,
) -> bool:
    """Detect clearly unrelated messages during lead qualification."""

    if not awaiting:
        return False

    text = message.lower().strip()

    unrelated_phrases = (
        "who won",
        "football match",
        "cricket match",
        "weather today",
        "write me a poem",
        "tell me a joke",
        "help me with homework",
        "solve this equation",
        "debug my code",
    )

    return any(phrase in text for phrase in unrelated_phrases)

def update_lead_from_filters(lead: dict,filters: dict,) -> None:
    """Reuse vehicle preferences already extracted by the car-search system."""

    make = filters.get("make")
    model = filters.get("model_name")

    if make and model:
        lead["preferred_vehicle"] = f"{make} {model}"
    elif make:
        lead["preferred_vehicle"] = make
    elif model:
        lead["preferred_vehicle"] = model

    # Preserve an existing preference when the current message
    # does not provide a replacement.
    if filters.get("search_query"):
        lead["requirements"] = filters["search_query"]

    # Prefer the upper selling-price limit as the stated budget.
    # Reuse search filters only when the corresponding lead field is empty.
    if lead.get("budget_aed") is None:
        if filters.get("max_price") is not None:
            lead["budget_aed"] = filters["max_price"]
        elif filters.get("min_price") is not None:
            lead["budget_aed"] = filters["min_price"]

    if lead.get("monthly_payment_aed") is None:
        if filters.get("max_monthly_payment") is not None:
            lead["monthly_payment_aed"] = filters["max_monthly_payment"]
        elif filters.get("min_monthly_payment") is not None:
            lead["monthly_payment_aed"] = filters["min_monthly_payment"]


def get_missing_fields(lead: dict) -> list[str]:
    """Return required lead details that have not yet been collected."""
    missing = []

    if not lead.get("name"):
        missing.append("name")

    if not lead.get("contact"):
        missing.append("contact")

    if not lead.get("preferred_vehicle"):
        missing.append("preferred_vehicle")

    if (
        lead.get("budget_aed") is None
        and lead.get("monthly_payment_aed") is None
    ):
        missing.append("budget")

    return missing

def get_next_question(missing_field: str) -> str:
    """Generate a prompt for the next missing detail."""

    questions = {
        "name": "To get started, what name should I put on your enquiry?",
        "contact": (
            "What's the best email address or phone number "
            "for the team to reach you?"
        ),
        "preferred_vehicle": (
            "Which car are you most interested in? "
            "You can share the make, model, or type of vehicle."
        ),
        "budget": (
            "What's your approximate budget in AED, "
            "or your preferred maximum monthly payment?"
        ),
    }

    return questions[missing_field]


def save_lead(lead: dict) -> None:
    """Append a qualified lead to the CSV file."""

    LEADS_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LEADS_FILE.exists()
    with LEADS_FILE.open(mode="a",newline="",encoding="utf-8",) as file:
        writer = csv.DictWriter(file, fieldnames=LEAD_FIELDS)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            field: lead.get(field)
            for field in LEAD_FIELDS})


def process_lead_message(conversation_id: str,message: str,
    filters: dict,) -> dict | None:
    """
    Process a message for lead qualification.

    Returns None when lead qualification has not started.
    Otherwise returns the current qualification status and response.
    """

    lead = lead_sessions.get(conversation_id)
    # Start qualification only when purchase intent is detected.
    if lead is None:
        if not detect_purchase_intent(message):
            return None

        lead = get_or_create_lead_session(conversation_id)

    # Do not create another CSV record for an already saved lead.
    if lead["saved"]:
        return {
            "status": "qualified",
            "lead_id": lead["lead_id"],
            "message": (
                "Your enquiry has already been recorded. "
                "The team can follow up using the contact details provided."
            ),}
    # Do not interpret unrelated questions as lead qualification answers.
    if is_unrelated_to_pending_field(message, lead.get("awaiting")):
        return {
            "status": "collecting",
            "lead_id": lead["lead_id"],
            "missing_fields": get_missing_fields(lead),
            "message": (
                "I'm here to help with car enquiries, so I can't answer "
                "that question. We can continue with your car enquiry. "
                + get_next_question(lead["awaiting"])
            ),
        }
    # Reuse vehicle details and budget from the existing search system.
    update_lead_from_filters(lead, filters)

    # Extract contact details when provided.
    contact = extract_contact(message)

    if contact:
        lead["contact"] = contact
    # Extract budget or monthly payment when provided directly.
    budget_result = extract_budget(message)

    if budget_result:
        field, amount = budget_result
        lead[field] = amount

    # Extract a name when the user explicitly provides one.
    name = extract_name(message)

    if name:
        lead["name"] = name
    elif lead.get("awaiting") == "name":
        # Accept a short response as a name when we explicitly asked for it.
        candidate = message.strip()

        if (1 <= len(candidate.split()) <= 4
            and len(candidate) <= 80
            and not extract_contact(candidate)):
            lead["name"] = candidate

    # Determine whether all required information is available.
    missing_fields = get_missing_fields(lead)

    if missing_fields:
        next_field = missing_fields[0]
        lead["awaiting"] = next_field

        return {
            "status": "collecting",
            "lead_id": lead["lead_id"],
            "missing_fields": missing_fields,
            "message": get_next_question(next_field),}

    # Save only once, after all required details are available.
    lead["status"] = "qualified"
    lead["awaiting"] = None

    save_lead(lead)

    lead["saved"] = True

    return {
        "status": "qualified",
        "lead_id": lead["lead_id"],
        "message": (
            "Thank you! I've recorded your enquiry and preferences. "
            "The team can follow up using the contact details you provided."
        ),}
