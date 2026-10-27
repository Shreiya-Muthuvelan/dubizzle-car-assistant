"""
get_filters_to_clear() function identifies the type of constraint user wants to remove
is_new_search() will help identify if its a new search and start from scratch
"""
from uuid import uuid4
from typing import Optional


# Temporary in-memory conversation storage.
# Conversations are lost when the application restarts.
conversations: dict[str, dict] = {}


# Fields that belong to the selling-price constraint.
PRICE_FIELDS = {"min_price","max_price",}

# Fields that belong to the monthly-payment constraint.
MONTHLY_PAYMENT_FIELDS = {"min_monthly_payment","max_monthly_payment",}

# Fields that belong to the mileage constraint.
MILEAGE_FIELDS = {"min_mileage","max_mileage",}

# Fields that belong to the manufacturing-year constraint.
YEAR_FIELDS = {"min_year","max_year",}


def get_or_create_conversation(conversation_id: Optional[str] = None,) -> tuple[str, dict]:
    """
    Retrieve an existing conversation or create a new one.
    """

    if conversation_id is None:
        conversation_id = str(uuid4())

    if conversation_id not in conversations:
        conversations[conversation_id] = {"messages": [],"search_filters": {},"last_listings": [],}

    return conversation_id, conversations[conversation_id]


def is_new_search(message: str) -> bool:
    """
    Detect when the user starts a new vehicle search rather than
    asking a follow-up about the previous results.
    """
    text = message.lower().strip()

    # Explicit requests to reset the search.
    reset_phrases = (
        "start a new search",
        "start over",
        "start again",
        "new search",
        "fresh search",
        "search from scratch",
        "forget my previous search",
        "forget the previous search",
        "clear all filters",
        "remove all filters",
        "without any restrictions",
        "without any filters",
        "no restrictions",
    )

    if any(phrase in text for phrase in reset_phrases):
        return True

    # Explicit requests to search for vehicles.
    new_search_phrases = (
        "show me",
        "find me",
        "find cars",
        "search for",
        "look for",
        "browse",
        "what cars",
        "which cars",
        "show cars",
        "show vehicles",
        "find vehicles",
        "looking for",
    )

    if any(phrase in text for phrase in new_search_phrases):
        return True

    return False


def get_filters_to_clear(message: str) -> set[str]:
    """
    Detect explicit requests to remove previously stored filters.

    This is intentionally rule-based for predictable behaviour.
    """

    text = message.lower().strip()
    fields_to_clear = set()

    # Remove selling-price constraints.
    price_phrases = [ "forget the budget","remove the budget",
        "clear the budget","ignore the budget",
        "no budget limit","without a budget limit","remove the price limit",
        "remove the selling price limit","forget the price limit",
        "no price limit",]

    if any(phrase in text for phrase in price_phrases):
        fields_to_clear.update(PRICE_FIELDS)

    # Remove monthly-payment constraints.
    payment_phrases = ["forget the monthly payment",
        "remove the monthly payment","clear the monthly payment","ignore the monthly payment",
        "no monthly payment limit","without a monthly payment limit", ]

    if any(phrase in text for phrase in payment_phrases):
        fields_to_clear.update(MONTHLY_PAYMENT_FIELDS)

    # Remove mileage constraints.
    mileage_phrases = [
        "forget the mileage","remove the mileage limit","clear the mileage limit",
        "no mileage limit","without a mileage limit",]

    if any(phrase in text for phrase in mileage_phrases):
        fields_to_clear.update(MILEAGE_FIELDS)

    # Remove manufacturing-year constraints.
    year_phrases = [
        "forget the year restriction","remove the year restriction",
        "clear the year restriction","remove the year limit",
        "no year restriction","without a year restriction",]

    if any(phrase in text for phrase in year_phrases):
        fields_to_clear.update(YEAR_FIELDS)

    return fields_to_clear


def merge_search_filters(previous_filters: dict,new_filters: dict,message: str = "",) -> dict:
    """
    Merge extracted filters with the existing search state.

    Rules:
    1. Explicit new-search requests reset previous filters.
    2. Explicit removal requests clear the relevant constraints.
    3. Newly extracted non-null values replace previous values.
    4. Unmentioned constraints remain unchanged.
    """

    # A new search must not inherit the previous search's filters.
    if is_new_search(message):
        merged = {}
    else:
        merged = previous_filters.copy()

    # Remove constraints explicitly cancelled by the user.
    fields_to_clear = get_filters_to_clear(message)

    for field in fields_to_clear:
        merged.pop(field, None)

    # Apply newly extracted values.
    for field, value in new_filters.items():
        if value is not None:
            merged[field] = value

    return merged


def update_conversation(
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    search_filters: dict,
    listings: Optional[list[dict]] = None,
) -> None:
    """
    Save the latest conversation turn, search filters,
    and optionally the retrieved vehicle listings.
    """

    state = conversations[conversation_id]
    state["search_filters"] = search_filters

    # Update listings only when explicitly provided.
    # This preserves previous listings for ordinary follow-ups.
    if listings is not None:
        state["last_listings"] = listings

    state["messages"].append({
        "role": "user",
        "content": user_message,
    })

    state["messages"].append({
        "role": "assistant",
        "content": assistant_message,
    })

    # Keep only the most recent 20 messages.
    state["messages"] = state["messages"][-20:]


