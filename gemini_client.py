
"""
Design choices :
* we are using pydantic as we need a proper strcuture for our python code 
* selling price and monthly payment are kept different since they are different constraints
* we use two gemini calls one to interpret the user's request and second to produce a readable answer
from the retrieved results. keeping them seperates helps in debugging

* rn the extraction function only handles car search messages, we need to extend it to intent handling for greetings, 
general automotive questions, booking requests, 
and lead qualification as we build those capabilities.

"""


import json
import os
from typing import Optional
from dotenv import load_dotenv
from google import genai
import pandas as pd
from pydantic import BaseModel, Field, model_validator

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found. Check your .env file.")

client = genai.Client(
    api_key=api_key,
    http_options={"timeout": 8000},
)
model_name = "gemini-3.8-flash"
from datetime import date, datetime, timedelta


class BookingRequest(BaseModel):
    """Structured information extracted from a vehicle-viewing request."""

    intent: str = Field(
        description=(
            "One of: booking, car_search, lead_enquiry, "
            "general_question, greeting"
        )
    )

    listing_id: Optional[str] = None
    booking_date: Optional[str] = None
    booking_time: Optional[str] = None


def extract_booking_request(
    message: str,
    today: date | None = None,
) -> BookingRequest:
    """
    Identify the user's intent and extract viewing appointment details.

    Relative dates are resolved using the supplied current date.
    """

    today = today or datetime.now().date()

    prompt = f"""
You are the intent and booking-detail extraction component of a
UAE automotive marketplace assistant.

Classify the user's message into exactly one intent:

- booking: the user wants to arrange, change, or discuss a
  vehicle-viewing appointment.
- car_search: the user wants to find or compare cars.
- lead_enquiry: the user expresses an intention to buy a car,
  contact sales, or speak to a salesperson.
- general_question: the user asks a general automotive question.
- greeting: the user only greets the assistant.

Extract these fields:
- intent
- listing_id
- booking_date
- booking_time

Rules:
1. Return only the structured JSON response.
2. Use null for information that is not stated or cannot be
   confidently inferred.
3. Only extract a listing ID if it appears in the message.
4. Dates must use YYYY-MM-DD format.
5. Times must use 24-hour HH:MM format.
6. Resolve relative dates using today's date: {today.isoformat()}.
7. Resolve "tomorrow" as the following calendar day.
8. Do not invent a date or time.
9. Do not classify an ordinary car-search request as a booking
   unless the user expresses an intention to arrange a viewing.
10. If the message is a response to a booking question, classify
    it as booking when the message contains booking details.

User message:
{message}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": BookingRequest,
            "temperature": 0,
        },
    )

    if not response.text:
        raise ValueError("Gemini returned an empty intent response.")

    data = json.loads(response.text)
    return BookingRequest.model_validate(data)

class CarSearchFilters(BaseModel):
    """Structured search parameters extracted from a user message."""

    make: Optional[str] = None
    model_name: Optional[str] = None

    min_year: Optional[int] = None
    max_year: Optional[int] = None

    min_price: Optional[float] = None
    max_price: Optional[float] = None

    min_monthly_payment: Optional[float] = None
    max_monthly_payment: Optional[float] = None

    min_mileage: Optional[float] = None
    max_mileage: Optional[float] = None

    search_query: Optional[str] = None

    @model_validator(mode="after")
    def validate_ranges(self):
        ranges = [ ("min_year", "max_year"),("min_price", "max_price"),
            ("min_monthly_payment", "max_monthly_payment"),("min_mileage", "max_mileage"),]

        for minimum, maximum in ranges:
            low = getattr(self, minimum)
            high = getattr(self, maximum)
            if low is not None and high is not None and low > high:
                raise ValueError( f"{minimum} cannot exceed {maximum}." )

        return self



def extract_search_filters(message: str) -> CarSearchFilters:
    """Extract car-search filters using Gemini, with a local fallback."""

    prompt = f"""
You are the query-understanding component of a UAE automotive
marketplace assistant.

Extract search requirements from the user's message.

Return ONLY a JSON object with these fields:
- make: vehicle manufacturer, if specified
- model_name: specific vehicle model, if specified
- min_year: minimum acceptable manufacturing year
- max_year: maximum acceptable manufacturing year
- min_price: minimum selling price in AED
- max_price: maximum selling price in AED
- min_monthly_payment: minimum monthly payment in AED
- max_monthly_payment: maximum monthly payment in AED
- min_mileage: minimum mileage in kilometres
- max_mileage: maximum mileage in kilometres
- search_query: remaining descriptive preferences useful for
  semantic search

Rules:
1. Use null for unspecified numeric or categorical filters.
2. "After 2023" means min_year is 2024.
3. "2023 onwards" means min_year is 2023.
4. "Before 2020" means max_year is 2019.
5. Convert AED amounts into numbers without currency symbols.
6. Do not invent requirements that the user did not specify.
7. Preserve descriptive preferences in search_query.
8. Do not treat a monthly payment budget as a selling-price budget.
9. If the message is not a car-search request, return null for
   the search filters and use an empty search_query.

User message:
{message}
"""

    try:
        print("Extracting search filters with Gemini")

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": CarSearchFilters,
                "temperature": 0,
            },
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        data = json.loads(response.text)
        return CarSearchFilters.model_validate(data)

    except Exception as exc:
        print(
            f"Gemini filter extraction failed: {exc}"
        )
        print("Attempting local search-filter fallback...")

        try:
            filters = fallback_extract_search_filters(message)
            print(f"Local fallback filters: {filters}")
            return filters

        except Exception as fallback_error:
            print(
                "Local search-filter fallback also failed: "
                f"{fallback_error}"
            )
            raise


def format_listings_response(listings: list[dict]) -> str:
    """Generate a readable response from retrieved listings without Gemini."""

    if not listings:
        return (
            "I couldn't find any matching vehicles. "
            "Try adjusting your search criteria."
        )

    lines = [
        f"I found {len(listings)} matching vehicle listing(s):"
    ]

    for index, car in enumerate(listings, start=1):
        make = car.get("make") or "Unknown make"
        model = car.get("model") or "Unknown model"
        listing_id = car.get("listing_id", "Not specified")

        lines.extend([
            "",
            f"**{index}. {make} {model}**",
            f"- Listing ID: {listing_id}",
        ])

        year = car.get("year")
        if pd.notna(year):
            lines.append(f"- Year: {year}")

        trim = car.get("trim")
        if trim and pd.notna(trim):
            lines.append(f"- Trim: {trim}")

        price = car.get("price_aed")
        if pd.notna(price):
            lines.append(f"- Selling price: AED {float(price):,.0f}")

        monthly_payment = car.get("monthly_payment_aed")
        if pd.notna(monthly_payment):
            lines.append(
                f"- Monthly payment: AED {float(monthly_payment):,.0f}"
            )

        mileage = car.get("mileage_km")
        if pd.notna(mileage):
            lines.append(f"- Mileage: {float(mileage):,.0f} km")

        description = car.get("clean_description") or car.get("description")
        if description and pd.notna(description):
            lines.append(f"- Description: {description}")

    lines.extend([
        "",
        "Would you like more details about any of these vehicles?",
    ])

    return "\n".join(lines)


def generate_car_response(message: str,filters: CarSearchFilters,listings: list[dict],) -> str:
    """Generate an answer grounded in retrieved vehicle listings."""

    if not listings:
        return ("I couldn't find any matching listings with the ""available information. You could try adjusting "
            "your budget, year range, or other requirements.")

    # Pass only relevant listing information to the LLM.
    relevant_fields = ["listing_id","make","model","trim","year","title","price_aed","monthly_payment_aed","mileage_km",
        "description","photo_url",]

    safe_listings = []

    for listing in listings:
        safe_listings.append({
            key: listing.get(key)
            for key in relevant_fields
            if key in listing
        })

    prompt = f"""
You are a helpful automotive marketplace assistant for the UAE.

Answer the user's request using ONLY the supplied vehicle listings.

Rules:
1. Never invent a vehicle, price, year, mileage, or payment amount.
2. Treat null values as unknown, not zero.
3. Do not claim that a vehicle meets a requirement unless the
   supplied listing data supports that claim.
4. If a field is unknown, say it is not available in the listing.
5. Recommend relevant vehicles and briefly explain why they match.
6. Mention the selling price and mileage when available.
7. Do not claim a vehicle is available for purchase unless the
   listing data explicitly establishes its availability.
8. If no listings were supplied, do not recommend any vehicles.
9. If a structured field is missing but the listing description
   explicitly provides that information, you may report the value
   while clearly attributing it to the description.
User message:
{message}

Extracted search requirements:
{filters.model_dump_json()}

Retrieved listings:
{json.dumps(safe_listings, ensure_ascii=False, default=str)}

Provide a concise, natural response. When recommending a vehicle,
identify it using its listing ID when one is available.
"""

    try:
        print("Generating final response with Gemini")

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"temperature": 0.2},
        )

        if not response.text:
            raise ValueError("Gemini returned an empty answer.")

        return response.text.strip()

    except Exception as exc:
        print(f"Gemini response generation failed: {exc}")
        print("Generating response locally from retrieved listings.")

        return format_listings_response(listings)

import re


def fallback_extract_search_filters(message: str) -> CarSearchFilters:
    """
    Extract common car-search filters locally when Gemini is unavailable.

    This is intentionally conservative: it extracts clear constraints
    rather than guessing information the user did not provide.
    """

    text = message.lower()
    filters = {
        "search_query": message,
    }

    # Load the available listing values locally.
    # Importing inside the function avoids loading the retrieval module
    # unnecessarily when the fallback is not needed.
    from src.semantic_retrieval import listings

    # --------------------------------------------------
    # 1. Extract make and model from the dataset
    # --------------------------------------------------

    available_makes = (
        listings["make"]
        .dropna()
        .astype(str)
        .unique()
    )

    for make in sorted(available_makes, key=len, reverse=True):
        if re.search(
            rf"(?<!\w){re.escape(make.lower())}(?!\w)",
            text,
        ):
            filters["make"] = make
            break

    available_models = (
        listings["model"]
        .dropna()
        .astype(str)
        .unique()
    )

    for model in sorted(available_models, key=len, reverse=True):
        if re.search(
            rf"(?<!\w){re.escape(model.lower())}(?!\w)",
            text,
        ):
            filters["model_name"] = model
            break

    # --------------------------------------------------
    # 2. Extract price or monthly-payment constraints
    # --------------------------------------------------

    amount = (
    r"(?:aed\s*)?"
    r"(\d[\d,]*(?:\.\d+)?)"
    r"\s*(k)?"
)

    def parse_amount(match):
        value = float(match.group(1).replace(",", ""))

        if match.group(2):
            value *= 1000

        return value

    monthly = bool(
        re.search(
            r"\b(monthly|per month|a month|installments?|instalments?)\b",
            text,
        )
    )

    constraint_pattern = (
    r"\b(?:under|below|less than|up to|upto|"
    r"maximum of|no more than|max)\s+"
    + amount
)
    price_match = re.search(constraint_pattern, text)

    if price_match:
        value = parse_amount(price_match)

        if monthly:
            filters["max_monthly_payment"] = value
        else:
            filters["max_price"] = value

    # --------------------------------------------------
    # 3. Extract year constraints
    # --------------------------------------------------

    year_range = re.search(
        r"\b(19\d{2}|20\d{2})\s*(?:to|-)\s*(19\d{2}|20\d{2})\b",
        text,
    )

    if year_range:
        filters["min_year"] = int(year_range.group(1))
        filters["max_year"] = int(year_range.group(2))

    else:
        year_from = re.search(
            r"\b(?:from|since|after)\s+(19\d{2}|20\d{2})\b",
            text,
        )

        if year_from:
            filters["min_year"] = int(year_from.group(1))

    # --------------------------------------------------
    # 4. Extract mileage constraints
    # --------------------------------------------------

    mileage_match = re.search(
        r"\b(?:under|below|less than|up to|upto|max)\s+"
        r"(\d[\d,]*)\s*(?:km|kms|kilometres?|kilometers?)\b",
        text,
    )

    if mileage_match:
        filters["max_mileage"] = float(
            mileage_match.group(1).replace(",", "")
        )

    return CarSearchFilters(**filters)