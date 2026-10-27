"""
what this can do :
* find a vehicle by listing id
* validate a requested date and time
* reject invalid bookings
* save confirmed appointments to csv
* return a confirmation containing booking ref
"""

import csv
from datetime import date, datetime, time, timezone
from pathlib import Path
from uuid import uuid4
import re

from src.semantic_retrieval import listings
from gemini_client import extract_booking_request
from lead_qualification import lead_sessions

BOOKINGS_FILE = Path("data/bookings.csv")

BOOKING_FIELDS = [
    "booking_id",
    "conversation_id",
    "listing_id",
    "vehicle",
    "customer_name",
    "contact",
    "booking_date",
    "booking_time",
    "status",
    "created_at",
]

# Temporary state for booking conversations.
booking_sessions: dict[str, dict] = {}


def get_or_create_booking_session(conversation_id: str) -> dict:
    """Retrieve an active booking session or create one."""

    if conversation_id not in booking_sessions:
        booking_sessions[conversation_id] = {
            "booking_id": str(uuid4()),
            "conversation_id": conversation_id,
            "listing_id": None,
            "vehicle": None,
            "customer_name": None,
            "contact": None,
            "booking_date": None,
            "booking_time": None,
            "status": "collecting",
            "awaiting": None,
            "saved": False,
        }

    return booking_sessions[conversation_id]


def get_vehicle(listing_id: str) -> dict | None:
    """Find a vehicle using its listing ID."""

    matches = listings[
        listings["listing_id"].astype(str) == str(listing_id)
    ]

    if matches.empty:
        return None

    vehicle = matches.iloc[0]

    return {
        "listing_id": str(vehicle["listing_id"]),
        "make": str(vehicle["make"]),
        "model": str(vehicle["model"]),
    }


def validate_booking_datetime(
    booking_date: str,
    booking_time: str,
) -> tuple[bool, str]:
    """
    Validate the requested viewing date and time.

    Expected formats:
        booking_date: YYYY-MM-DD
        booking_time: HH:MM (24-hour format)
    """

    try:
        requested_date = date.fromisoformat(booking_date)
    except (TypeError, ValueError):
        return False, "Please provide the date in YYYY-MM-DD format."

    try:
        requested_time = time.fromisoformat(booking_time)
    except (TypeError, ValueError):
        return False, "Please provide the time in HH:MM format."

    today = datetime.now().date()

    if requested_date < today:
        return False, "That date is in the past. Please choose another date."

    # Monday = 0; Sunday = 6.
    if requested_date.weekday() == 6:
        return False, "Viewings are unavailable on Sundays. Please choose another day."

    opening_time = time(8, 0)
    closing_time = time(20, 0)

    if not opening_time <= requested_time <= closing_time:
        return False, (
            "Viewing appointments are available between "
            "8:00 AM and 8:00 PM. Please choose another time."
        )

    # If the booking is for today, reject a time that has already passed.
    if requested_date == today:
        current_time = datetime.now().time().replace(tzinfo=None)

        if requested_time <= current_time:
            return False, (
                "That time has already passed today. "
                "Please choose a later time."
            )

    return True, "The requested date and time are valid."


def save_booking(booking: dict) -> None:
    """Append a confirmed booking to the CSV file."""

    BOOKINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = BOOKINGS_FILE.exists()

    with BOOKINGS_FILE.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=BOOKING_FIELDS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            field: booking.get(field)
            for field in BOOKING_FIELDS
        })


def create_booking(
    conversation_id: str,
    listing_id: str,
    booking_date: str,
    booking_time: str,
    customer_name: str | None = None,
    contact: str | None = None,
) -> dict:
    """Validate and save a viewing appointment."""

    vehicle = get_vehicle(listing_id)

    if vehicle is None:
        return {
            "status": "error",
            "message": (
                f"I couldn't find a vehicle with listing ID {listing_id}. "
                "Please select a valid listing."
            ),
        }

    valid, validation_message = validate_booking_datetime(
        booking_date,
        booking_time,
    )

    if not valid:
        return {
            "status": "error",
            "message": validation_message,
        }

    booking = {
        "booking_id": str(uuid4()),
        "conversation_id": conversation_id,
        "listing_id": vehicle["listing_id"],
        "vehicle": f'{vehicle["make"]} {vehicle["model"]}',
        "customer_name": customer_name,
        "contact": contact,
        "booking_date": booking_date,
        "booking_time": booking_time,
        "status": "confirmed",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    save_booking(booking)

    return {
        "status": "confirmed",
        "booking": booking,
        "message": (
            f'Your viewing for {booking["vehicle"]} is confirmed '
            f'on {booking_date} at {booking_time}. '
            f'Your booking reference is {booking["booking_id"]}.'
        ),
    }



def parse_booking_date(message: str) -> str | None:
    """
    Extract a date from a message and return it as YYYY-MM-DD.

    Supported examples:
    - 2026-12-05
    - 5/12/2026 or 05-12-2026 (DD/MM/YYYY)
    - 5 December 2026
    - 5th December 2026
    - December 5, 2026
    - 5th Dec
    - tomorrow
    - next Monday
    """
    text = message.strip().lower()

    # Handle relative dates.
    today = date.today()

    if re.search(r"\btomorrow\b", text):
        return date.fromordinal(today.toordinal() + 1).isoformat()

    weekday_match = re.search(
        r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        text,
    )

    if weekday_match:
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        target_weekday = weekdays[weekday_match.group(1)]
        days_ahead = (target_weekday - today.weekday()) % 7

        # "Next Monday" means the Monday in the following week,
        # rather than today if today happens to be Monday.
        if days_ahead == 0:
            days_ahead = 7

        return date.fromordinal(today.toordinal() + days_ahead).isoformat()

    # Handle ISO dates: 2026-12-05
    match = re.search(r"\b(\d{4}-\d{1,2}-\d{1,2})\b", text)

    if match:
        try:
            return date.fromisoformat(match.group(1)).isoformat()
        except ValueError:
            return None

    # Handle numeric dates: 05/12/2026 or 05-12-2026.
    # These are interpreted as DD/MM/YYYY.
    match = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b", text)

    if match:
        date_text = match.group(1)

        for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(date_text, fmt).date().isoformat()
            except ValueError:
                continue

        return None

    # Remove ordinal suffixes: 1st, 2nd, 3rd, 5th, etc.
    normalised_text = re.sub(
        r"\b(\d{1,2})(st|nd|rd|th)\b",
        r"\1",
        text,
    )

    # Handle dates such as:
    # 5 December 2026
    # 5 Dec 2026
    # December 5, 2026
    date_patterns = (
        (r"\b\d{1,2}\s+[a-z]+\s+\d{4}\b", ("%d %B %Y", "%d %b %Y")),
        (r"\b[a-z]+\s+\d{1,2},?\s+\d{4}\b", ("%B %d %Y", "%b %d %Y")),
    )

    for pattern, formats in date_patterns:
        match = re.search(pattern, normalised_text)

        if not match:
            continue

        date_text = match.group(0).replace(",", "")

        for fmt in formats:
            try:
                return datetime.strptime(date_text, fmt).date().isoformat()
            except ValueError:
                continue

        return None

    # Handle dates without a year, e.g. 5th Dec or 5 December.
    # Assume the next upcoming occurrence.
    match = re.search(
        r"\b(\d{1,2})\s+([a-z]+)\b",
        normalised_text,
    )

    if match:
        date_text = f"{match.group(1)} {match.group(2)}"

        for fmt in ("%d %B", "%d %b"):
            try:
                parsed = datetime.strptime(date_text, fmt).date()
                candidate = parsed.replace(year=today.year)

                if candidate < today:
                    candidate = candidate.replace(year=today.year + 1)

                return candidate.isoformat()
            except ValueError:
                continue

    return None


def parse_booking_time(message: str) -> str | None:
    """
    Extract a time from a message and return it as HH:MM.

    Supported examples:
    - 9 AM
    - 9:00 AM
    - 9:00am
    - at 3 PM
    - 14:30
    - 20:00
    - I want it for 9:00 AM
    """
    text = message.strip().lower()

    # Match 12-hour times, with or without minutes.
    # Examples: 9 AM, 9:00 AM, 9:00am, 12:30 pm
    match = re.search(
        r"\b(1[0-2]|0?[1-9])"
        r"(?::([0-5]\d))?\s*(a\.?m\.?|p\.?m\.?)\b",
        text,
    )

    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        period = match.group(3)[0]

        if period == "p" and hour != 12:
            hour += 12
        elif period == "a" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"

    # Match 24-hour times such as 09:00, 14:30 or 20:00.
    match = re.search(
        r"\b([01]?\d|2[0-3]):([0-5]\d)\b",
        text,
    )

    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))

        return f"{hour:02d}:{minute:02d}"

    return None




def fallback_extract_booking_request(message: str) -> dict:
    """
    Locally identify basic booking intent and extract a listing ID.
    Used when Gemini booking extraction fails.
    """

    text = message.lower().strip()

    booking_phrases = (
        "book a viewing",
        "book viewing",
        "booking",
        "schedule a viewing",
        "arrange a viewing",
        "arrange viewing",
        "view this car",
        "viewing for this car",
        "test drive",
        "book an appointment",
    )

    intent = (
        "booking"
        if any(phrase in text for phrase in booking_phrases)
        else "other"
    )

    listing_id = None

    # Recognise explicit listing IDs, e.g. "listing ID 66".
    match = re.search(
        r"\blisting\s*(?:id|number|no\.?)\s*[:#]?\s*(\d+)\b",
        text,
    )

    if match:
        listing_id = match.group(1)

    return {
        "intent": intent,
        "listing_id": listing_id,
        "booking_date": parse_booking_date(message),
        "booking_time": parse_booking_time(message),
    }


def get_booking_request(message: str):
    """
    Try Gemini first, then use local extraction if Gemini fails.
    """

    try:
        return extract_booking_request(message)

    except Exception as exc:
        print(f"Gemini booking extraction failed: {exc}")
        print("Using local booking-intent fallback.")

        fallback = fallback_extract_booking_request(message)

        # Return the same general interface used by the booking flow.
        return type(
            "BookingRequestFallback",
            (),
            fallback,
        )()


def is_new_search_during_booking(message: str) -> bool:
    """Detect when a user wants to start a new car search during booking."""
    text = message.lower().strip()

    search_phrases = (
        "show me",
        "show some",
        "show cars",
        "show vehicles",
        "find me",
        "find cars",
        "search for",
        "look for",
        "browse",
        "what other cars",
        "which other cars",
        "other cars",
        "another car",
        "different car",
    )

    return any(phrase in text for phrase in search_phrases)


def process_booking_message(
    conversation_id: str,
    message: str,
    customer_name: str | None = None,
    contact: str | None = None,
) -> dict | None:
    """
    Process a new viewing request or continue an active booking.

    Gemini is used to identify new booking intent.
    Standard date/time follow-ups are parsed locally.
    """

    session = booking_sessions.get(conversation_id)

  
    # Allow the user to start a new car search during an unfinished booking.
    if (
        session is not None
        and not session.get("saved")
        and is_new_search_during_booking(message)
    ):
        booking_sessions.pop(conversation_id, None)
        print(
            "User started a new car search. "
            "Cleared the unfinished booking session."
        )
        return None


    # --------------------------------------------------
    # 1. Handle an existing booking session
    # --------------------------------------------------
    if session is not None:

        # Prevent a confirmed booking from being saved again.

# A completed booking should only handle messages about that booking.
        if session.get("saved"):
            text = message.lower().strip()

            booking_status_phrases = (
                "booking status",
                "booking confirmation",
                "my booking",
                "my appointment",
                "my viewing",
                "booking reference",
                "booking id",
            )

            if any(phrase in text for phrase in booking_status_phrases):
                booking = session.get("booking")

                if booking:
                    return {
                        "status": "confirmed",
                        "booking_id": booking["booking_id"],
                        "booking": booking,
                        "message": (
                            "Your viewing is confirmed for "
                            f'{booking["booking_date"]} at '
                            f'{booking["booking_time"]}. '
                            f'Your booking ID is {booking["booking_id"]}.'
                        ),
                    }

                return {
                    "status": "confirmed",
                    "booking_id": session["booking_id"],
                    "message": "This booking has already been confirmed.",
                }

            # Let unrelated messages continue through normal chat routing.
            return None



        # The current question tells us what kind of reply to expect.
        awaiting = session.get("awaiting")

        if awaiting == "booking_date":
            parsed_date = parse_booking_date(message)

            if parsed_date is None:
                return {
                    "status": "collecting",
                    "booking_id": session["booking_id"],
                    "message": (
                        "I couldn't recognise that date. "
                        "Please provide it as YYYY-MM-DD, "
                        "for example 2026-10-28."
                    ),
                }

            session["booking_date"] = parsed_date

        elif awaiting == "booking_time":
            parsed_time = parse_booking_time(message)

            if parsed_time is None:
                return {
                    "status": "collecting",
                    "booking_id": session["booking_id"],
                    "message": (
                        "I couldn't recognise that time. "
                        "Please provide it as HH:MM or "
                        "a time such as 4 PM."
                    ),
                }

            session["booking_time"] = parsed_time

        elif awaiting == "booking_date_or_time":
            # After validation fails, allow the user to correct
            # either the date or the time.
            parsed_date = parse_booking_date(message)

            if parsed_date is not None:
                session["booking_date"] = parsed_date
            else:
                parsed_time = parse_booking_time(message)

                if parsed_time is not None:
                    session["booking_time"] = parsed_time
                else:
                    return {
                        "status": "collecting",
                        "booking_id": session["booking_id"],
                        "message": (
                            "Please provide a valid date "
                            "or time so I can correct the booking."
                        ),
                    }

        elif awaiting == "listing_id":
            # For an ID-only reply, use it directly rather than
            # making another Gemini intent-extraction call.
            listing_id = message.strip()

            vehicle = get_vehicle(listing_id)

            if vehicle is None:
                return {
                    "status": "collecting",
                    "booking_id": session["booking_id"],
                    "message": (
                        "I couldn't find that listing. "
                        "Please provide a valid listing ID."
                    ),
                }

            session["listing_id"] = vehicle["listing_id"]
            session["vehicle"] = (
                f'{vehicle["make"]} {vehicle["model"]}'
            )

        else:
            # The current session is not waiting for a known field.
            # Check whether this is a new booking request.
            request = get_booking_request(message)

            if request.intent != "booking":
                return None

            if request.listing_id:
                vehicle = get_vehicle(request.listing_id)

                if vehicle is None:
                    return {
                        "status": "collecting",
                        "booking_id": session["booking_id"],
                        "message": (
                            "I couldn't find that listing. "
                            "Please provide a valid listing ID."
                        ),
                    }

                session["listing_id"] = vehicle["listing_id"]
                session["vehicle"] = (
                    f'{vehicle["make"]} {vehicle["model"]}'
                )

            if request.booking_date:
                session["booking_date"] = request.booking_date

            if request.booking_time:
                session["booking_time"] = request.booking_time

        if customer_name:
            session["customer_name"] = customer_name

        if contact:
            session["contact"] = contact

    # --------------------------------------------------
    # 2. Identify a new booking request
    # --------------------------------------------------
    else:
        request = get_booking_request(message)

        if request.intent != "booking":
            return None

        session = get_or_create_booking_session(conversation_id)

        if request.listing_id:
            vehicle = get_vehicle(request.listing_id)

            if vehicle is None:
                session["awaiting"] = "listing_id"

                return {
                    "status": "collecting",
                    "booking_id": session["booking_id"],
                    "message": (
                        "I couldn't find that listing. "
                        "Please provide a valid listing ID."
                    ),
                }

            session["listing_id"] = vehicle["listing_id"]
            session["vehicle"] = (
                f'{vehicle["make"]} {vehicle["model"]}'
            )

        if request.booking_date:
            session["booking_date"] = request.booking_date

        if request.booking_time:
            session["booking_time"] = request.booking_time

        if customer_name:
            session["customer_name"] = customer_name

        if contact:
            session["contact"] = contact

    # --------------------------------------------------
    # 3. Ask for missing booking details
    # --------------------------------------------------
    if not session["listing_id"]:
        session["awaiting"] = "listing_id"

        return {
            "status": "collecting",
            "booking_id": session["booking_id"],
            "message": (
                "Which car would you like to view? "
                "Please share its listing ID."
            ),
        }

    if not session["booking_date"]:
        session["awaiting"] = "booking_date"

        return {
            "status": "collecting",
            "booking_id": session["booking_id"],
            "message": (
                f'You would like to view the {session["vehicle"]}. '
                "What date would you prefer?"
            ),
        }

    if not session["booking_time"]:
        session["awaiting"] = "booking_time"

        return {
            "status": "collecting",
            "booking_id": session["booking_id"],
            "message": (
                "What time would you prefer? "
                "Viewings are available between 8:00 AM and 8:00 PM, "
                "Monday through Saturday."
            ),
        }

    # --------------------------------------------------
    # 4. Validate and save the booking
    # --------------------------------------------------
    result = create_booking(
        conversation_id=conversation_id,
        listing_id=session["listing_id"],
        booking_date=session["booking_date"],
        booking_time=session["booking_time"],
        customer_name=session["customer_name"],
        contact=session["contact"],
    )

    if result["status"] == "error":
        session["awaiting"] = "booking_date_or_time"

        return {
            "status": "collecting",
            "booking_id": session["booking_id"],
            "message": (
                f'{result["message"]} '
                "Please provide a valid date or time."
            ),
        }

    session["status"] = "confirmed"
    session["saved"] = True
    session["awaiting"] = None
    session["booking"] = result["booking"]

    return result

