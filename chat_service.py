
from gemini_client import (CarSearchFilters,extract_search_filters,generate_car_response,)
from src.hybrid_retrieval import hybrid_search
from lead_qualification import process_lead_message
from conversation_memory import (get_or_create_conversation,merge_search_filters,update_conversation)
import time
import pandas as pd
import re
from booking_service import process_booking_message

def looks_like_booking_request(message: str) -> bool:
    """Identify messages that are likely to request a vehicle viewing."""

    text = message.lower()

    booking_phrases = ["book", "booking", "view","viewing", "arrange a viewing", "schedule a viewing", "test drive",]

    return any(phrase in text for phrase in booking_phrases)

def is_greeting(message: str) -> bool:
    """Identify simple greetings without calling Gemini."""

    text = message.lower().strip()
    greetings = { "hi", "hello", "hey", "hi there","hello there", "good morning", "good afternoon", "good evening","hey there",}

    return text in greetings
def is_out_of_scope(message: str) -> bool:
    """Identify clearly unrelated requests."""

    text = message.lower().strip()

    out_of_scope_phrases = (  "write me a poem","write a poem","solve this math","solve this equation", 
        "help me with my homework","help with my homework","calculus assignment","calculus homework",
        "help me with calculus","math assignment","math homework","solve this calculus problem",
        "debug my code","write python code","write me a story","tell me a joke",
        "what is the weather","who is the president",)

    return any(phrase in text for phrase in out_of_scope_phrases)

DATA_PATH = "data/car_listings_cleaned.csv"

def mentions_known_vehicle(message: str) -> bool:
    """Check whether a message explicitly mentions a make or model in our dataset."""
    df = pd.read_csv(DATA_PATH)
    text = message.lower()
    vehicle_terms = set()

    for column in ["make", "model", "trim"]:
        if column in df.columns:
            values = df[column].dropna().astype(str).unique()

            for value in values:
                value = value.strip().lower()

                if value:
                    vehicle_terms.add(value)

    # Check longer vehicle names first, such as "Mercedes-Benz C-Class".
    for term in sorted(vehicle_terms, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"

        if re.search(pattern, text):
            return True

    return False

def answer_vehicle_followup(message: str,listings: list[dict]) -> str | None:

    # Explicit vehicle mentions should go through normal retrieval.
    if mentions_known_vehicle(message):
        return None

    text = message.lower().strip()

    # If the user names a specific make/model, let normal retrieval
    # identify the correct listing instead of using the previous results.
    explicit_vehicle_phrases = ( "bmw", "toyota","nissan", "mercedes",
        "mercedes-benz","audi", "honda", "hyundai",
        "kia","ford","lexus","mitsubishi",
        "mazda","volkswagen","porsche","land rover", "range rover",
        "tesla", "chevrolet", "jeep", "volvo","corolla",
        "camry","civic","accord","3 series","5 series","7 series",
        "x3", "x5", "m3", "m5",)

    if any(phrase in text for phrase in explicit_vehicle_phrases):
        return None

    # These phrases indicate a reference to a previously displayed vehicle.
    references_previous_car = any(
        phrase in text
        for phrase in ( "this car","that car", "this vehicle", "that vehicle",
            "this listing", "that listing",
            "its price", "it's price", "its mileage", "its year",
            "its monthly payment","what year is it","what year is this","what is its year",
            "how much is it","how much does it cost",
            "what is the price","what is the mileage","what is the monthly payment",
            "what is its mileage", "what is its monthly payment","tell me more",
            "more details","describe this car",)  )

    if not references_previous_car or not listings:
        return None

    # This handler uses the first result only when the user refers
    # to a previously displayed vehicle without specifying another one.
    car = listings[0]

    # Selling price.
    if any(word in text for word in ("price", "cost", "expensive", "cheap")):
        price = car.get("price_aed")

        if price is None:
            return "The selling price isn't available in this listing."

        return ( f"The selling price of the {car.get('make', '')} "
            f"{car.get('model', '')} is AED {float(price):,.0f}.")

    # Monthly payment.
    if any(word in text for word in ("monthly payment", "installment", "instalment")):
        payment = car.get("monthly_payment_aed")
        if payment is None:
            return "The monthly payment isn't available in this listing."

        return f"The monthly payment is AED {float(payment):,.0f}."

    # Mileage.
    if any(word in text for word in ("mileage", "kilomet", "kilometer")):
        mileage = car.get("mileage_km")
        if mileage is None:
            return "The mileage isn't available in this listing."

        return ( f"The listed mileage of the {car.get('make', '')} " f"{car.get('model', '')} is {float(mileage):,.0f} km." )

    # Manufacturing year.
    if "year" in text or "old is" in text:
        year = car.get("year")

        if year is None:
            return "The manufacturing year isn't available in this listing."

        return ( f"The listed manufacturing year of the "
                f"{car.get('make', '')} {car.get('model', '')} is {year}."  )

    # Trim.
    if "trim" in text or "variant" in text:
        trim = car.get("trim")

        if not trim:
            return "The trim or variant isn't specified in this listing."

        return f"The listed trim or variant is {trim}."

    # General details.
    if any(
        phrase in text
        for phrase in ("more details","tell me more",
            "describe this",
            "description",
            "what features", ) ):
        description = car.get("description")

        if description:
            return ( f"Here are the details for the "
                    f"{car.get('make', '')} {car.get('model', '')}: "
                f"{description}" )

        return "This listing doesn't provide an additional vehicle description."

    return None


def chat(message: str, top_k: int = 5,conversation_id:str | None=None) -> dict:
    """Process a car-search message while maintaining conversation memory
    and support lead qualification"""

    conversation_id,state=get_or_create_conversation(conversation_id)
        # Handle active booking conversations before extracting car-search filters.
    if is_greeting(message):
        answer = ("Hi! I'm your automotive assistant. I can help you find cars, "
            "compare listings, check available vehicle details, and arrange "
            "a viewing. What are you looking for?")

        update_conversation( conversation_id=conversation_id, user_message=message,
            assistant_message=answer, search_filters=state["search_filters"],  )

        return {
            "conversation_id": conversation_id,"message": message,
            "filters": state["search_filters"], "count": 0,
            "results": [], "answer": answer,
            "lead": None,"booking": None,}
    if is_out_of_scope(message):
        answer = ("I'm here to help with cars and automotive-related questions. "
            "I can help you find vehicles, compare listings, understand "
            "vehicle details, and arrange a viewing. What would you like "
            "to know about a car?" )

        update_conversation( conversation_id=conversation_id,user_message=message,
            assistant_message=answer, search_filters=state["search_filters"], )

        return {
            "conversation_id": conversation_id, "message": message,
            "filters": state["search_filters"], "count": 0,
            "results": [],"answer": answer,"lead": None, "booking": None, }
    # Handle simple follow-up questions about previously retrieved vehicles.
    previous_listings = state.get("last_listings", [])

    followup_answer = answer_vehicle_followup(message=message,listings=previous_listings, )

    if followup_answer is not None:
        update_conversation(conversation_id=conversation_id,
            user_message=message,assistant_message=followup_answer,search_filters=state["search_filters"],)

      
        return {"conversation_id": conversation_id,
    "message": message,"filters": state["search_filters"],
    "count": 0,"results": [],"answer": followup_answer,"lead": None,"booking": None,}


    from booking_service import booking_sessions

    if conversation_id in booking_sessions:
        booking_result = process_booking_message( conversation_id=conversation_id,message=message, )

        if booking_result is not None:
            answer = booking_result["message"]

            update_conversation( conversation_id=conversation_id,user_message=message,
                assistant_message=answer, search_filters=state["search_filters"], )

            return {
                "conversation_id": conversation_id,
                "message": message, "filters": state["search_filters"],
                "count": 0,"results": [],
                "answer": answer,"lead": None,
                "booking": {
                    "status": booking_result["status"],
                    "booking_id": booking_result.get("booking_id")
                    or booking_result.get("booking", {}).get("booking_id"),  },}
        # Handle explicit new booking requests before car-search extraction.
    if looks_like_booking_request(message):
        booking_result = process_booking_message(conversation_id=conversation_id, message=message,)

        if booking_result is not None:
            answer = booking_result["message"]

            update_conversation( conversation_id=conversation_id, user_message=message,
                assistant_message=answer, search_filters=state["search_filters"], )

            return {
                "conversation_id": conversation_id,"message": message,
                "filters": state["search_filters"], "count": 0,
                "results": [],"answer": answer,
                "lead": None,"booking": {
                    "status": booking_result["status"],
                    "booking_id": booking_result.get("booking_id")
                    or booking_result.get("booking", {}).get("booking_id"),  }, }
    total_start=time.perf_counter()

    print("Extracting search filters with Gemini")
    start = time.perf_counter()

    try:
        extracted_filters = extract_search_filters(message)
    except Exception as exc:
        print(f"Gemini filter extraction failed: {exc}")

        answer = ( "I'm temporarily unable to process your request because my "
            "AI service is unavailable. Please try again shortly." )

        update_conversation( conversation_id=conversation_id,user_message=message,assistant_message=answer,
            search_filters=state["search_filters"],)

        return {"conversation_id": conversation_id, "message": message,
            "filters": state["search_filters"],"count": 0,
            "results": [],"answer": answer,
            "lead": None,"booking": None,}

    print(f"Filter extraction took {time.perf_counter() - start:.2f}")
    print("Extracted filters")
    new_filters=extracted_filters.model_dump()
    previous_filters=state["search_filters"]
    print("DEBUG - Previous filters:", previous_filters)
    print("DEBUG - Newly extracted filters:", new_filters)

    merged_filters=merge_search_filters(previous_filters,new_filters,message)
    filters=CarSearchFilters.model_validate(merged_filters)
    print(f"Updated search filters: {filters.model_dump()}")

    # lead qualification
    lead_result=process_lead_message(conversation_id=conversation_id,
                                     message=message,filters=filters.model_dump())
    if lead_result is not None:
        answer=lead_result["message"]
        update_conversation(conversation_id=conversation_id,user_message=message,
                            assistant_message=answer,search_filters=filters.model_dump())

        print("Lead qualification handled the message")
        print(f"Total chat processing time : {time.perf_counter()-total_start:.2f}")
        return { "conversation_id": conversation_id, "message": message, 
                "filters": filters.model_dump(), "count": 0, "results": [], "answer": answer, 
                "lead": { "status": lead_result["status"], "lead_id": lead_result["lead_id"], 
                         "missing_fields": lead_result.get("missing_fields", []), }, }


    
    search_query = filters.search_query

    print("Running hybrid retrieval")
    start=time.perf_counter()
    results = hybrid_search(
        query=search_query,make=filters.make,model_name=filters.model_name,
        min_year=filters.min_year,max_year=filters.max_year,
        min_price=filters.min_price,max_price=filters.max_price,
        min_monthly_payment=filters.min_monthly_payment,max_monthly_payment=filters.max_monthly_payment,
        min_mileage=filters.min_mileage,max_mileage=filters.max_mileage,
        top_k=top_k,)
    print(f"Hybrid retrieval took {time.perf_counter() - start:.2f}")
    print(f"Retrieved {len(results)} listings")

    # Convert missing Pandas values to JSON-compatible None.
    results = results.astype(object).where(results.notna(), None,)
    listings = results.to_dict(orient="records")

    print("Generating final response with Gemini")
    start = time.perf_counter()

    try:
        answer = generate_car_response( message=message,filters=filters,listings=listings, )
    except Exception as exc:
        print(f"Gemini response generation failed: {exc}")

        answer = ("I found the matching vehicle listings, but I'm temporarily "
            "unable to generate a response. Please try again shortly." )

    print(f"Answer generation took {time.perf_counter() - start:.2f}")
    update_conversation(conversation_id=conversation_id,user_message=message,
    assistant_message=answer,search_filters=filters.model_dump(),listings=listings,)
    
    print("Conversation memory updated")
    total_elapsed=time.perf_counter() - total_start
    print(f"Total chat processing time : {total_elapsed:.2f}")
    print("Generated final response")
    print("Returning API response")
  
    return {
    "conversation_id": conversation_id,"message": message,
    "filters": filters.model_dump(),"count": len(listings),
    "results": listings, "answer": answer,
    "lead":None,}

