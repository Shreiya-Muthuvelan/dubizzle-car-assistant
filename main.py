
"""
GET / health -> confirms if the API is running
GET / card -> calls hybrid_search() and returns result as JSON
GET / cars / {listing_id} -> endpoint retrieves a specific listing by its ID, it will be useful when the assistant needs to provide additional info
"""
from typing import Optional
import pandas as pd
from pydantic import BaseModel,Field
from fastapi import FastAPI, HTTPException, Query
from src.hybrid_retrieval import hybrid_search
from src.semantic_retrieval import listings
from chat_service import chat


class ChatRequest(BaseModel):
    message: str = Field(min_length=1,max_length=2000,
        description="The user's message to the automotive assistant.",)

    top_k: int = Field(default=5,ge=1,le=10,
        description="Maximum number of vehicle listings to retrieve.",)
    
    conversation_id: Optional[str] = Field(default=None,
        min_length=1,max_length=100,description=(
            "Identifier for an existing conversation. "
            "Omit it to start a new conversation."),)

app = FastAPI(title="Dubizzle Automotive Assistant API",description="API for searching car listings using hybrid retrieval.",
              version="1.0.0",)


@app.get("/health")
def health_check():
    """Check whether the API is running."""
    return {"status": "healthy","service": "Dubizzle Automotive Assistant", }


@app.get("/cars")
def search_cars_endpoint(
    query: Optional[str] = None,make: Optional[str] = None,model_name: Optional[str] = None,
    min_year: Optional[int] = Query(default=None, ge=1900, le=2100),
    max_year: Optional[int] = Query(default=None, ge=1900, le=2100),
    min_price: Optional[float] = Query(default=None, ge=0),
    max_price: Optional[float] = Query(default=None, ge=0),
    min_monthly_payment: Optional[float] = Query(default=None, ge=0),
    max_monthly_payment: Optional[float] = Query(default=None, ge=0),
    min_mileage: Optional[float] = Query(default=None, ge=0),
    max_mileage: Optional[float] = Query(default=None, ge=0),
    top_k: int = Query(default=10, ge=1, le=50),):
    """Search car listings using structured filters and semantic ranking."""

    if min_year is not None and max_year is not None:
        if min_year > max_year:
            raise HTTPException(status_code=400,detail="min_year cannot exceed max_year.",)

    if min_price is not None and max_price is not None:
        if min_price > max_price:
            raise HTTPException(status_code=400,detail="min_price cannot exceed max_price.",)

    if (min_monthly_payment is not None and max_monthly_payment is not None):
        if min_monthly_payment > max_monthly_payment:
            raise HTTPException(status_code=400,detail=("min_monthly_payment cannot exceed ""max_monthly_payment."),)

    if min_mileage is not None and max_mileage is not None:
        if min_mileage > max_mileage:
            raise HTTPException(status_code=400, detail="min_mileage cannot exceed max_mileage.",)

    try:
        results = hybrid_search(query=query,make=make,model_name=model_name,
            min_year=min_year,max_year=max_year,min_price=min_price,max_price=max_price,
            min_monthly_payment=min_monthly_payment,max_monthly_payment=max_monthly_payment,
            min_mileage=min_mileage,max_mileage=max_mileage,top_k=top_k, )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Convert missing Pandas values to JSON-compatible null values.
    results = results.astype(object).where(pd.notna(results), None)

    return {"count": len(results),"results": results.to_dict(orient="records"),}


@app.get("/cars/{listing_id}")
def get_car(listing_id: str):
    """Retrieve one listing by its ID."""

    matches = listings[listings["listing_id"].astype(str) == listing_id]
    if matches.empty:
        raise HTTPException(status_code=404,detail=f"No listing found with ID {listing_id}.",)

    listing = matches.iloc[0]

    # Return the original listing fields rather than internal
    # embedding-related fields.
    output_columns = [ "listing_id", "make", "model", "trim", "year", "title", "description",
        "photo_url","price_aed","monthly_payment_aed","mileage_km",]

    record = listing[output_columns].astype(object)
    record = record.where(pd.notna(record), None)
    return record.to_dict()

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Understand a user's message and maintain conversation memory."""
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=422,
            detail="Message cannot be empty or contain only whitespace.", )

    try:
        return chat(message=message, top_k=request.top_k,
            conversation_id=request.conversation_id,)

    except ValueError as exc:
        raise HTTPException(status_code=400,detail=str(exc),) from exc

    except Exception as exc:
        # Keep detailed errors in the server logs, not the API response.
        print(f"Chat endpoint error: {exc}")
        raise HTTPException(status_code=502, detail="The assistant could not process your request.",) from exc

