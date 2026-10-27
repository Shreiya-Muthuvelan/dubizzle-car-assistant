import numpy as np
from .semantic_retrieval import (listings,listing_embeddings,model,)


def hybrid_search(query=None,make=None,model_name=None,min_year=None,
    max_year=None,min_price=None,max_price=None,min_monthly_payment=None,max_monthly_payment=None,
    min_mileage=None,max_mileage=None,top_k=10,):
    """Apply structured filters and rank matching cars semantically."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    
    df = listings.copy()

    # Apply exact text filters
    if make:
        df = df[df["make"].str.casefold() == make.strip().casefold()]

    if model_name:
        df = df[df["model"].str.casefold() == model_name.strip().casefold()]

    # Apply year filters.
    if min_year is not None:
        df = df[df["year"].notna() & (df["year"] >= min_year)]

    if max_year is not None:
        df = df[df["year"].notna() & (df["year"] <= max_year)]

    # Apply selling price filters
    if min_price is not None:
        df = df[df["price_aed"].notna()& (df["price_aed"] >= min_price)]

    if max_price is not None:
        df = df[df["price_aed"].notna() & (df["price_aed"] <= max_price)]

    # Apply monthly payment filters
    if min_monthly_payment is not None:
        df = df[df["monthly_payment_aed"].notna() & (df["monthly_payment_aed"] >= min_monthly_payment)]

    if max_monthly_payment is not None:
        df = df[ df["monthly_payment_aed"].notna() & (df["monthly_payment_aed"] <= max_monthly_payment)]

    # Apply mileage filters
    if min_mileage is not None:
        df = df[df["mileage_km"].notna()& (df["mileage_km"] >= min_mileage)]

    if max_mileage is not None:
        df = df[df["mileage_km"].notna() & (df["mileage_km"] <= max_mileage)]

    # No candidates satisfy the hard constraints
    if df.empty:
        return df.assign(similarity_score=[])

    # If no query was supplied, return the eligible listings
    # without semantic ranking.
    if not query or not query.strip():
        return df.head(top_k).reset_index(drop=True)

    # Encode the query and calculate similarity against all listings.
    query_embedding = model.encode([query],normalize_embeddings=True,)[0]

    scores = listing_embeddings @ query_embedding

    # Match candidate rows to their corresponding embedding scores.
    score_by_id = dict(zip(listings["listing_id"], scores))

    df["similarity_score"] = df["listing_id"].map(score_by_id)

    return (df.sort_values("similarity_score", ascending=False).head(top_k).reset_index(drop=True) )


if __name__ == "__main__":
    results = hybrid_search(
        query="A sporty car for weekend drives",
        make="BMW",
        max_price=250000,max_mileage=50000,top_k=5,)

    print("\nHybrid retrieval results:")

    if results.empty:
        print("No listings satisfy the specified constraints.")
    else:
        print( results[["listing_id","make",  "model","year","price_aed","mileage_km","similarity_score", ]].to_string(index=False) )