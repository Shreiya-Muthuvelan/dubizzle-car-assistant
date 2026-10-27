from pathlib import Path
import pandas as pd

"""
some design choice used here-
* make and model filtering is case-insensitive
* initial text search uses exact substring matching
* listing id is only an internal identifier


"""

data_file=Path("data/car_listings_cleaned.csv")

def load_listings() -> pd.DataFrame:
    """Load and validate car listings"""
    if not data_file.exists():
        raise FileNotFoundError(f"Processed dataset not found")
    df=pd.read_csv(data_file)
    required_columns=["make","model","trim","year","title","description","search_text","price_aed",
                      "monthly_payment_aed","mileage_km"]
    
    numeric_columns=["year","price_aed","monthly_payment_aed","mileage_km"]
    for column in numeric_columns:
        df[column]=pd.to_numeric(df[column],errors="coerce")

    for column in ["make","model","trim"]:
        df[column]=df[column].fillna("").astype(str).str.strip()

    # create a new identifer for each listing
    df=df.reset_index(drop=True)
    df.insert(0,"listing_id",df.index.astype(str))
    return df

def search_cars( make: str | None = None, model: str | None = None, min_year: int | None = None,
                 max_year: int | None = None, min_price: float | None = None, max_price: float | None = None,
                min_monthly_payment: float | None = None, max_monthly_payment: float | None = None, min_mileage: float | None = None,
                  max_mileage: float | None = None, query: str | None = None, top_k: int = 10, ) -> pd.DataFrame:
    """Search for car listings using filters or text matching
    Missing numerical values are not included when the filter is applied and the listing cannot be verified"""

    if top_k < 1: 
        raise ValueError("top_k must be at least 1.") 
    if min_price is not None and max_price is not None: 
        if min_price > max_price: 
            raise ValueError("min_price cannot exceed max_price.")
    if min_mileage is not None and max_mileage is not None: 
        if min_mileage > max_mileage:
            raise ValueError("min_mileage cannot exceed max_mileage.") 
    if min_year is not None and max_year is not None: 
        if min_year > max_year: 
            raise ValueError("min_year cannot exceed max_year.")

    df=load_listings()
    results=df.copy()

    # manufacturer filter
    if make:
        results=results[results["make"].str.casefold() == make.strip().casefold()]

    # model filter
    if model:
        results = results[ results["model"].str.casefold() == model.strip().casefold() ]

    # Year filters. 
    if min_year is not None: 
        results = results[ results["year"].notna() & (results["year"] >= min_year) ] 
    if max_year is not None:
        results = results[ results["year"].notna() & (results["year"] <= max_year) ]
    
    # Selling-price filters. 
    if min_price is not None: 
        results = results[ results["price_aed"].notna() & (results["price_aed"] >= min_price) ] 
    if max_price is not None: 
        results = results[ results["price_aed"].notna() & (results["price_aed"] <= max_price) ] 
    # Monthly-payment filters. 
    if min_monthly_payment is not None: 
        results = results[ results["monthly_payment_aed"].notna() & ( results["monthly_payment_aed"] >= min_monthly_payment ) ] 
    if max_monthly_payment is not None:
        results = results[ results["monthly_payment_aed"].notna() & ( results["monthly_payment_aed"] <= max_monthly_payment ) ]
    # Mileage filters. 
    if min_mileage is not None: 
        results = results[ results["mileage_km"].notna() & (results["mileage_km"] >= min_mileage) ] 
    if max_mileage is not None: 
        results = results[ results["mileage_km"].notna() & (results["mileage_km"] <= max_mileage) ] 
    # Initial text retrieval: case-insensitive substring matching.
    if query and query.strip(): 
        query_text = query.strip().casefold() 
        results = results[ results["search_text"] .fillna("") .str.casefold() .str.contains(query_text, regex=False) ] 
    return results.head(top_k).reset_index(drop=True) 

if __name__ == "__main__": 
    listings = load_listings() 
    print(f"Loaded {len(listings)} listings.") 
    print("\nAvailable makes:") 
    print(sorted(listings["make"].unique())) 
    print("\nBMW listings:") 
    print( search_cars(make="BMW")[ ["listing_id", "make", "model", "year", "price_aed"] ].to_string(index=False) )