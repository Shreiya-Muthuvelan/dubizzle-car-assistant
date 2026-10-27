"""
We will use this text to first clean the text and then extract uesful attributes using python regex matching

No LLM has been involved in this as :

 * dataset has only 100 listings
 * regex is sufficient for many predictable formats
 * it avoids unecessary API calls and potenial hallucinations

Functions that we will be doing in this script
* clean HTML from descriptions
* deduplicate record that have matching title+description pairs
* extract stated selling price, monthly payment amounts, mileage etc
* save the preprocessed dataset
"""
import pandas as pd
import re
import html
from bs4 import BeautifulSoup
from pathlib import Path

input_file=Path("data/car_listings.xlsx")
output_file=Path("data/car_listings_cleaned.csv")


def clean_description(text):
    """remove html markup and normalize whitespaces"""
    if pd.isna(text):
        return ""
    text=str(text)

    # detecting html entites and convert to text
    text=html.unescape(text)
    text=BeautifulSoup(text,"html.parser").get_text(" ",strip=True)

    text=re.sub(r"\s+", " ", text).strip()
    return text

def extract_selling_price(text):
    """ Extract a stated total selling price for the car. We will match the following patterns :
    1. 'AED 115750 in cash'
    2. 'Price : 329000 AED'
    3. Any larger amount that is not related to monthly payment
    
    """
    if pd.isna(text):
        return None
    text = str(text)

    # 1. Explicit cash price, e.g. AED 115,750 in cash
    cash_pattern = (r"(?i)\bAED\s*([\d,]+(?:\.\d{1,2})?)"
                    r"\s*(?:in cash|cash price)\b" )
    match = re.search(cash_pattern, text)
    if match:
        return float(match.group(1).replace(",", ""))

    # 2. Explicit price label, e.g. Price: 329,000 AED
    price_pattern = ( r"(?i)\bprice\s*:\s*"
        r"(?:AED\s*)?([\d,]+(?:\.\d{1,2})?)"
        r"\s*(?:AED)?\b")
    match = re.search(price_pattern, text)
    if match:
        return float(match.group(1).replace(",", ""))
    
    # 3. Amount followed by AED, e.g. 249,000 AED
    # Exclude amounts associated with monthly payments.
    amount_pattern = ( r"(?i)\b([\d,]+(?:\.\d{1,2})?)\s*AED\b")
    for match in re.finditer(amount_pattern, text):
        amount = float(match.group(1).replace(",", ""))
        if amount < 10000:
            continue
        # Inspect nearby context to avoid extracting installments.
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        context = text[start:end].lower()
        if re.search( r"monthly|per month|/month|\bp\.?\s*m\.?\b",context):
            continue
        return amount

    return None


def extract_monthly_payment(text):
    """
    Extract stated monthly payment. We wil match the following patterns :
    1. 'AED 2040 monthly, 'AED 1611/month', '4567 AED / Month
    2. 'AED 35800 P.M', '4000 P.M.', Starting from 1611 per month"""
    if pd.isna(text):
        return None

    text = str(text)
    patterns = [
        # AED 1,611/month
        r"(?i)\bAED\s*([\d,]+(?:\.\d{1,2})?)\s*/\s*month\b",
        # AED 3,580 P.M
        r"(?i)\bAED\s*([\d,]+(?:\.\d{1,2})?)\s*P\.?\s*M\.?\b",
        # AED 2,040 monthly
        r"(?i)\bAED\s*([\d,]+(?:\.\d{1,2})?)\s*monthly\b",
        # Starting from 1,611 per month
        r"(?i)\b(?:starting from|from)\s*"
        r"([\d,]+(?:\.\d{1,2})?)\s*(?:AED\s*)?per month\b",
        # AED 1,430 / Month
        r"(?i)\b([\d,]+(?:\.\d{1,2})?)\s*AED\s*/\s*month\b",]

    candidates = []
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidates.append(match)
    
    if not candidates:
        return None
    
    match = min(candidates, key=lambda item: item.start())
    return float(match.group(1).replace(",", ""))
                
def extract_mileage(text):
    """
    Extract mileage in km . Match patterns of format:
    'mileage/odometer' , 'km/kms'"""
    if not text:
        return None
    text = str(text)

    # Normalize whitespace while preserving the surrounding context.
    text = re.sub(r"\s+", " ", text)

    # Match numbers such as 37,062 km, 37062 kms, or 37.062 km.
    number = r"(\d{1,3}(?:[,\.]\d{3})+|\d+)"
    unit = r"\s*(?:km|kms|kilometres|kilometers)\b"

    # Prefer explicit mileage/odometer labels.
    explicit_pattern = re.compile(rf"\b(?:mileage|odometer)\s*[:\-]?\s*{number}{unit}",re.IGNORECASE,)

    for match in explicit_pattern.finditer(text):
        value = int(match.group(1).replace(",", "").replace(".", ""))

        # Check nearby preceding context for warranty/service references.
        prefix = text[max(0, match.start() - 100):match.start()].lower()

        if re.search(r"warranty|service plan|service contract|service interval",prefix,):
            continue

        return value

    #  Find kilometre values and reject misleading contexts.
    candidate_pattern = re.compile(rf"{number}{unit}",re.IGNORECASE,)

    for match in candidate_pattern.finditer(text):
        value = int( match.group(1).replace(",", "").replace(".", ""))

        # Examine text around the candidate to identify its meaning.
        context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)].lower()

        # Ignore warranty coverage and service-related distances.
        if re.search(r"warranty|service plan|service contract|"r"service interval|covered until|coverage", context,):
            continue

        # Ignore advertised electric driving range.
        if re.search(r"\brange\b|single charge|driving range",context, ):
            continue

        # Ignore financing percentages mistakenly associated with km.
        if re.search(r"\bdown[\s-]?payment\b", context):
            continue

        if re.search(
            r"\bmileage\b|\bodometer\b|"
            r"\bkm only\b|\bkilometres only\b|"
            r"\bkilometers only\b",
            context,):
            return value

    return None

def preprocess_listings(input_file=input_file):
    df=pd.read_excel(input_file)
    df["clean_description"]=df["description"].apply(clean_description)
    # we create new column called clean description instead of overwriting since it will make it easier to investigate errors later
    
    # We look for duplicates with matching title+ description and keep only the first row
    df_clean = df.drop_duplicates(subset=["title", "clean_description"]).copy()
    df_clean = df_clean.reset_index(drop=True)

    df_clean["search_text"] = (df_clean["title"].fillna("").astype(str)+ " "
    + df_clean["clean_description"].fillna("").astype(str) )

    # Add structured attributes
    df_clean["price_aed"] = df_clean["search_text"].apply(extract_selling_price)
    df_clean["monthly_payment_aed"] = df_clean["search_text"].apply(extract_monthly_payment)
    df_clean["mileage_km"] = df_clean["search_text"].apply(extract_mileage)
    
    return df, df_clean

def main():
    """Run preprocessing pipeline and save the preprocessed dataset"""
    df_original,df_clean=preprocess_listings(input_file)
    output_file.parent.mkdir(parents=True,exist_ok=True)
    df_clean.to_csv(output_file,index=False)

    print("Preprocessing complete")
    print(f"Original listings: {len(df_original)}")
    print(f"Listings after deduplication: {len(df_clean)}")
    print(f"Rows removed: {len(df_original) - len(df_clean)}")
    print(f"Selling prices extracted: {df_clean['price_aed'].notna().sum()}")
    print( "Monthly installments extracted: "
            f"{df_clean['monthly_payment_aed'].notna().sum()}")
    print(f"Mileage values extracted: {df_clean['mileage_km'].notna().sum()}")
    print(f"Saved processed data to: {output_file}")
    
    
if __name__ == "__main__":
    main()