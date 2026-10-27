import hashlib
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from retrieval import load_listings


model_name= "sentence-transformers/all-MiniLM-L6-v2"
cache_dir = Path("data")
embeddings_file = cache_dir / "listing_embeddings.npy"
metadata_file = cache_dir/ "listing_embeddings_metadata.json"
model = SentenceTransformer(model_name)


def build_semantic_text(df):
    """Create the text representation used for each listing."""
    return ( "Make: " + df["make"].fillna("").astype(str)
        + ". Model: " + df["model"].fillna("").astype(str)
        + ". Trim: " + df["trim"].fillna("").astype(str)
        + ". Year: " + df["year"].fillna("").astype(str)
        + ". Description: " + df["search_text"].fillna("").astype(str))


def get_dataset_fingerprint(df):
    """Create a fingerprint of the text used to generate embeddings."""
    semantic_text = build_semantic_text(df)
    combined_text = "\n".join(semantic_text.tolist())
    return hashlib.sha256(combined_text.encode("utf-8")).hexdigest()


def build_or_load_embeddings():
    """Load cached embeddings or generate and save them."""
    df = load_listings()
    semantic_text = build_semantic_text(df)
    fingerprint = get_dataset_fingerprint(df)

    cache_dir.mkdir(parents=True, exist_ok=True)

    if embeddings_file.exists() and metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        cache_is_valid = (metadata.get("model_name") == model_name and metadata.get("dataset_fingerprint") == fingerprint
            and metadata.get("listing_ids") == df["listing_id"].tolist())

        if cache_is_valid:
            embeddings = np.load(embeddings_file)
            if embeddings.shape[0] == len(df):
                print("Loaded cached listing embeddings.")
                return df, embeddings

    print("Generating listing embeddings")
    embeddings = model.encode(semantic_text.tolist(), normalize_embeddings=True, show_progress_bar=True,)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(embeddings_file, embeddings)

    metadata = { "model_name": model_name,"dataset_fingerprint": fingerprint,"listing_ids": df["listing_id"].tolist(),}

    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Saved listing embeddings")
    return df, embeddings

listings,listing_embeddings = build_or_load_embeddings()

def semantic_search(query, top_k=10):
    """Rank listings by semantic similarity."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_embedding = model.encode( [query],normalize_embeddings=True,)[0]

    scores = listing_embeddings@ query_embedding
    results = listings.copy()
    results["similarity_score"] = scores

    return (results.sort_values("similarity_score", ascending=False).head(top_k).reset_index(drop=True) )


if __name__ == "__main__":
    query = "A sporty car for weekend drives"
    results = semantic_search(query, top_k=5)
    print(f"\nQuery: {query}")
    print(results[[ "listing_id","make", "model", "trim", "year","similarity_score",]].to_string(index=False))
