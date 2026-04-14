import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings", "symptoms_embeddings.npy")
METADATA_PATH = os.path.join(BASE_DIR, "data", "embeddings", "symptoms_metadata.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(METADATA_PATH)

    if len(df) != len(embeddings):
        raise ValueError(
            f"Metadata rows ({len(df)}) do not match embedding rows ({len(embeddings)})"
        )

    return df, embeddings


def truncate_text(text: str, max_len: int = 120) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def get_top_k_similar(df, embeddings, query_index, k=3):
    """
    Return the top-k most similar rows to the query row,
    excluding the query row itself.
    """
    query_vector = embeddings[query_index].reshape(1, -1)
    similarities = cosine_similarity(query_vector, embeddings)[0]

    sorted_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in sorted_indices:
        if idx == query_index:
            continue

        results.append({
            "rank": len(results) + 1,
            "query_index": query_index,
            "match_index": idx,
            "similarity": round(float(similarities[idx]), 4),
            "focus": df.iloc[idx]["focus"],
            "source": df.iloc[idx]["source"],
            "clinical_note": df.iloc[idx]["clinical_note"],
        })

        if len(results) == k:
            break

    return results


def print_similarity_results(df, embeddings, query_index, k=3):
    query_row = df.iloc[query_index]
    results = get_top_k_similar(df, embeddings, query_index, k=k)

    print("\n" + "=" * 90)
    print(f"QUERY: {query_row['focus']} ({query_row['source']})")
    print(f"NOTE:  {truncate_text(query_row['clinical_note'], 160)}")
    print("=" * 90)

    result_df = pd.DataFrame(results)[["rank", "focus", "source", "similarity"]]
    print(result_df.to_string(index=False))

    print("\nTop match note previews:")
    for row in results:
        print(f"\n#{row['rank']} {row['focus']} | score={row['similarity']}")
        print(f"   {truncate_text(row['clinical_note'], 140)}")


def save_similarity_examples(df, embeddings, query_indices, k=3):
    """
    Save clean similarity outputs to CSV for slide creation.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []
    for query_index in query_indices:
        if query_index >= len(df):
            continue

        query_row = df.iloc[query_index]
        results = get_top_k_similar(df, embeddings, query_index, k=k)

        for row in results:
            all_rows.append({
                "query_focus": query_row["focus"],
                "query_source": query_row["source"],
                "query_note": query_row["clinical_note"],
                "rank": row["rank"],
                "match_focus": row["focus"],
                "match_source": row["source"],
                "similarity": row["similarity"],
                "match_note": row["clinical_note"],
            })

    output_path = os.path.join(OUTPUT_DIR, "similarity_examples.csv")
    pd.DataFrame(all_rows).to_csv(output_path, index=False)
    print(f"\nSaved similarity examples to: {output_path}")


def main():
    df, embeddings = load_data()

    print("Metadata shape:", df.shape)
    print("Embeddings shape:", embeddings.shape)

    # Pick a few representative examples for presentation
    example_indices = [0, 10, 25]

    for idx in example_indices:
        if idx < len(df):
            print_similarity_results(df, embeddings, idx, k=3)

    save_similarity_examples(df, embeddings, example_indices, k=3)


if __name__ == "__main__":
    main()