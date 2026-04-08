import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings", "symptoms_embeddings.npy")
METADATA_PATH = os.path.join(BASE_DIR, "data", "embeddings", "symptoms_metadata.csv")


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(METADATA_PATH)

    # basic alignment check
    if len(df) != len(embeddings):
        raise ValueError(
            f"Metadata rows ({len(df)}) do not match embedding rows ({len(embeddings)})"
        )

    return df, embeddings


def get_top_k_similar(df, embeddings, query_index, k=5):
    """
    Return the top-k most similar rows to the query row,
    excluding the query row itself.
    """
    query_vector = embeddings[query_index].reshape(1, -1)

    similarities = cosine_similarity(query_vector, embeddings)[0]

    # sort descending by similarity
    sorted_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in sorted_indices:
        if idx == query_index:
            continue

        results.append({
            "index": idx,
            "similarity": similarities[idx],
            "focus": df.iloc[idx]["focus"],
            "source": df.iloc[idx]["source"],
            "clinical_note": df.iloc[idx]["clinical_note"],
        })

        if len(results) == k:
            break

    return results


def print_similarity_results(df, embeddings, query_index, k=5):
    query_row = df.iloc[query_index]

    print("\n" + "=" * 100)
    print(f"QUERY INDEX: {query_index}")
    print(f"FOCUS: {query_row['focus']}")
    print(f"SOURCE: {query_row['source']}")
    print(f"CLINICAL NOTE: {query_row['clinical_note']}")
    print("=" * 100)

    results = get_top_k_similar(df, embeddings, query_index, k=k)

    for rank, result in enumerate(results, start=1):
        print(f"\nTop {rank} match")
        print(f"Index: {result['index']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Focus: {result['focus']}")
        print(f"Source: {result['source']}")
        print(f"Clinical note: {result['clinical_note']}")


def main():
    df, embeddings = load_data()

    print("Metadata shape:", df.shape)
    print("Embeddings shape:", embeddings.shape)

    example_indices = [0, 1, 10]

    for idx in example_indices:
        if idx < len(df):
            print_similarity_results(df, embeddings, idx, k=5)


if __name__ == "__main__":
    main()