import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "symptoms_clinical_notes.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "embeddings")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_processed_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # drop rows with missing or blank notes
    df["clinical_note"] = df["clinical_note"].fillna("").astype(str)
    df = df[df["clinical_note"].str.strip() != ""].copy()

    return df


def generate_embeddings(texts, model_name: str = MODEL_NAME):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading processed dataset...")
    df = load_processed_data(PROCESSED_DATA_PATH)
    print("Dataset shape:", df.shape)

    print(f"Loading model: {MODEL_NAME}")
    embeddings = generate_embeddings(df["clinical_note"].tolist())

    print("Embeddings shape:", embeddings.shape)

    # save embeddings
    embeddings_path = os.path.join(OUTPUT_DIR, "symptoms_embeddings.npy")
    np.save(embeddings_path, embeddings)

    # save aligned metadata copy
    metadata_path = os.path.join(OUTPUT_DIR, "symptoms_metadata.csv")
    df.to_csv(metadata_path, index=False)

    print(f"Saved embeddings to: {embeddings_path}")
    print(f"Saved metadata to: {metadata_path}")

    # quick sanity check
    print("\nSample note:")
    print(df.iloc[0]["clinical_note"])
    print("\nEmbedding length:", len(embeddings[0]))


if __name__ == "__main__":
    main()