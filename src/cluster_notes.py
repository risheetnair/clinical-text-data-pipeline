import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings", "symptoms_embeddings.npy")
METADATA_PATH = os.path.join(BASE_DIR, "data", "embeddings", "symptoms_metadata.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "clusters")

N_CLUSTERS = 4
RANDOM_STATE = 42


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(METADATA_PATH)

    if len(df) != len(embeddings):
        raise ValueError(
            f"Metadata rows ({len(df)}) do not match embedding rows ({len(embeddings)})"
        )

    return df, embeddings


def run_kmeans(embeddings, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    cluster_labels = model.fit_predict(embeddings)
    return model, cluster_labels


def summarize_clusters(df):
    print("\nCluster counts:")
    print(df["cluster"].value_counts().sort_index())

    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]

        print("\n" + "=" * 100)
        print(f"CLUSTER {cluster_id} | size={len(subset)}")
        print("=" * 100)

        preview_cols = ["focus", "source", "clinical_note"]
        print(subset[preview_cols].head(5).to_string(index=False))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading embeddings and metadata...")
    df, embeddings = load_data()
    print("Metadata shape:", df.shape)
    print("Embeddings shape:", embeddings.shape)

    print(f"\nRunning KMeans with k={N_CLUSTERS}...")
    model, cluster_labels = run_kmeans(embeddings)

    df["cluster"] = cluster_labels

    print("Inertia:", model.inertia_)

    summarize_clusters(df)

    output_path = os.path.join(OUTPUT_DIR, f"clustered_notes_k{N_CLUSTERS}.csv")
    df.to_csv(output_path, index=False)

    print(f"\nSaved clustered dataset to: {output_path}")


if __name__ == "__main__":
    main()