import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings", "symptoms_embeddings.npy")
CLUSTERED_DATA_PATH = os.path.join(BASE_DIR, "data", "clusters", "clustered_notes_k5.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(CLUSTERED_DATA_PATH)

    if len(df) != len(embeddings):
        raise ValueError(
            f"Clustered data rows ({len(df)}) do not match embedding rows ({len(embeddings)})"
        )

    return df, embeddings


def reduce_with_pca(embeddings, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    explained_variance = pca.explained_variance_ratio_
    return reduced, explained_variance


def plot_clusters(df, reduced_embeddings, explained_variance, output_path):
    plt.figure(figsize=(10, 7))

    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        indices = subset.index

        plt.scatter(
            reduced_embeddings[indices, 0],
            reduced_embeddings[indices, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.8
        )

    plt.xlabel(f"PC1 ({explained_variance[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({explained_variance[1] * 100:.1f}% variance)")
    plt.title("PCA Projection of Clinical Note Embeddings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading clustered data and embeddings...")
    df, embeddings = load_data()

    print("Reducing embeddings to 2D with PCA...")
    reduced_embeddings, explained_variance = reduce_with_pca(embeddings)

    print("Explained variance ratio:")
    print(explained_variance)
    print("Total explained variance (2 components):", explained_variance.sum())

    output_path = os.path.join(OUTPUT_DIR, "pca_clusters.png")
    plot_clusters(df, reduced_embeddings, explained_variance, output_path)

    # save coordinates too
    df["pca_x"] = reduced_embeddings[:, 0]
    df["pca_y"] = reduced_embeddings[:, 1]
    df.to_csv(os.path.join(OUTPUT_DIR, "clustered_notes_with_pca.csv"), index=False)

    print(f"Saved PCA plot to: {output_path}")


if __name__ == "__main__":
    main()