import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

IN_PATH = "methods/embeddings/egypt_edu_sentences_embeddings.parquet"
OUT_DIR = "methods/pca"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(IN_PATH)
    X = np.vstack(df["embedding"].values)

    n = X.shape[0]
    k = min(10, n)
    pca = PCA(n_components=k, random_state=42)
    Z = pca.fit_transform(X)

    out = df.drop(columns=["embedding"]).copy()
    for i in range(Z.shape[1]):
        out[f"pc{i+1}"] = Z[:, i]

    scores_path = os.path.join(OUT_DIR, "egypt_edu_sent_pca_scores.tsv")
    out.to_csv(scores_path, sep="\t", index=False)

    # Explained variance
    plt.figure()
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained variance ratio")
    plt.title("Egypt education sentences — PCA explained variance")
    plt.tight_layout()
    var_path = os.path.join(OUT_DIR, "egypt_edu_pca_variance.png")
    plt.savefig(var_path, dpi=200)

    # Scatter
    if Z.shape[1] >= 2:
        plt.figure()
        plt.scatter(Z[:,0], Z[:,1])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Egypt education sentences — PCA scatter (PC1 vs PC2)")
        plt.tight_layout()
        sc_path = os.path.join(OUT_DIR, "egypt_edu_pca_scatter_pc1_pc2.png")
        plt.savefig(sc_path, dpi=200)

    print("WROTE:", scores_path)
    print("WROTE:", var_path)
    if Z.shape[1] >= 2:
        print("WROTE:", sc_path)
    print("Explained variance ratios:", pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()
