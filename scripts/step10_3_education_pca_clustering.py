import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path

BASE = Path("data/derived/step10_education_dataset")
INPUT = BASE / "education_country_dataset.csv"

print("[step10_3] loading:", INPUT)

df = pd.read_csv(INPUT)

features = [
    "n_norm_share",
    "n_rule_share",
    "n_strategy_share",
    "statements_per_doc"
]

X = df[features].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
pcs = pca.fit_transform(X_scaled)

df["pca1"] = pcs[:,0]
df["pca2"] = pcs[:,1]
df["pca3"] = pcs[:,2]

print("[step10_3] PCA variance:", pca.explained_variance_ratio_)

kmeans = KMeans(n_clusters=2, random_state=42)
df["cluster_id"] = kmeans.fit_predict(X_scaled)

OUT = BASE / "education_governance_clusters.csv"
df.to_csv(OUT, index=False)

print("[step10_3] wrote:", OUT)
print("[step10_3] countries:", len(df))
