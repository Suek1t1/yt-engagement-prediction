import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

df = pd.read_csv("english_titles.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

X_embed = model.encode(
    df["title"].astype(str).tolist(),
    show_progress_bar=True
)

pca = PCA(n_components=20, random_state=42)

X_pca = pca.fit_transform(X_embed)

pca_df = pd.DataFrame(
    X_pca,
    columns=[f"emb_{i+1}" for i in range(20)]
)

pca_df.insert(
    0,
    "video_id",
    df["video_id"]
)


pca_df.to_csv(
    "embedding_features_20.csv",
    index=False
)