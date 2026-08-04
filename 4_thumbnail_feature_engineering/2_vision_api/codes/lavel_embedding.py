# ラベルの埋め込みとk-meansでクラスタリングをするコード
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd

# 準備：ユニークなラベルのリストを取得（先ほどの集計結果を使用）
# all_labels_set は先ほどのコードで取得した集合です
labels = list(all_labels_set)

# モデルのロード（軽量かつ高性能な日本語/多言語対応モデル）
# これで単語をベクトル（数値の列）に変換します
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(labels)

# k-meansでクラスタリング
# n_clusters=50 は「50個のグループに分類する」という意味です
# ここを調整すると、分類の細かさを変えられます
num_clusters = 50 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)

# 結果の整理
df_clusters = pd.DataFrame({'label': labels, 'cluster': kmeans.labels_})

# 各クラスタの中身を確認
for i in range(num_clusters):
    cluster_labels = df_clusters[df_clusters['cluster'] == i]['label'].tolist()
    print(f"\n--- グループ {i} (代表: {cluster_labels[0]}) ---")
    # 表示が長くなるので最初の10個だけ表示
    print(cluster_labels[:10])

# 5. CSVとして保存（これを手作業のルール作りの土台にします）
df_clusters.to_csv("label_clusters.csv", index=False, encoding='utf-8-sig')