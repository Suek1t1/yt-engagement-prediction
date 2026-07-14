# thumbnail_features_1000.csvのユニークなラベルの種類数を数えるコード
import pandas as pd
import ast
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# データの読み込み
df = pd.read_csv("thumbnail_features_1000.csv")

# 文字列になっているリストを、本物のリスト型に変換する
# (もしデータフレーム読み込み時点でリスト型ならこの処理は不要です)
def convert_to_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df['labels_list'] = df['labels'].apply(convert_to_list)

# 全てのラベルを1つのリストに集約し、重複を排除（setを使用）
all_labels_set = set()
for label_list in df['labels_list']:
    all_labels_set.update(label_list)

print(f"ユニークなラベルの総数: {len(all_labels_set)} 種類")

# （おまけ）出現回数順にカウントして上位を確認する
from collections import Counter
all_labels_flat = [label for label_list in df['labels_list'] for label in label_list]
label_counts = Counter(all_labels_flat)

print("\n--- 上位20個の頻出ラベル ---")
for label, count in label_counts.most_common(20):
    print(f"{label}: {count}回")

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