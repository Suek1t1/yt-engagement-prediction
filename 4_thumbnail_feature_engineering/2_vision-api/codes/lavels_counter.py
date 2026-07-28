# thumbnail_features_1000.csvのユニークなラベルの種類数を数えるコード
import pandas as pd
import ast
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# データの読み込み
df = pd.read_csv("datasets/thumbnail_features_1000.csv")

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
