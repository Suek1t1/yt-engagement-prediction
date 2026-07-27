import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

# 使用アルゴリズム：ランダムフォレスト
# viewなし
# 特徴量追加(!,?,大文字,投稿日の処理,サムネイル文字数)

# ==========================================
# 1. CSV読み込みと友達の画像データの結合
# ==========================================
print("--- データを読み込み中 ---")

df = pd.read_csv("english_titles.csv")

# 友達の画像データの読み込みと必要な列の抽出
df_vision = pd.read_csv("thumbnail_features_1000.csv")
vision_cols = ['video_id', 'extracted_text', 'labels', 'web_entities']
available_cols = [c for c in vision_cols if c in df_vision.columns]

# 💡 パターンA: 画像特徴量がある約900件だけに絞り込んで検証する場合 (how="inner")
# ※全3.7万件で保存したい場合は how="left" に変更してください
df = pd.merge(df, df_vision[available_cols], on="video_id", how="inner")
print(f"対象データ件数: {len(df)} 件")

# ==========================================
# 2. 欠損値の事前処理
# ==========================================
df["description"] = df["description"].fillna("")
df["tags"] = df["tags"].fillna("")

# ==========================================
# 3. 新しい特徴量の作成（投稿前にわかる情報）
# ==========================================
# テキスト系特徴量
df["title_len"] = df["title"].str.len()
df["desc_len"] = df["description"].str.len()
df["tag_count"] = df["tags"].str.count(r"\|") + 1
df["tag_len"] = df["tags"].str.len()

# 記号・大文字フラグ
df["has_exclamation"] = df["title"].str.contains(r"!").astype(int)
df["has_question"] = df["title"].str.contains(r"\?").astype(int)
df["is_uppercase"] = df["title"].str.isupper().astype(int)

# 日時系の処理
df["publish_time"] = pd.to_datetime(df["publish_time"])
df["publish_hour"] = df["publish_time"].dt.hour
df["publish_dayofweek"] = df["publish_time"].dt.dayofweek

# カテゴリIDのターゲットエンコーディング
category_target_mean = df.groupby("category_id")["likes"].mean()
df["category_mean_likes"] = df["category_id"].map(category_target_mean)

# 画像（サムネイル）からのテキスト・ラベル特徴量
df["vision_text_len"] = df["extracted_text"].fillna("").astype(str).str.len()
df["vision_label_len"] = df["labels"].fillna("").astype(str).str.len()
df["vision_web_len"] = df["web_entities"].fillna("").astype(str).str.len()

# ==========================================
# 💾 【新機能】作成した特徴量をCSVファイルとして保存！
# ==========================================
#output_filename = "engineered_features.csv"
#df.to_csv(output_filename, index=False, encoding="utf-8-sig")
#print(f"✅ 特徴量を含むデータを '{output_filename}' に保存しました！")


# ==========================================
# 4. 特徴量(X)とターゲット(y)の選定
# ==========================================
feature_cols = [
    "title_len",
    "desc_len",
    "tag_count",
    "tag_len",
    "publish_hour",
    "publish_dayofweek",
    "category_mean_likes",
    "has_exclamation",
    "has_question",
    "is_uppercase",
    "vision_text_len",   # サムネイル文字数
    "vision_label_len",  # サムネイルラベル長
    "vision_web_len",    # Web関連ワード長
]

X = df[feature_cols]
y = np.log1p(df["likes"])

# ==========================================
# 5. データ分割とモデル学習
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)

# ==========================================
# 6. 予測と評価
# ==========================================
y_pred = model.predict(X_test)

y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)

mask = y_test_original >= 10
y_test_filtered = y_test_original[mask]
y_pred_filtered = y_pred_original[mask]

print("\n--- モデル評価結果 ---")
print(f"平均エラー（何いいねズレているか）: {mean_absolute_error(y_test_original, y_pred_original):.1f}")
print(f"R2スコア（予測の正確さ 0~1）: {r2_score(y_test, y_pred):.4f}")
print(f"平均二乗誤差 (MSE): {mean_squared_error(y_test_original, y_pred_original):.4f}")
print(f"平均パーセント誤差: {mean_absolute_percentage_error(y_test_filtered, y_pred_filtered):.4f}")

# 特徴量の重要度を表示
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\n【投稿前データの中での重要度】")
print(importances.sort_values(ascending=False))

# グラフ化
plt.scatter(y_test, y_pred, alpha=0.5)
max_val = max(max(y_test), max(y_pred))
plt.plot([0, max_val], [0, max_val], color="red", linestyle="--")
plt.xlabel("Real Likes (Log)")
plt.ylabel("Predicted Likes (Log)")
plt.show()