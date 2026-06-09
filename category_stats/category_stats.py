"""
カテゴリーごとの likes・views・dislikes の平均・中央値を集計するスクリプト
出力: category_stats.csv
"""

import pandas as pd

# ── 1. データ読み込み ────────────────────────────────────────────
df = pd.read_csv("english_titles.csv")
df = df.drop_duplicates(subset="video_id").reset_index(drop=True)

# 数値変換
for col in ["likes", "views", "dislikes"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["likes", "views", "dislikes"])

# ── 2. カテゴリーID → 名称のマッピング ──────────────────────────
category_map = {
    1:  "Film & Animation",
    2:  "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    18: "Short Movies",
    19: "Travel & Events",
    20: "Gaming",
    21: "Videoblogging",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism",
}

# ── 3. カテゴリーごとに集計 ──────────────────────────────────────
stats = df.groupby("category_id").agg(
    likes_mean   =("likes",    "mean"),
    likes_median =("likes",    "median"),
    views_mean   =("views",    "mean"),
    views_median =("views",    "median"),
    dislikes_mean  =("dislikes", "mean"),
    dislikes_median=("dislikes", "median"),
    count        =("likes",    "count"),
).reset_index()

# 小数点以下を整数に丸める
for col in stats.columns[1:]:
    stats[col] = stats[col].round(0).astype(int)

stats["category_name"] = stats["category_id"].map(category_map).fillna("Unknown")

# category_name を category_id の直後に移動
cols = ["category_id", "category_name"] + [c for c in stats.columns if c not in ["category_id", "category_name"]]
stats = stats[cols].sort_values("category_id").reset_index(drop=True)

# ── 3. 表示・CSV出力 ─────────────────────────────────────────────
print(stats.to_string(index=False))
stats.to_csv("category_stats.csv", index=False)
print("\nCSV保存完了: category_stats.csv")
