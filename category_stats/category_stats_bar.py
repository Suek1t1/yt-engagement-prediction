"""
category_stats.csv を読み込み、likes・dislikes・views の
平均・中央値をカテゴリーごとに棒グラフで出力するスクリプト
出力: category_likes_bar.png, category_dislikes_bar.png, category_views_bar.png
"""

import pandas as pd
import matplotlib.pyplot as plt

# ── 1. CSV読み込み ───────────────────────────────────────────────
df = pd.read_csv("category_stats/category_stats.csv")
labels = df["category_name"]
x = range(len(labels))
width = 0.4

# ── 2. グラフ描画関数 ────────────────────────────────────────────
def plot_bar(mean_col, median_col, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar([i - width/2 for i in x], df[mean_col],   width=width, label="Mean",   color="steelblue")
    ax.bar([i + width/2 for i in x], df[median_col], width=width, label="Median", color="orange")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"保存完了: {filename}")
    plt.close()

# ── 3. 各指標のグラフを出力 ─────────────────────────────────────
plot_bar("likes_mean",    "likes_median",    "Likes by Category",    "Likes",    "category_likes_bar.png")
plot_bar("dislikes_mean", "dislikes_median", "Dislikes by Category", "Dislikes", "category_dislikes_bar.png")
plot_bar("views_mean",    "views_median",    "Views by Category",    "Views",    "category_views_bar.png")
