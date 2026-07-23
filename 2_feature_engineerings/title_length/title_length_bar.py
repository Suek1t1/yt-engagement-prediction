"""
タイトル文字数ごとの登場回数を棒グラフで可視化するスクリプト
出力: title_length_bar.png
"""

import pandas as pd
import matplotlib.pyplot as plt

# ── 1. CSVから読み込み ───────────────────────────────────────────
length_order = pd.read_csv("title_length/title_length_order.csv")

# ── 2. 棒グラフ描画 ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))

ax.bar(length_order["title_length"], length_order["count"], color="steelblue", width=0.8)

ax.set_xlabel("Title Length (characters)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Number of Videos by Title Length", fontsize=14)
ax.set_xlim(0, length_order["title_length"].max() + 1)

plt.tight_layout()
plt.savefig("title_length_bar.png", dpi=150)
print("保存完了: title_length_bar.png")
plt.show()
