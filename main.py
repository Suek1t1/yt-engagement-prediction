import pandas as pd
import matplotlib.pyplot as plt

# CSV読み込み
df = pd.read_csv("USvideos.csv - Sheet1.csv")

# 整数型に変換
df["views"] = df["views"].astype(int)
df["likes"] = df["likes"].astype(int)
df["category_id"] = df["category_id"].astype(int)

# グラフサイズ
plt.figure(figsize=(12, 8))

# category_idごとに色分けして散布図
categories = df["category_id"].unique()

for category in categories:
    if category == 10:
        continue
    subset = df[df["category_id"] == category]

    plt.scatter(
        subset["views"],
        subset["likes"],
        label=f"Category {category}",
        alpha=0.6
    )

# 軸ラベル
plt.xlabel("Views")
plt.ylabel("Likes")

# タイトル
plt.title("Views vs Likes by Category")

# 凡例
plt.legend()

# グリッド
plt.grid(True)

# 表示
plt.show()