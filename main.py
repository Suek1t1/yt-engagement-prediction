import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("USvideos.csv - Sheet1.csv")

# 整数化
df["views"] = df["views"].astype(int)
df["likes"] = df["likes"].astype(int)
df["category_id"] = df["category_id"].astype(int)

# category一覧
categories = sorted(df["category_id"].unique())

# 8個ずつに分割
chunk_size = 8
chunks = [categories[i:i+chunk_size] for i in range(0, len(categories), chunk_size)]
print(len(categories))

# グラフ枚数
num_figs = len(chunks)
print(len(chunks))

for idx, chunk in enumerate(chunks):
    plt.figure(figsize=(12, 8))

    for category in chunk:
        subset = df[df["category_id"] == category]

        if category in [10, 24]:
            continue

        plt.scatter(
            subset["views"],
            subset["likes"],
            label=f"Category {category}",
            alpha=0.5
        )

    plt.xlabel("Views")
    plt.ylabel("Likes")
    plt.title(f"Views vs Likes (Category group {idx+1})")
    plt.legend()
    plt.grid(True)

    # 保存
    plt.savefig(f"scatter_group_{idx+1}.png", dpi=300, bbox_inches="tight")
    plt.close()