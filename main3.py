import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

#実行するとファイルがたくさん増えます注意
for file in glob.glob("png_id_tags/*.png"):
    os.remove(file)
    print("deleted:", file)

df = pd.read_csv("USvideos.csv - Sheet1.csv")

# 数値化
df["views"] = df["views"].astype(int)
df["likes"] = df["likes"].astype(int)
df["category_id"] = df["category_id"].astype(int)
df["comment_count"] = df["comment_count"].fillna(0).astype(int)


# =========================
# comment bin
# =========================
bins = [0, 1, 10, 100, 1000, 10000, 100000, float("inf")]
labels = [0, 1, 2, 3, 4, 5, 6]

df["comment_bin"] = pd.cut(
    df["comment_count"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

df["comment_bin"] = df["comment_bin"].fillna(0).astype(int)

# =========================
# 直積ID
# =========================
df["cat_com_id"] = df["comment_bin"]

# ユニークID
combos = sorted(df["cat_com_id"].unique())

# ★ここが重要：cat_com_idでchunk分割
chunk_size = 8
chunks = [combos[i:i+chunk_size] for i in range(0, len(combos), chunk_size)]

print("unique (category × comment_bin):", len(combos))
print("num chunks:", len(chunks))

# =========================
# 描画
# =========================
for idx, chunk in enumerate(chunks):
    plt.figure(figsize=(12, 8))

    for cid in chunk:
        subset = df[df["cat_com_id"] == cid]

        plt.scatter(
            subset["views"],
            subset["likes"],
            label=f"{cid}",
            alpha=0.5
        )

    plt.xlabel("Views")
    plt.ylabel("Likes")
    plt.title(f"Views vs Likes (cat_com_id group {idx+1})")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"png_id_tags/scatter_group_{idx+1}.png", dpi=300, bbox_inches="tight")
    plt.close()