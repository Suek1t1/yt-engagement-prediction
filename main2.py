import pandas as pd
import matplotlib.pyplot as plt
import math

#実行するとファイルがたくさん増えます注意
df = pd.read_csv("USvideos.csv - Sheet1.csv")

# 数値化
df["views"] = df["views"].astype(int)
df["likes"] = df["likes"].astype(int)
df["category_id"] = df["category_id"].astype(int)

# ★ここ追加（最大値チェック）
print("max category_id:", df["category_id"].max())
print("category_id length:", len(df["category_id"]))

# =========================
# ① tagsを「個数」に変換
# =========================
# "a, b, c" -> 3
df["tag_count"] = df["tags"].fillna("").apply(
    lambda x: 0 if x.strip() == "" else len([t for t in x.split("|") if t.strip() != ""])
)

# ★ここ追加（tag_count作成後）
print("max tag_count:", df["tag_count"].max())
print("tag_count length:", len(df["tag_count"]))
print("unique category_id:", df["category_id"].nunique())
# =========================
# ② (category_id, tag_count) を直積ID化
# =========================
# 例: category 22 × tag_count 5 → 22*100 + 5 みたいに圧縮
# （100はtag数の最大想定より十分大きくする）
df["cat_tag_id"] = df["category_id"] * 100 + df["tag_count"]

# =========================
# ユニークID一覧
# =========================
combos = sorted(df["cat_tag_id"].unique())

# 適当に分割（多すぎると見づらいので）
chunk_size = 8
chunks = [combos[i:i+chunk_size] for i in range(0, len(combos), chunk_size)]

print("unique categories:", df["category_id"].nunique())
print("unique (category, tag_count):", len(combos))
print("unique tag_count:", df["tag_count"].nunique())

# =========================
# ③ 描画
# =========================
for idx, chunk in enumerate(chunks):
    plt.figure(figsize=(12, 8))

    for combo in chunk:
        subset = df[df["cat_tag_id"] == combo]

        plt.scatter(
            subset["views"],
            subset["likes"],
            label=f"cat_tag {combo}",
            alpha=0.5
        )

    plt.xlabel("Views")
    plt.ylabel("Likes")
    plt.title(f"Views vs Likes (cat_tag group {idx+1})")
    plt.legend()
    plt.grid(True)

    #plt.savefig(f"png_id_tags/scatter_cat_tag_group_{idx+1}.png", dpi=300, bbox_inches="tight")
    plt.close()