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
    # --- 【変更点】通常グラフと拡大グラフの2枚を保存するようにします ---
    
    # 拡大パターンを2つ（通常サイズとズームサイズ）ループで回す
    for is_zoom in [False, True]:
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
        plt.grid(True)

        if is_zoom:
            # ズーム用の設定（ご提示の条件に合わせて範囲を制限）
            plt.xlim(-100000, 10000000) # 横軸：0〜1,000万 (少しマイナス側も入れて見やすく)
            plt.ylim(-5000, 500000)     # 縦軸：0〜50万
            plt.title(f"Views vs Likes [ZOOM] (Category group {idx+1})")
            plt.legend(loc='upper left')
            
            # 拡大版として保存
            plt.savefig(f"scatter_group_{idx+1}_zoom.png", dpi=300, bbox_inches="tight")
        else:
            # 通常版として保存
            plt.title(f"Views vs Likes (Category group {idx+1})")
            plt.legend()
            plt.savefig(f"scatter_group_{idx+1}.png", dpi=300, bbox_inches="tight")
            
        plt.close()