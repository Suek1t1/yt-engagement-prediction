# 目的変数である「likes」と、説明変数である「views」「dislikes」「comment_count」の関係を散布図で可視化するコードです。
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd

df = pd.read_csv('USvideos.csv - Sheet1.csv')

# グラフを3つ横に並べるための設定
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# それぞれの散布図を描画
axes[0].scatter(df['views'], df['likes'], alpha=0.3, color='blue')
axes[0].set_title('Views vs Likes')
axes[0].set_xlabel('Views')
axes[0].set_ylabel('Likes')

axes[1].scatter(df['dislikes'], df['likes'], alpha=0.3, color='red')
axes[1].set_title('Dislikes vs Likes')
axes[1].set_xlabel('Dislikes')

axes[2].scatter(df['comment_count'], df['likes'], alpha=0.3, color='green')
axes[2].set_title('Comments vs Likes')
axes[2].set_xlabel('Comment Count')

# レイアウトを整える
plt.tight_layout()
plt.show()