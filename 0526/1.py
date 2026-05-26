import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 元のCSVファイルから読み込む
file_path = '/Users/goyataichi/Library/Application Support/Claude/local-agent-mode-sessions/ca346680-c659-44bc-9d67-36a1951a87db/331f6740-ebff-44b5-837d-cda0cbc648fe/local_0076a054-d831-4113-bcf6-653105de2476/uploads/USvideos.csv - Sheet1.csv'

df = pd.read_csv(file_path)
views = df['views'].values
likes = df['likes'].values
dislikes = df['dislikes'].values

print("データ読み込み完了")
print(f"データ数: {len(views)} 件")

# 3Dグラフを作成
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 散布図をプロット（カラーマップで低評価数を表現）
scatter = ax.scatter(views, likes, dislikes, c=dislikes, cmap='viridis', marker='o', s=20, alpha=0.6)

# 軸ラベルを設定
ax.set_xlabel('Views (再生回数)', fontsize=12, fontweight='bold')
ax.set_ylabel('Likes (高評価数)', fontsize=12, fontweight='bold')
ax.set_zlabel('Dislikes (低評価数)', fontsize=12, fontweight='bold')
ax.set_title('YouTube Videos: 3D Analysis (Views × Likes × Dislikes)', fontsize=14, fontweight='bold')

# カラーバーを追加
cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('Dislikes', fontsize=10)

# グリッドを表示
ax.grid(True, alpha=0.3)

# 図を保存
output_path = '/Users/goyataichi/Documents/info-dm-g5/0526/3d_graph.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ グラフを保存しました: {output_path}")

# 統計情報を表示
print("\n=== データ統計 ===")
print(f"Views (再生回数):")
print(f"  最小値: {views.min():,.0f}")
print(f"  最大値: {views.max():,.0f}")
print(f"  平均値: {views.mean():,.0f}")
print(f"  中央値: {np.median(views):,.0f}")

print(f"\nLikes (高評価数):")
print(f"  最小値: {likes.min():,.0f}")
print(f"  最大値: {likes.max():,.0f}")
print(f"  平均値: {likes.mean():,.0f}")
print(f"  中央値: {np.median(likes):,.0f}")

print(f"\nDislikes (低評価数):")
print(f"  最小値: {dislikes.min():,.0f}")
print(f"  最大値: {dislikes.max():,.0f}")
print(f"  平均値: {dislikes.mean():,.0f}")
print(f"  中央値: {np.median(dislikes):,.0f}")

# グラフを表示
plt.show()
print("\n完了！")