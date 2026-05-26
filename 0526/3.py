import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 元のCSVファイルから読み込む
file_path = '/Users/goyataichi/Library/Application Support/Claude/local-agent-mode-sessions/ca346680-c659-44bc-9d67-36a1951a87db/331f6740-ebff-44b5-837d-cda0cbc648fe/local_0076a054-d831-4113-bcf6-653105de2476/uploads/USvideos.csv - Sheet1.csv'

df = pd.read_csv(file_path)

print("データ読み込み完了")
print(f"総ビデオ数: {len(df)} 件")

# グラフを作成
fig, ax = plt.subplots(figsize=(14, 10))

# 散布図をプロット
scatter = ax.scatter(df['views'], df['likes'], 
                     alpha=0.5, s=40, c=df['dislikes'], 
                     cmap='plasma', edgecolors='none')

# X軸とY軸の両方に対数スケールを適用
ax.set_xscale('log')
ax.set_yscale('log')

# ラベルとタイトル
ax.set_xlabel('Views (再生回数) - Log Scale', fontsize=14, fontweight='bold')
ax.set_ylabel('Likes (高評価数) - Log Scale', fontsize=14, fontweight='bold')
ax.set_title('YouTube Videos: Views vs Likes (Log-Log Scale)', fontsize=16, fontweight='bold')

# グリッド表示
ax.grid(True, alpha=0.3, which='both')

# カラーバーを追加
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Dislikes (低評価数)', fontsize=12)

plt.tight_layout()

# グラフを保存
output_path = '/Users/goyataichi/Documents/info-dm-g5/0526/views_likes_loglog.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ グラフを保存しました: {output_path}")

# 統計情報を表示
print("\n=== Views vs Likes 統計情報 ===")
correlation = df['views'].corr(df['likes'])
print(f"相関係数: {correlation:.4f}")

print(f"\nViews (再生回数):")
print(f"  最小値: {df['views'].min():,.0f}")
print(f"  最大値: {df['views'].max():,.0f}")
print(f"  平均値: {df['views'].mean():,.0f}")
print(f"  中央値: {df['views'].median():,.0f}")

print(f"\nLikes (高評価数):")
print(f"  最小値: {df['likes'].min():,.0f}")
print(f"  最大値: {df['likes'].max():,.0f}")
print(f"  平均値: {df['likes'].mean():,.0f}")
print(f"  中央値: {df['likes'].median():,.0f}")

print(f"\nDislikes (低評価数):")
print(f"  最小値: {df['dislikes'].min():,.0f}")
print(f"  最大値: {df['dislikes'].max():,.0f}")
print(f"  平均値: {df['dislikes'].mean():,.0f}")
print(f"  中央値: {df['dislikes'].median():,.0f}")

# グラフを表示
plt.show()
print("\n完了！")