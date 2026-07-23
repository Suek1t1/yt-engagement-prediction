# 累積寄与率を計算し、80% と 90% の情報を保持するために必要な次元数を特定するコード
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- 設定：フォントの豆腐文字（文字化け）を防ぐ設定 ---
plt.rcParams['font.family'] = 'sans-serif' 

# --------------------------------------------------
# 1. データの読み込み
# --------------------------------------------------
print("データを読み込んでいます...")
# 無加工のファイル名に適宜書き換えてください
df = pd.read_pickle("sample_10000_data.pkl") 

print("\n=== データ確認 ===")
print(f"元のデータフレームの形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# 'vector' 列の numpy 配列をすべて縦に積み重ねて、[1000, 768] の 2次元行列に変換します
features = np.stack(df['vector'].values)
n_samples, n_features = features.shape
print(f"PCAに入力する特徴量 (features) の形状: {features.shape} (サンプル数: {n_samples}, 元の次元数: {n_features})")
print("=================\n")

# --------------------------------------------------
# 2. PCA（主成分分析）の実行
# --------------------------------------------------
# データの行数（1000）と特徴量の次元数（768）のうち、小さい方を上限（今回は768）にします
max_components = min(n_samples, n_features)

print(f"PCAを実行中（最大次元数: {max_components}）...")
pca = PCA(n_components=max_components)
pca.fit(features)

# 各主成分の寄与率と、その累積寄与率を計算
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# --------------------------------------------------
# 3. 閾値（80% と 90%）を満たす次元数の特定
# --------------------------------------------------
dim_80 = np.argmax(cumulative_variance >= 0.80) + 1
dim_90 = np.argmax(cumulative_variance >= 0.90) + 1

print("\n" + "="*40)
print("【次元数 決定プロセス結果】")
print(f"・情報の 80% を保持するために必要な次元数: {dim_80} 次元 / 最大 {max_components} 次元中")
print(f"・情報の 90% を保持するために必要な次元数: {dim_90} 次元 / 最大 {max_components} 次元中")
print("="*40 + "\n")

# --------------------------------------------------
# 4. プロット（グラフ描画）
# --------------------------------------------------
plt.figure(figsize=(10, 6))

# 累積寄与率の曲線をプロット
plt.plot(
    range(1, len(cumulative_variance) + 1), 
    cumulative_variance, 
    color='blue', 
    linewidth=2, 
    label='Cumulative Explained Variance'
)

# 80% と 90% の目標ライン
plt.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='80% Info Line')
plt.axvline(x=dim_80, color='red', linestyle=':', alpha=0.7)

plt.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='90% Info Line')
plt.axvline(x=dim_90, color='green', linestyle=':', alpha=0.7)

# 交点にマーカーをプロット
plt.scatter([dim_80, dim_90], [0.80, 0.90], color=['red', 'green'], zorder=5)

# グラフの見た目調整
plt.title('PCA Cumulative Explained Variance', fontsize=14, fontweight='bold')
plt.xlabel('Number of PCA Components (PCAの次元数)', fontsize=12)
plt.ylabel('Cumulative Explained Variance (累積寄与率)', fontsize=12)
plt.xlim(0, max_components)
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', fontsize=11)

plt.tight_layout()
plt.show()