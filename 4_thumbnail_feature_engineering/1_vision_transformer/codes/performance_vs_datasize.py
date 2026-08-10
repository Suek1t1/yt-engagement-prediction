# データセット数を増やして、ランダムフォレストの性能を比較し、学習曲線を描画するコード
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import remove_outliers_iqr, random_forest, plot_predictions

# 1. データの読み込み
# ※注意：10,000件分のベクトルデータが入ったファイル名を指定してください
data = pd.read_pickle("datasets/4_thumbnail_feature_engineering/1_vision_transformer/sample_10000_data_pca.pkl") 

# 2. フェンス（IQR法）による外れ値の除外
df = remove_outliers_iqr(data, target_col='likes')


# 3. データの準備
X_all = np.stack(df.drop(columns=['likes']).values)
y_all = df['likes'].values

r2_scores = []
mape_scores = []

n_components = len(df.drop(columns=['likes']).columns)  # 元の次元数を取得
print(f"PCAの次元数を {n_components} に固定し、データ数を変えながら検証します...")

data_sizes = list(range(1000, 7700, 1000))  # 1000から最大まで1000データサイズを指定

# 4. 学習のループ処理
for size in tqdm(data_sizes, desc="学習進捗"):
    # そのループのサイズ分だけデータを先頭から切り出す
    X_subset = X_all[:size]
    y_subset = y_all[:size]

    # ランダムフォレストで学習と予測
    y_test, y_pred, r2, mse, mape = random_forest(X_subset, y_subset)
    
    # 評価指標をリストに追加
    r2_scores.append(r2)
    mape_scores.append(mape)

    print("-" * 50)

print("すべての評価が完了しました。グラフを描画します。")

# 4. グラフの描画
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# グラフ1: MAPE vs データ数
ax1.plot(data_sizes, mape_scores, marker='o', color='b', markersize=6, linewidth=2)
ax1.set_xlabel('データ数 (Data Size)')
ax1.set_ylabel('MAPE (%)')
ax1.set_title(f'MAPE vs データ数 (PCA={n_components}次元)')
ax1.grid(True)

# グラフ2: R^2 vs データ数
ax2.plot(data_sizes, r2_scores, marker='o', color='r', markersize=6, linewidth=2)
ax2.set_xlabel('データ数 (Data Size)')
ax2.set_ylabel('決定係数 (R^2)')
ax2.set_title(f'R^2 vs データ数 (PCA={n_components}次元)')
ax2.grid(True)

plt.tight_layout()

# 4. プロットの保存
plt.savefig("4_thumbnail_feature_engineering/1_vision_transformer/figures/performance_vs_datasize_10000.png")
plt.show()
