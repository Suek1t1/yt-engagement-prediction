# データセット数を増やして、ランダムフォレストの性能を比較するコード
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm

# 1. データの読み込み
# ※注意：10,000件分のベクトルデータが入ったファイル名を指定してください
data = pd.read_pickle("sample_10000_data.pkl") 

# --- IQR（四分位範囲）による外れ値の自動除外 ---
Q1 = data['likes'].quantile(0.25)
Q3 = data['likes'].quantile(0.75)
IQR = Q3 - Q1

# 統計学的な外れ値の基準を設定
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 基準内のデータだけを抽出（下限は0未満にならないように調整）
mask = (data['likes'] >= max(100, lower_bound)) & (data['likes'] <= upper_bound)
df_clean = data[mask].copy()

# データの準備
X_all = np.stack(df_clean['vector'].values)
y_all = df_clean['likes'].values

# 検証するデータ数のリスト (1,000件 〜 10,000件)
data_sizes = [1000, 3000, 6000, 10000]

# 次元数は一番結果が良かった「20次元」に固定
n_components = 20

r2_scores = []
mape_scores = []

print(f"PCAの次元数を {n_components} に固定し、データ数を変えながら検証します...")

# 2. ループ処理
for size in tqdm(data_sizes, desc="学習進捗"):
    # そのループのサイズ分だけデータを先頭から切り出す
    X_subset = X_all[:size]
    y_subset = y_all[:size]
    
    # 前処理 (標準化 -> PCA)
    # ※データサイズごとに「重要な軸」が変わる可能性があるため、毎回PCAをかけ直します
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    
    # 訓練データとテストデータに分割 (80%学習、20%テスト)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_subset, test_size=0.2, random_state=42)
    
    # ランダムフォレストの構築と学習
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 予測
    predictions = model.predict(X_test)
    
    # 評価 (R^2)
    r2 = model.score(X_test, y_test)
    r2_scores.append(r2)
    
    # 評価 (MAPE - 0除算を回避)
    mask = y_test >= 100 
    y_true_clean = y_test[mask]
    y_pred_clean = predictions[mask]
    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
    mape_scores.append(mape * 100)
    

    # 結果を表示
    '''
    print("下位1%、3.75%、5%の値:")
    print(df_clean['likes'].quantile([0.01, 0.0375, 0.05])) # 下位1%、3.75%、5%の値を表示
    print("上位95%、97.5%、99%の値:")
    print(df_clean['likes'].quantile([0.95, 0.975, 0.99]))  # 上位95%、97.5%、99%の値を表示
    '''
    print(f"データサイズ {size} の時の y_test 最小値: {y_test.min()}")
    print(f"データサイズ {size} の時の y_test 最大値: {y_test.max()}")
    print(f"データサイズ {size} の時の y_test 平均: {y_test.mean():.2f}")
    print(f"データサイズ {size} の時の y_test 分散: {np.var(y_test):.2f}")
    print(f"データサイズ {size} の時の MAPE: {mape_scores[-1]:.2f}%")
    print(f"データサイズ {size} の時の R^2: {r2:.4f}")
    
    print("-" * 50)

print("すべての評価が完了しました。グラフを描画します。")

# 3. グラフの描画
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
plt.show()