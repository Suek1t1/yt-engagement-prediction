# 次元数を変更しながらランダムフォレストの性能を比較し、どの次元数が最適なのか求める
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm  # 進捗バーを表示するためのライブラリ

# 1. データの読み込み
# 圧縮前（768次元）の元のpklファイルを読み込みます
data = pd.read_pickle("sample_1000_data.pkl")

# データの準備
X = np.stack(data['vector'].values)
y = data['likes'].values

# 2. 前処理 (標準化)
# 標準化は次元数に依存しないため、ループの外で1回だけ行えば処理が速くなります
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 結果を保存するリスト
dimensions = list(range(10, 761, 10))  # 10から760まで10刻み
r2_scores = []
mape_scores = []

print("次元数を10から760まで変更しながら検証を開始します...")

# 3. ループ処理 (tqdmで進捗バーを表示)
for n in tqdm(dimensions, desc="学習進捗"):
    # PCAでn次元に圧縮
    pca = PCA(n_components=n)
    X_reduced = pca.fit_transform(X_scaled)
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    
    # ランダムフォレストの構築と学習
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 予測
    predictions = model.predict(X_test)
    
    # 評価 (R^2)
    r2 = model.score(X_test, y_test)
    r2_scores.append(r2)
    
    # 評価 (MAPE - 0除算を回避)
    mask = y_test > 0
    y_true_clean = y_test[mask]
    y_pred_clean = predictions[mask]
    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
    mape_scores.append(mape * 100)

print("すべての評価が完了しました。グラフを描画します。")

# 4. グラフの描画
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# グラフ1: MAPE vs 次元数
ax1.plot(dimensions, mape_scores, marker='o', color='b', markersize=4)
ax1.set_xlabel('次元数 (PCA Components)')
ax1.set_ylabel('MAPE (%)')
ax1.set_title('MAPE vs 次元数')
ax1.grid(True)

# グラフ2: R^2 vs 次元数
ax2.plot(dimensions, r2_scores, marker='o', color='r', markersize=4)
ax2.set_xlabel('次元数 (PCA Components)')
ax2.set_ylabel('決定係数 (R^2)')
ax2.set_title('R^2 vs 次元数')
ax2.grid(True)

plt.tight_layout()
plt.show()