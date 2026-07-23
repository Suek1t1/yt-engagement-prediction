# ランダムフォレストとpklファイルを使用して、likes数を予測するモデルを構築するコード
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


# データの読み込み
df = pd.read_pickle("sample_10000_data.pkl")
print("--- 列名 ---")
print(df.columns)
print("\n---データ数 ---")
print(len(df))

# 2. フェンス（IQR法）による外れ値の除外
Q1 = df['likes'].quantile(0.25)
Q3 = df['likes'].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

# 閾値の範囲内に収まるデータのみを抽出（フィルタリング）
df = df[(df['likes'] >= lower_fence) & (df['likes'] <= upper_fence)]
print(f"フェンス適用後の最終データ数: {len(df)} 件\n")

# データの準備
X = df.drop('likes', axis=1).values
y = df['likes'].values
# 訓練データとテストデータに分割 (80%学習、20%テスト)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストの構築
# likes数は連続値なので Regressor を使用します
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

print("学習開始...")
model.fit(X_train, y_train)

# 予測と評価
predictions = model.predict(X_test)
print(f"決定係数 (R^2): {model.score(X_test, y_test)}")
print(f"平均二乗誤差 (MSE): {mean_squared_error(y_test, predictions)}")

# 平均絶対百分率誤差 (MAPE) を計算して表示する
mask = y_test > 0  # 0より大きいデータだけ残す
y_true_clean = y_test[mask]
y_pred_clean = predictions[mask]
mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
print(f"MAPE: {mape * 100:.2f}%")

# 予測値と実際の値をプロットする
plt.scatter(y_test, predictions)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('説明変数= one-hotエンコードされたタグ')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()