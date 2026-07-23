# USvideos.csvを使用して、ランダムフォレスト回帰モデルを構築し、予測精度を評価するコード
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
import japanize_matplotlib

# 1. データセットの読み込み
df = pd.read_csv('datasets/USvideos.csv - Sheet1.csv')

# 2. フェンス（IQR法）による外れ値の除外（目的変数は likes）
Q1 = df['likes'].quantile(0.25)
Q3 = df['likes'].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

# 閾値の範囲内に収まるデータのみを抽出
df = df[(df['likes'] >= lower_fence) & (df['likes'] <= upper_fence)]

print(f'フェンス適用後のデータ数: {len(df)} 件\n')

# 3. 特徴量の選択
# 除外する列のリスト
exclude_cols = ['likes', 'dislikes', 'views', 'comment_count', 'video_id']

# データフレームから「数値型の列」かつ「除外リストに含まれない列」を自動抽出
numeric_cols = df.select_dtypes(include=[np.number]).columns
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f'選択された特徴量（{len(feature_cols)}個）: {feature_cols}\n')

X = df[feature_cols]
y = df['likes']

# 4. データをトレーニングセットとテストセットに分割する
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. モデルのトレーニングと予測
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 評価指標の計算
print(f'決定係数 (R^2): {model.score(X_test, y_test)}')
print(f'平均二乗誤差 (MSE): {mean_squared_error(y_test, y_pred)}')

mask = y_test > 0
y_true_clean = y_test[mask]
y_pred_clean = y_pred[mask]
mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
print(f'MAPE: {mape * 100:.2f}%\n')

# 6. 予測値と実際の値をプロットする
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('説明変数: USvideos の数値列')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

# プロット保存
plt.savefig('figures/USvideos_features.png')
plt.show()