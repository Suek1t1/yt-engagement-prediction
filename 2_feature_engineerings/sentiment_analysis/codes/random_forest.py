# 感情分析の特徴量を使用して、ランダムフォレスト回帰モデルを構築し、予測精度を評価するコード
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import japanize_matplotlib

# 共通で使用する英語タイトルのデータセットを読み込み
df_titles = pd.read_csv('datasets/english_titles.csv')

# 1. 感情分析のデータセットを読み込みと結合
# ※実際のファイル名やパスに合わせて適宜変更してください
embedding = pd.read_csv('datasets/sentiment_analysis.csv')

# video_id 列や title 列が重複して混乱しないよう、embedding側から不要な列を落とす
drop_cols = [col for col in ['video_id', 'title'] if col in embedding.columns]
embedding_features = embedding.drop(columns=drop_cols)

# インデックスを揃えて横に連結 (axis=1)
df = pd.concat([df_titles, embedding_features], axis=1)

# 2. フェンス（IQR法）による外れ値の除外
Q1 = df['likes'].quantile(0.25)
Q3 = df['likes'].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

# 閾値の範囲内に収まるデータのみを抽出
df = df[(df['likes'] >= lower_fence) & (df['likes'] <= upper_fence)]

# 3. データ件数の確認
print(f'titles の行数: {len(df_titles)} 件')
print(f'embedding の行数: {len(embedding)} 件')
print(f'結合後のデータ数: {len(df)} 件')
print(f'フェンス適用後の最終データ数: {len(df)} 件\n')

# 4. 特徴量の選択（感情分析の4つの列を指定）
embedding_cols = [col for col in embedding_features.columns]
X = df[embedding_cols]
y = df['likes']

# 5. データをトレーニングセットとテストセットに分割する
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. モデルのトレーニングと予測
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

# 7. 予測値と実際の値をプロットする
plt.figure(figsize=(6, 6))  # グラフが重ならないように新規フィギュアを作成
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('説明変数: 感情分析特徴量 (vader 4指標)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

# プロット保存
plt.savefig('figures/sentiment_analysis_rf.png')
plt.show()  # 表示してリセット
plt.close()