# ワードトランスフォーマーの埋め込み特徴量を使用して、ランダムフォレスト回帰モデルを構築し、予測精度を評価するコード
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import japanize_matplotlib

# 共通で使用する英語タイトルのデータセットを読み込み
df_titles = pd.read_csv('datasets/english_titles.csv')

# 処理する次元数のリスト
dimensions = [20, 50]

for dim in dimensions:
  print(f'=== {dim}次元の処理を開始します ===')

  # 1. 各次元のデータセットを読み込みと結合
  embedding_file = f'datasets/embedding_features_{dim}.csv'
  embedding = pd.read_csv(embedding_file)

  # video_id 列が重複して混乱しないよう、embedding側の video_id は落としてから結合
  if 'video_id' in embedding.columns:
    embedding_features = embedding.drop(columns=['video_id'])
  else:
    embedding_features = embedding

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

  # 4. 特徴量の選択
  embedding_cols = [col for col in embedding.columns if col != 'video_id']
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
  plt.title(f'説明変数: embedding ({dim}次元)')
  plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

  # プロット保存 (ファイル名を動的に変更)
  plt.savefig(f'figures/embedding_{dim}.png')
  plt.show()  # 表示してリセット
  plt.close()