# ワードトランスフォーマーの埋め込み特徴量を使用して、ランダムフォレスト回帰モデルを構築し、予測精度を評価するコード
import pandas as pd
from utils import remove_outliers_iqr, random_forest, plot_predictions

# 共通で使用する英語タイトルのデータセットを読み込み
df_titles = pd.read_csv('datasets/2_feature_engineerings/english_titles.csv')

# 処理する次元数のリスト
dimensions = [20, 50]

for dim in dimensions:

  # 1. データセットの読み込みと結合
  embedding = pd.read_csv(f'datasets/3_feature_engineerings/embedding_features_{dim}.csv')

  # video_id 列が重複して混乱しないよう、embedding側の video_id は落としてから結合
  if 'video_id' in embedding.columns:
    embedding_features = embedding.drop(columns=['video_id'])
  else:
    embedding_features = embedding

  # インデックスを揃えて横に連結 (axis=1)
  df = pd.concat([df_titles, embedding_features], axis=1)

  # 2. フェンス（IQR法）による外れ値の除外（外部ファイル化）
  df = remove_outliers_iqr(df, target_col='likes')

  # 3. 特徴量と目的変数の設定
  embedding_cols = [col for col in embedding.columns if col != 'video_id']
  X = df[embedding_cols]
  y = df['likes']

  # 4 & 5. モデルのトレーニングと予測（外部ファイル化）
  y_test, y_pred, r2, mse, mape = random_forest(X, y)

  # 6. 予測値と実際の値をプロットして保存
  plot_predictions(y_test, y_pred,
                  title=f'Embedding ({dim}次元)',
                  save_path=f'1_first_experiments/figures/embedding_{dim}.png')