# 感情分析の特徴量を使用して、ランダムフォレスト回帰モデルを構築し、予測精度を評価するコード
import pandas as pd
from utils import remove_outliers_iqr, plot_predictions, random_forest

# 共通で使用する英語タイトルのデータセットを読み込み
df_titles = pd.read_csv('datasets/2_feature_engineerings/english_titles.csv')

# 1. 感情分析のデータセットを読み込みと結合
# ※実際のファイル名やパスに合わせて適宜変更してください
embedding = pd.read_csv('datasets/2_feature_engineerings/sentiment_analysis.csv')

# video_id 列や title 列が重複して混乱しないよう、embedding側から不要な列を落とす
drop_cols = [col for col in ['video_id', 'title'] if col in embedding.columns]
embedding_features = embedding.drop(columns=drop_cols)

# インデックスを揃えて横に連結 (axis=1)
df = pd.concat([df_titles, embedding_features], axis=1)

# 2. フェンス（IQR法）による外れ値の除外
df = remove_outliers_iqr(df, target_col='likes')

# 3. 特徴量の選択（感情分析の4つの列を指定）
embedding_cols = [col for col in embedding_features.columns]
X = df[embedding_cols]
y = df['likes']

# 4. データの分割とランダムフォレスト回帰モデルの学習・評価
y_test, y_pred, r2, mse, mape = random_forest(X, y)

# 7. 実際の値と予測値の散布図をプロット
plot_predictions(y_test, y_pred, 
                 title='特徴量＝感情分析', 
                 save_path='2_feature_engineerings/sentiment_analysis/results/random_forest_predictions.png')