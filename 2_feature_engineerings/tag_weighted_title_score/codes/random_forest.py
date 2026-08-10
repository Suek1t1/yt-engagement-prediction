# タグのレアリティ重み付けスコアを使用して、ランダムフォレスト回帰モデルを構築するコード
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import japanize_matplotlib
from utils import remove_outliers_iqr, random_forest, plot_predictions

# 共通で使用する英語タイトルのデータセットを読み込み
df_titles = pd.read_csv('datasets/2_feature_engineerings/english_titles.csv')

# 1. タグ重み付けスコアのデータセットを読み込み
tag_data = pd.read_csv('datasets/2_feature_engineerings/tag_weighted_title_score.csv')

# データ件数の確認（結合前）
print("=== 結合前のデータ数 ===")
print(f'df_titles の行数: {len(df_titles)} 件')
print(f'tag_data の行数: {len(tag_data)} 件\n')

# 結合時に 'title' 列が重複して 'title_x', 'title_y' になるのを防ぐため、
# tag_data 側から 'title' 列をあらかじめ削除しておく
if 'title' in tag_data.columns:
    tag_data = tag_data.drop(columns=['title'])

# 2. video_id をキーにして結合 (inner merge)
df = pd.merge(df_titles, tag_data, on='video_id', how='inner')

print("=== 結合後のデータ数 ===")
print(f'結合後のデータ数: {len(df)} 件\n')

# 3. フェンス（IQR法）による外れ値の除外
df = remove_outliers_iqr(df, target_col='likes')

# 4. 特徴量の選択（今回は 'score' のみ）
# ※ X は 2次元配列である必要があるため、リストで列名を指定します
X = df[['score']]
y = df['likes']

# 5. データの分割とランダムフォレスト回帰モデルの学習・評価
y_test, y_pred, r2, mse, mape = random_forest(X, y)

# 6. 予測値と実際の値をプロットする
plot_predictions(y_test, y_pred, 
                 title='特徴量=タグのレアリティ重み付けスコア', 
                 save_path='2_feature_engineerings/tag_weighted_title_score/figures/random_forest_predictions.png')