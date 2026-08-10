# チャンネル平均視聴量を特徴量として追加し、ランダムフォレスト回帰モデルを構築するコード
import pandas as pd
from utils import remove_outliers_iqr, random_forest, plot_predictions

# 1. データセットの読み込み
# ※各ファイル名は実際のパスに合わせて変更してください
df_titles = pd.read_csv('datasets/2_feature_engineerings/english_titles.csv')
channel_avg = pd.read_csv('datasets/2_feature_engineerings/channel_avg_views.csv')

# 2. チャンネル名 ('channel_title') をキーにして横方向に結合 (左外部結合)
# 確認用：結合前のデータ数
print(f'english_titles の行数: {len(df_titles)} 件')
print(f'channel_avg_views の行数: {len(channel_avg)} 件')

df = pd.df_titles.merge(channel_avg, on='channel_title', how='left') if hasattr(pd, 'df_titles') else pd.merge(df_titles, channel_avg, on='channel_title', how='left')
df = df.dropna(subset=['avg_views'])

print(f'結合・欠損値処理後のデータ数: {len(df)} 件\n')

# 3. フェンス（IQR法）による外れ値の除外
df = remove_outliers_iqr(df, target_col='likes')

# 4. 特徴量の選択
# チャンネルの平均視聴数 (avg_views) や video_count を特徴量として使用します
# (文字列の channel_title や video_id などの不要な列は除外)
feature_cols = ['avg_views', 'video_count'] # 必要に応じて他の数値列も追加可能
X = df[feature_cols]
y = df['likes']

# 5. データの分割とランダムフォレスト回帰モデルの学習・評価
y_test, y_pred, r2, mse, mape = random_forest(X, y)

# 6. 予測値と実際の値をプロットする
plot_predictions(y_test, y_pred, 
                 title='説明変数＝チャンネル平均視聴量を特徴量', 
                 save_path='2_feature_engineerings/channel_avg_views/figures/channel_avg_rf.png')
