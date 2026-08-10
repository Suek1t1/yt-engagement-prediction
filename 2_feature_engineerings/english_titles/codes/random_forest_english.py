# 英語タイトルのデータを使用して、ランダムフォレスト回帰モデルを構築し、予測精度を評価するコード
import pandas as pd
from utils import remove_outliers_iqr, random_forest, plot_predictions

# 1. データセットの読み込み (先日決めたわかりやすい名前に変更しています)
df = pd.read_csv('datasets/2_feature_engineerings/english_titles.csv')

# 2. フェンス（IQR法）による外れ値の除外（外部ファイル化）
df = remove_outliers_iqr(df, target_col='likes')

# 3. 特徴量と目的変数の設定
X = df[['dislikes', 'views', 'comment_count']]
y = df['likes']

# 4 & 5. モデルのトレーニングと予測（外部ファイル化）
y_test, y_pred = random_forest(X, y)

# 6. 予測値と実際の値をプロットする（外部ファイル化）
plot_predictions(y_test, y_pred,
                 title='説明変数: 視聴数, dislike数, コメント数以外',
                 save_path='2_feature_engineerings/english_titles/figures/EnglishTitles_RandomForest.png')