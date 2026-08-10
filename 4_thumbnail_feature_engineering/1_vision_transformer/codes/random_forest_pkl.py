# ランダムフォレストとpklファイルを使用して、likes数を予測するモデルを構築するコード
# ワードトランスフォーマーの埋め込み特徴量を使用して、ランダムフォレスト回帰モデルを構築し、予測精度を評価するコード
import pandas as pd
import numpy as np
from utils import random_forest, plot_predictions

# 1. データセットの読み込みと結合
df = pd.read_csv('datasets/4_thumbnail_features/vision_transformer/sample_1000_data.csv')

# 2. 特徴量と目的変数の設定
X = np.array(df['vector'].tolist(), dtype=float)
y = df['likes'].values
print(f"変換後の X の形状: {X.shape}")

# 3. モデルのトレーニングと予測
y_test, y_pred = random_forest(X, y)

# 4. 予測値と実際の値をプロットして保存
plot_predictions(y_test, y_pred,
              title=f'説明変数= vision-transformer 埋め込み特徴量',
              save_path=f'4_thumbnail_features/vision_transformer/figures/1000_data.png')