# ユーティリティモジュール
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
import japanize_matplotlib

# --- 外れ値を除外する関数 ---
def remove_outliers_iqr(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1

    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    # 閾値の範囲内に収まるデータのみを抽出
    filtered_df = df[(df[target_col] >= lower_fence) & (df[target_col] <= upper_fence)]
    
    print(f'フェンス適用後のデータ数: {len(filtered_df)} 件 (基準列: {target_col})')
    return filtered_df

# --- ランダムフォレストで学習と評価を行う関数 ---
def random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 評価指標の計算
    print(f'決定係数 (R^2): {model.score(X_test, y_test):.4f}')
    print(f'平均二乗誤差 (MSE): {mean_squared_error(y_test, y_pred):.4f}')

    # MAPEの計算 (ゼロ割り算を防ぐためのマスク処理)
    mask = y_test > 0
    mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask])
    print(f'MAPE: {mape * 100:.2f}%\n')

    return y_test, y_pred

# --- 線形回帰モデルで学習と評価を行う関数 ---
def linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 評価指標の計算
    print(f'決定係数 (R^2): {model.score(X_test, y_test):.4f}')
    print(f'平均二乗誤差 (MSE): {mean_squared_error(y_test, y_pred):.4f}')

    # MAPEの計算 (ゼロ割り算を防ぐためのマスク処理)
    mask = y_test > 0
    mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask])
    print(f'MAPE: {mape * 100:.2f}%\n')

    return y_test, y_pred

# --- 実際の値と予測値をプロットする関数 ---
def plot_predictions(y_test, y_pred, title=None, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('実際の値')
    plt.ylabel('予測値')
    plt.title(title)
    
    # 理想的な予測線の描画 (y=xの直線)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

    # 保存と表示
    if save_path != None:
        plt.savefig(save_path) 
    plt.show()