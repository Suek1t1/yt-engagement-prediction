"""
english_titles.csv を用いた likes 予測システム
構成: ジャンル(category_id)分割 + likes対数変換 + RandomForest

処理の流れ:
  ① データをジャンル(category_id)ごとに分割する
  ② 各ジャンル内で、目的変数 likes を log1p 変換する
  ③ views / dislikes / comment_count から対数likeを RandomForest で予測
  ④ 予測値を expm1 で元のlikeスケールに戻して評価

中間報告書のベースライン(USvideos・全体1モデル・変換なし)との比較:
  ベースライン : R2=0.9600  MSE=1.77e9  MAPE=60.39%
  本システム   : R2=0.9878  MSE=6.31e8  MAPE=27.21%
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 日本語が文字化けする場合に備えてCJKフォントを優先指定
plt.rcParams["font.family"] = [
    "Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# このスクリプトの場所を基準にパスを解決する。
# english_titles.csv が同フォルダになければ親フォルダ(プロジェクト直下)を探す。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_csv(name="english_titles.csv"):
    here = os.path.join(SCRIPT_DIR, name)
    if os.path.exists(here):
        return here
    parent = os.path.join(os.path.dirname(SCRIPT_DIR), name)
    if os.path.exists(parent):
        return parent
    return name  # 最後はカレントディレクトリにフォールバック


CSV = _resolve_csv()
FEATURES = ["views", "dislikes", "comment_count"]
TARGET = "likes"
MIN_SAMPLES = 5          # これ未満のジャンルは学習に使わない
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100


def plot_pred_vs_true(y_true, y_pred, r2, mape_val,
                      path=os.path.join(SCRIPT_DIR, "likes_rf_genre_log_plot.png")):
    """中間報告書と同じ形式の散布図(横軸=実際の値, 縦軸=予測値, 対角線つき)。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=12, alpha=0.5, color="#1f77b4")

    # 理想線 y=x (予測=実際)
    lim = max(y_true.max(), y_pred.max())
    plt.plot([0, lim], [0, lim], "k--", linewidth=1)

    plt.title("説明変数= views, dislikes, comment_count\n"
              "(english_titles / ジャンル分割 + likes対数)")
    plt.xlabel("実際の値")
    plt.ylabel("予測値")
    # 指標を図中に表示
    plt.text(0.05, 0.95,
             f"$R^2$ = {r2:.4f}\nMAPE = {mape_val:.2f}%",
             transform=plt.gca().transAxes,
             va="top", ha="left",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> {path} を保存しました")


def mape(y_true, y_pred):
    """平均絶対パーセント誤差(%)。likes=0 の行は除外。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    df = pd.read_csv(CSV)
    for c in FEATURES + [TARGET]:
        df[c] = df[c].astype(float)

    all_true, all_pred = [], []
    per_genre = []

    # ① ジャンルごとに分割
    for cat, g in df.groupby("category_id"):
        if len(g) < MIN_SAMPLES:
            continue

        X = g[FEATURES]
        y = g[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # ② likes を対数変換して学習し、③ RandomForest で予測
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, np.log1p(y_train))

        # ④ expm1 で元スケールへ戻す
        pred = np.expm1(model.predict(X_test))

        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())

        per_genre.append(
            {
                "category_id": int(cat),
                "n": len(g),
                "R2": r2_score(y_test, pred),
                "MAPE": mape(y_test, pred),
            }
        )

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # ---- 全体評価 ----
    print("=== 全体評価 (english_titles / ジャンル分割 + likes対数) ===")
    print(f"R^2  : {r2_score(all_true, all_pred):.4f}")
    print(f"MSE  : {mean_squared_error(all_true, all_pred):.4e}")
    print(f"RMSE : {np.sqrt(mean_squared_error(all_true, all_pred)):,.0f}")
    print(f"MAPE : {mape(all_true, all_pred):.2f}%")

    # ---- ジャンル別評価 ----
    per_genre_df = pd.DataFrame(per_genre).sort_values("category_id")
    print("\n=== ジャンル別評価 ===")
    print(per_genre_df.to_string(index=False))

    out_csv = os.path.join(SCRIPT_DIR, "likes_rf_genre_log_per_genre.csv")
    per_genre_df.to_csv(out_csv, index=False)
    print(f"\n-> {out_csv} を保存しました")

    # ---- 中間報告書形式の散布図 ----
    plot_pred_vs_true(
        all_true,
        all_pred,
        r2=r2_score(all_true, all_pred),
        mape_val=mape(all_true, all_pred),
    )


if __name__ == "__main__":
    main()
