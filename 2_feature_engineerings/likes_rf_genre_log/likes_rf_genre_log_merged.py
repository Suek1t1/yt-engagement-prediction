"""
english_titles.csv を用いた likes 予測システム（小カテゴリー統合版）
構成: ジャンル(category_id)分割 + likes対数変換 + RandomForest
      ※ サンプル数が少ないカテゴリー(n < 500)を「その他(-1)」に統合する。

改善点（元の likes_rf_genre_log.py との差分）:
  - n < MERGE_THRESHOLD(=500) のカテゴリーを 1 つの「その他」グループ(-1)に統合。
    元コードでは category 29(n=56) や 43(n=57) のような極小カテゴリーが
    単独学習されており、test 件数が十数件しかないため R^2 が 0.9999 のような
    非現実的な値になり、split の引き次第で大きくぶれていた（splitガチャ）。
    小カテゴリーをまとめることで test 件数が確保され、評価が安定する。
  - per_genre 出力に、その行が統合グループか単独カテゴリーかを示す
    'merged' 列と、統合された元 category_id を示す 'members' 列を追加。

処理の流れ:
  ① n < 500 のカテゴリーを category_id = -1（その他）に置き換える
  ② データを（統合後の）ジャンルごとに分割する
  ③ 各ジャンル内で目的変数 likes を log1p 変換して RandomForest を学習
  ④ 予測値を expm1 で元の likes スケールに戻して評価
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
MIN_SAMPLES = 5          # これ未満のグループは学習に使わない
MERGE_THRESHOLD = 500    # n がこの値未満のカテゴリーは「その他(-1)」に統合する
OTHER_ID = -1            # 統合先のカテゴリーID
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100


def merge_small_categories(df, threshold=MERGE_THRESHOLD, other_id=OTHER_ID):
    """サンプル数が threshold 未満のカテゴリーを other_id にまとめた
    新しい列 'cat_group' を付与して返す。統合の対応関係も返す。"""
    counts = df["category_id"].value_counts()
    small = set(counts[counts < threshold].index)
    df = df.copy()
    df["cat_group"] = df["category_id"].apply(
        lambda c: other_id if c in small else c
    )
    return df, sorted(small)


def plot_pred_vs_true(y_true, y_pred, r2, mape_val,
                      path=os.path.join(SCRIPT_DIR,
                                        "likes_rf_genre_log_merged_plot.png")):
    """中間報告書と同じ形式の散布図(横軸=実際の値, 縦軸=予測値, 対角線つき)。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=12, alpha=0.5, color="#1f77b4")

    lim = max(y_true.max(), y_pred.max())
    plt.plot([0, lim], [0, lim], "k--", linewidth=1)

    plt.title("説明変数= views, dislikes, comment_count\n"
              "(english_titles / ジャンル分割[小カテゴリー統合] + likes対数)")
    plt.xlabel("実際の値")
    plt.ylabel("予測値")
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

    # ① 小カテゴリーを「その他(-1)」に統合
    df, merged_ids = merge_small_categories(df)
    if merged_ids:
        print(f"統合対象カテゴリー (n < {MERGE_THRESHOLD}): "
              f"{merged_ids} -> その他({OTHER_ID})")
    else:
        print(f"n < {MERGE_THRESHOLD} のカテゴリーはありませんでした")

    all_true, all_pred = [], []
    per_genre = []

    # ② 統合後のグループごとに分割
    for cat, g in df.groupby("cat_group"):
        if len(g) < MIN_SAMPLES:
            continue

        X = g[FEATURES]
        y = g[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # ③ likes を対数変換して学習
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

        is_merged = int(cat) == OTHER_ID
        per_genre.append(
            {
                "cat_group": int(cat),
                "n": len(g),
                "n_test": len(X_test),
                "merged": is_merged,
                "members": ",".join(map(str, merged_ids)) if is_merged else str(int(cat)),
                "R2": r2_score(y_test, pred),
                "MAPE": mape(y_test, pred),
            }
        )

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # ---- 全体評価 ----
    print("\n=== 全体評価 (english_titles / 小カテゴリー統合 + likes対数) ===")
    print(f"R^2  : {r2_score(all_true, all_pred):.4f}")
    print(f"MSE  : {mean_squared_error(all_true, all_pred):.4e}")
    print(f"RMSE : {np.sqrt(mean_squared_error(all_true, all_pred)):,.0f}")
    print(f"MAPE : {mape(all_true, all_pred):.2f}%")

    # ---- グループ別評価 ----
    per_genre_df = pd.DataFrame(per_genre).sort_values("cat_group")
    print("\n=== グループ別評価 ===")
    print(per_genre_df.to_string(index=False))

    out_csv = os.path.join(SCRIPT_DIR,
                           "likes_rf_genre_log_merged_per_genre.csv")
    per_genre_df.to_csv(out_csv, index=False)
    print(f"\n-> {out_csv} を保存しました")

    # ---- 散布図 ----
    plot_pred_vs_true(
        all_true,
        all_pred,
        r2=r2_score(all_true, all_pred),
        mape_val=mape(all_true, all_pred),
    )


if __name__ == "__main__":
    main()
