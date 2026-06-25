"""
english_titles.csv を用いた likes 予測システム（投稿時点特徴量・リークなし正式版）
================================================================================
目的:
  「投稿された瞬間に分かる情報」だけから likes を予測する、リークのない予測モデル。
  従来モデル(likes_rf_genre_log.py)は views/dislikes/comment_count という
  likes と同時に決まる量を入力にしており、予測ではなく逆算になっていた。
  本モデルはそれらを一切使わない。

リーク対策のポイント:
  - 説明変数に views / dislikes / comment_count を含めない。
  - channel_mean_likes（チャンネルの過去平均likes）は強力な特徴だが、
    全データから算出すると test の情報が train に漏れる。
    → 各 fold で「学習データのみ」から算出し、test には map するだけにする
       (target encoding の正しいやり方)。未知チャンネルは train 全体平均で補完。

特徴量(投稿時点で分かるもの):
  title_len, title_word_count, title_upper_ratio, title_has_excl,
  title_has_question, tag_count, publish_hour, publish_dow,
  comments_disabled, ratings_disabled, category_id,
  channel_mean_likes_log (fold内学習データのみで算出)

評価:
  5-fold CV の out-of-fold 予測を集約し、元スケール/対数スケール両方で R2 を報告。
  対数スケール R2 は fold ごとの平均±標準偏差も出す(安定性の確認)。
  併せて RMSE・MAE・MAPE を報告。中間報告書形式の散布図も出力。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_csv(name="english_titles.csv"):
    for p in (os.path.join(SCRIPT_DIR, name),
              os.path.join(os.path.dirname(SCRIPT_DIR), name)):
        if os.path.exists(p):
            return p
    return name


CSV = _resolve_csv()
TARGET = "likes"
N_SPLITS = 5
RANDOM_STATE = 42
N_ESTIMATORS = 200

# 投稿時点で分かる素性（チャンネル特徴は fold 内で別途付与する）
BASE_FEATURES = [
    "title_len", "title_word_count", "title_upper_ratio",
    "title_has_excl", "title_has_question", "tag_count",
    "publish_hour", "publish_dow",
    "comments_disabled", "ratings_disabled", "category_id",
]
CHANNEL_FEATURE = "channel_mean_likes_log"


def build_base_features(df):
    """チャンネル特徴を除く、投稿時点の素性を構築する。"""
    out = pd.DataFrame(index=df.index)
    title = df["title"].fillna("").astype(str)
    out["title_len"] = title.str.len()
    out["title_word_count"] = title.str.split().apply(len)
    out["title_upper_ratio"] = title.apply(
        lambda s: sum(c.isupper() for c in s) / len(s) if len(s) else 0.0
    )
    out["title_has_excl"] = title.str.contains("!").astype(int)
    out["title_has_question"] = title.str.contains(r"\?").astype(int)

    tags = df["tags"].fillna("[none]").astype(str)
    out["tag_count"] = np.where(tags == "[none]", 0, tags.str.count(r"\|") + 1)

    pt = pd.to_datetime(df["publish_time"], errors="coerce", utc=True)
    out["publish_hour"] = pt.dt.hour.fillna(-1).astype(int)
    out["publish_dow"] = pt.dt.dayofweek.fillna(-1).astype(int)

    out["comments_disabled"] = df["comments_disabled"].astype(int)
    out["ratings_disabled"] = df["ratings_disabled"].astype(int)
    out["category_id"] = df["category_id"].astype(int)
    return out


def add_channel_feature(base, df_rows, channel_means, global_mean):
    """学習データから算出した channel_means を map して付与（test にも train にも）。
    未知チャンネルは global_mean（train 全体平均）で補完。"""
    X = base.copy()
    mapped = df_rows["channel_title"].map(channel_means).fillna(global_mean)
    X[CHANNEL_FEATURE] = np.log1p(mapped.values)
    return X


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = y_true != 0
    return np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100


def plot_pred_vs_true(y_true, y_pred, r2, mape_val,
                      path=os.path.join(SCRIPT_DIR, "likes_pretime_rf_plot.png")):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=12, alpha=0.4, color="#1f77b4")
    lim = max(y_true.max(), y_pred.max())
    plt.plot([0, lim], [0, lim], "k--", linewidth=1)
    plt.title("投稿時点特徴量のみ（リークなし）\n"
              "title長/タグ数/投稿時刻/カテゴリ/チャンネル過去平均")
    plt.xlabel("実際の値")
    plt.ylabel("予測値")
    plt.text(0.05, 0.95, f"$R^2$ = {r2:.4f}\nMAPE = {mape_val:.2f}%",
             transform=plt.gca().transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> {path} を保存しました")


def main():
    df = pd.read_csv(CSV).dropna(subset=[TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(float)
    y = df[TARGET].values

    base = build_base_features(df)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(y))
    r2_log_folds = []
    importances_acc = None

    for fold, (tr, te) in enumerate(kf.split(df), start=1):
        # --- fold 内学習データのみから channel 平均を算出（リーク防止）---
        ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
        global_mean = df.iloc[tr][TARGET].mean()

        X_tr = add_channel_feature(base.iloc[tr], df.iloc[tr], ch_means, global_mean)
        X_te = add_channel_feature(base.iloc[te], df.iloc[te], ch_means, global_mean)

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(X_tr, np.log1p(y[tr]))

        pred_log = model.predict(X_te)
        oof_pred[te] = np.expm1(pred_log)
        r2_log_folds.append(r2_score(np.log1p(y[te]), pred_log))

        imp = model.feature_importances_
        importances_acc = imp if importances_acc is None else importances_acc + imp
        print(f"  fold {fold}: R2(log)={r2_log_folds[-1]:.4f}")

    # ---- 集約評価 ----
    r2_real = r2_score(y, oof_pred)
    r2_log_mean = float(np.mean(r2_log_folds))
    r2_log_std = float(np.std(r2_log_folds))
    rmse = float(np.sqrt(mean_squared_error(y, oof_pred)))
    mae = float(mean_absolute_error(y, oof_pred))
    mape_val = mape(y, oof_pred)

    print("\n" + "=" * 64)
    print("=== 投稿時点特徴量モデル（リークなし・5-fold CV）===")
    print("=" * 64)
    print(f"R2 (実スケール)   : {r2_real:.4f}")
    print(f"R2 (対数スケール) : {r2_log_mean:.4f} ± {r2_log_std:.4f}")
    print(f"RMSE             : {rmse:,.0f}")
    print(f"MAE              : {mae:,.0f}")
    print(f"MAPE             : {mape_val:.2f}%")

    # ---- 特徴量重要度（fold 平均）----
    feat_names = BASE_FEATURES + [CHANNEL_FEATURE]
    imp_avg = importances_acc / N_SPLITS
    imp_df = (pd.DataFrame({"feature": feat_names, "importance": imp_avg})
              .sort_values("importance", ascending=False).reset_index(drop=True))
    print("\n=== 特徴量重要度（fold平均）===")
    print(imp_df.to_string(index=False))

    imp_df.to_csv(os.path.join(SCRIPT_DIR, "likes_pretime_rf_importance.csv"),
                  index=False)

    summary = pd.DataFrame([{
        "model": "pretime_rf (no leak)",
        "R2_real": r2_real, "R2_log": r2_log_mean, "R2_log_std": r2_log_std,
        "RMSE": rmse, "MAE": mae, "MAPE": mape_val,
    }])
    summary.to_csv(os.path.join(SCRIPT_DIR, "likes_pretime_rf_summary.csv"),
                   index=False)
    print("\n-> likes_pretime_rf_importance.csv / _summary.csv を保存しました")

    plot_pred_vs_true(y, oof_pred, r2_real, mape_val)


if __name__ == "__main__":
    main()
