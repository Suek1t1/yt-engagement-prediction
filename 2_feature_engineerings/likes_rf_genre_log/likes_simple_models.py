"""
シンプルなモデルへの引き算 — 「効く要素だけ残す」検証
================================================================================
狙い:
  分散分解（第12回）で チャンネル31.5% / 時代11.9% / 内容6.8% と判明。内容（TF-IDF・DL）を
  削っても説明力の約8割は残るはず。どこまでシンプルにして精度が保てるかを、現行モデルと
  同一の正規プロトコル（重複除去・時系列 rolling-origin・D4ターゲット4段階）で測る。

比較する4モデル（おすすめ順）:
  L1_formula : 学習なし。予測段階 = 「縮小チャンネル平均 ÷ 学習トレンド水準」の相対比を
               学習分位点で4段階に切るだけ。特徴量ゼロ・TF-IDFもRFも不要。
  L2_rf6     : 特徴6個だけのRF（chan_shrunk_log / category_id / publish_dow /
               publish_hour / title_len / tag_count）。TF-IDF 300次元を捨てる。
  L3_logit2  : 4段階をやめ「中央値超えか」の二値をロジスティック回帰（特徴はL2と同じ6個）。
               係数がそのまま各要素の効き。二値なのでAUCで評価。
  現行_full  : 参照値。likes_rolling_eval_folds.csv の D4_naive（TF-IDF+全特徴RF・4段階）。

評価: 4段階は acc/±1段階/macroF1/低F1、二値は AUC。ρ（順位）も併記。

実行: python3 likes_simple_models.py   （L1/L2/L3は軽く、全fold前景で完走）
出力: likes_simple_folds.csv / likes_simple_summary.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_fscore_support)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from likes_4class_v3 import (load_dedupe, build_base, weekly_denominators,
                             N_CLASSES, TARGET, RANDOM_STATE)
from likes_rolling_eval import get_folds

SHRINK_K = 1.0
SIMPLE_FEATURES = ["channel_mean_likes_log", "category_id", "publish_dow",
                   "publish_hour", "title_len", "tag_count"]


def shrunk_channel_mean(df, tr, idx):
    tr_df = df.iloc[tr]
    ll = np.log1p(tr_df[TARGET])
    cnt = tr_df.groupby("channel_title")[TARGET].count()
    chm = ll.groupby(tr_df["channel_title"].values).mean()
    catm = ll.groupby(tr_df["category_id"].values).mean()
    gmean = ll.mean()
    sub = df.iloc[idx]
    n = sub["channel_title"].map(cnt).fillna(0).values
    c = sub["channel_title"].map(chm).fillna(0).values
    k = sub["category_id"].map(catm).fillna(gmean).values
    return (n * c + SHRINK_K * k) / (n + SHRINK_K)


def cls_metrics(y, pred):
    p, r, f, s = precision_recall_fscore_support(
        y, pred, labels=list(range(N_CLASSES)), zero_division=0)
    return {"accuracy": round(accuracy_score(y, pred), 3),
            "adj1": round(float(np.mean(np.abs(pred - y) <= 1)), 3),
            "macroF1": round(float(f1_score(y, pred, average="macro")), 3),
            "低F1": round(float(f[0]), 3)}


def run_fold(fid, df, base, y, week_idx, ts, te_end):
    tr = np.where(week_idx < ts)[0]
    te = np.where((week_idx >= ts) & (week_idx < te_end))[0]
    (_, _, _, denom_d4, week, med_by_week, tr_med,
     coef, coef3, last_level) = weekly_denominators(df, y, tr)
    denom = np.clip(np.where(np.arange(len(df)) < len(tr),
                             denom_d4, last_level), 1.0, None)
    ratio = y / denom
    q = np.linspace(0, 1, N_CLASSES + 1)[1:-1]
    edges = np.quantile(ratio[tr], q)
    yb = np.digitize(ratio, edges)

    # 特徴行列（chan は縮小平均に差し替え）
    def feat(idx):
        X = base.iloc[idx].copy()
        X["channel_mean_likes_log"] = shrunk_channel_mean(df, tr, idx)
        return X[SIMPLE_FEATURES].values.astype(float)

    Xtr, Xte = feat(tr), feat(te)
    rows = []

    # --- L1: 学習なし。縮小chan平均÷トレンド水準 を段階化 ---
    chan_ratio_te = np.expm1(shrunk_channel_mean(df, tr, te)) / denom[te]
    pred_l1 = np.digitize(chan_ratio_te, edges)
    rows.append({"fold": fid, "model": "L1_formula", **cls_metrics(yb[te], pred_l1),
                 "AUC": None,
                 "rho": round(float(spearmanr(chan_ratio_te, ratio[te]).correlation), 3)})

    # --- L2: 特徴6個のRF（4段階） ---
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                n_jobs=-1, class_weight="balanced_subsample")
    rf.fit(Xtr, yb[tr])
    proba = np.zeros((len(te), N_CLASSES))
    for j, c in enumerate(rf.classes_):
        proba[:, int(c)] = rf.predict_proba(Xte)[:, j]
    pred_l2 = proba.argmax(1)
    esc = proba @ np.arange(N_CLASSES)
    rows.append({"fold": fid, "model": "L2_rf6", **cls_metrics(yb[te], pred_l2),
                 "AUC": None, "rho": round(float(spearmanr(esc, ratio[te]).correlation), 3)})

    # --- L3: 二値ロジスティック回帰（中央値超え） ---
    yb2_tr = (ratio[tr] > np.median(ratio[tr])).astype(int)
    yb2_te = (ratio[te] > np.median(ratio[tr])).astype(int)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(sc.transform(Xtr), yb2_tr)
    prob2 = lr.predict_proba(sc.transform(Xte))[:, 1]
    auc = roc_auc_score(yb2_te, prob2) if len(np.unique(yb2_te)) > 1 else np.nan
    rows.append({"fold": fid, "model": "L3_logit2",
                 "accuracy": round(accuracy_score(yb2_te, (prob2 > 0.5).astype(int)), 3),
                 "adj1": None, "macroF1": None, "低F1": None,
                 "AUC": round(float(auc), 3),
                 "rho": round(float(spearmanr(prob2, ratio[te]).correlation), 3)})
    return rows


def main():
    df = load_dedupe()
    y = df[TARGET].values.astype(float)
    base = build_base(df)
    week = df["trend_dt"].dt.to_period("W").dt.start_time
    week_idx = ((week - week.min()).dt.days // 7).values
    folds = get_folds(int(week_idx.max()) + 1)

    rows = []
    for fid, (ts, te_end) in enumerate(folds):
        rows += run_fold(fid, df, base, y, week_idx, ts, te_end)
        print(f"fold{fid} done")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(SCRIPT_DIR, "likes_simple_folds.csv"), index=False)

    sub = res[res["fold"] < 5]
    summ = sub.groupby("model")[["accuracy", "adj1", "macroF1", "低F1", "AUC", "rho"]]\
              .mean().round(3)
    # 現行fullの参照値
    ref_p = os.path.join(SCRIPT_DIR, "likes_rolling_eval_folds.csv")
    if os.path.exists(ref_p):
        ref = pd.read_csv(ref_p)
        ref = ref[(ref["method"] == "D4_naive") & (ref["fold"] < 5)]
        print("[参照] 現行_full (rolling D4_naive, TF-IDF+全特徴RF): "
              "acc %.3f / adj1 %.3f / 低F1 %.3f"
              % (ref["accuracy"].mean(), ref["adj1"].mean(), ref["低_f1"].mean()))
    summ.to_csv(os.path.join(SCRIPT_DIR, "likes_simple_summary.csv"))
    print("fold0-4平均:")
    print(summ.to_string())


if __name__ == "__main__":
    main()
