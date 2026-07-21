"""
方向性3: Conformal Prediction による最低保証の理論武装
================================================================================
背景:
  第12回の方向性7で、GBR分位点回帰(10%分位点)の実測カバレッジは平均77.4%と
  目標90%を大きく下回った（分布シフト下では分位点lossの較正が効かない）。
  そこで split conformal prediction により、分布仮定なしの有限サンプル保証を持つ
  「最低保証段階 predict_floor 互換」を構築する。

方法（改善ロードマップ.md 方向性3 準拠 + rolling-origin基盤で fold別報告）:
  - 各 rolling fold の学習窓のうち、末尾（較正窓）を較正セットに割き、
    その手前だけでRFを学習する（モデルは較正ラベルを見ない）。
  - 非適合度スコア: s_i = P̂(y ≤ y_i | x_i)（累積予測確率。ordinal floor と自然に対応）。
    下限段階 floor(x) = max{c : P̂(y ≤ c-1 | x) ≤ τ}。
    保証違反 (floor > y) ⇔ s ≤ τ なので、τ を較正スコアの
    ⌊α(n+1)⌋ 番目の順序統計量にとれば P(違反) ≤ α が理論保証される。
  - 比較1: conformal較正 vs 無較正（τ=α、モデル確率をそのまま信じる）
    → 方向性7の「較正されていない下限は崩れる」の再現と対策の実証。
  - 比較2: 較正窓 全20% vs 直近4週 vs 直近2週（分布シフト対策の適応的較正窓）。
    ※ロードマップ原案は「全 vs 8週 vs 4週」だが、rolling foldの学習窓が最短10週で
      8週較正は学習データが残らないため、4週/2週に置き換えた。
  - ターゲット: v1（絶対log-likes分位）と D4（時間相対比・最終水準持ち越し）の両方。
  - カバレッジ目標: 90% / 95%。

sandbox 45秒対策: fold毎に分割実行→assemble。
  python3 likes_conformal_floor.py <fold_id>
  python3 likes_conformal_floor.py assemble

出力:
  likes_conformal_folds.csv    fold×target×較正法×水準 の実測カバレッジ等
  likes_conformal_summary.csv  較正法別の平均カバレッジ・情報量
  likes_conformal_plot.png     fold別カバレッジ（目標線つき）
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from likes_4class_v3 import (load_dedupe, build_base, text_corpus,
                             weekly_denominators, NUM_FEATURES, N_CLASSES,
                             CLASS_NAMES, TARGET, RANDOM_STATE, TFIDF_MAX)
from likes_rolling_eval import get_folds

PART_DIR = os.path.join(SCRIPT_DIR, "_conformal_parts")
N_ESTIMATORS = 300
ALPHAS = [0.10, 0.05]                 # 目標カバレッジ 90% / 95%
TARGETS = ["v1", "D4"]
CAL_FRAC = 0.2                        # 学習窓の末尾20%(週)を較正に
WINDOWS = ["all", "recent4", "recent2"]   # 較正窓: 全較正期間 / 直近4週 / 直近2週
METHODS = ["conformal", "uncalibrated"]


def make_bins(vals, tr_idx):
    q = np.linspace(0, 1, N_CLASSES + 1)[1:-1]
    edges = np.quantile(vals[tr_idx], q)
    return np.digitize(vals, edges), edges


def floor_from_proba(proba, tau):
    """floor(x) = max{c: cumP(y<=c-1) <= tau}（c=0は常に許可）"""
    cum_below = np.cumsum(proba, axis=1)          # P(y<=c)
    F = np.hstack([np.zeros((len(proba), 1)), cum_below[:, :-1]])  # P(y<=c-1)
    ok = F <= tau
    ok[:, 0] = True
    return (ok * np.arange(N_CLASSES)[None, :]).max(axis=1) if False else \
        np.array([np.max(np.where(row)[0]) for row in ok])


def scores_from_proba(proba, y_true):
    """s_i = P̂(y <= y_i | x_i)"""
    cum = np.cumsum(proba, axis=1)
    return cum[np.arange(len(y_true)), y_true]


def conformal_tau(scores, alpha):
    """⌊α(n+1)⌋番目の順序統計量。k<1なら-inf（floor常に0=無情報だが保証は成立）"""
    n = len(scores)
    k = int(np.floor(alpha * (n + 1)))
    if k < 1:
        return -np.inf
    return np.sort(scores)[k - 1]


def evaluate_floor(proba_te, yb_te, tau):
    floor = floor_from_proba(proba_te, tau)
    cover = float(np.mean(yb_te >= floor))
    return {"coverage": round(cover, 3),
            "mean_floor": round(float(floor.mean()), 3),
            "informative_rate": round(float(np.mean(floor >= 1)), 3)}


def run_fold(fold_id, df, base, corpus, y, week_idx, train_end_week, test_end_week):
    tr_all = np.where(week_idx < train_end_week)[0]
    te = np.where((week_idx >= train_end_week) & (week_idx < test_end_week))[0]

    # 較正窓: 学習窓の末尾 max(2, ceil(20%)) 週
    n_cal_weeks = max(2, int(np.ceil(train_end_week * CAL_FRAC)))
    cal_start_week = train_end_week - n_cal_weeks
    tr = tr_all[week_idx[tr_all] < cal_start_week]          # 真の学習
    cal = tr_all[week_idx[tr_all] >= cal_start_week]        # 較正

    # 分母・ラベルは「真の学習」期間のみから（較正ラベルもモデル・境界には使わない）
    (denom_d1, denom_d2, denom_d3, denom_d4, week, med_by_week, tr_med,
     coef, coef3, last_level) = weekly_denominators(df, y, tr)
    # 較正・test週の分母は最終水準持ち越し（tr以降はすべて未来扱い）
    denom_d4 = np.where(np.arange(len(df)) < tr[-1] + 1, denom_d4, last_level)

    labels = {}
    labels["v1"], _ = make_bins(np.log1p(y), tr)
    labels["D4"], _ = make_bins(y / np.clip(denom_d4, 1.0, None), tr)

    # 特徴量（チャンネル平均は真の学習期間のみ）
    ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()

    def features(idx):
        X = base.iloc[idx].copy()
        m = df.iloc[idx]["channel_title"].map(ch_means).fillna(gmean)
        X["channel_mean_likes_log"] = np.log1p(m.values)
        return X[NUM_FEATURES]

    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr])
    Xt_cal = vec.transform(corpus[cal])
    Xt_te = vec.transform(corpus[te])
    X_tr = hstack([csr_matrix(features(tr).values.astype(float)), Xt_tr]).tocsr()
    X_cal = hstack([csr_matrix(features(cal).values.astype(float)), Xt_cal]).tocsr()
    X_te = hstack([csr_matrix(features(te).values.astype(float)), Xt_te]).tocsr()

    rows = []
    for tgt in TARGETS:
        yb = labels[tgt]
        yb_tr, yb_cal, yb_te = yb[tr], yb[cal], yb[te]
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                     random_state=RANDOM_STATE, n_jobs=-1,
                                     class_weight="balanced_subsample")
        clf.fit(X_tr, yb_tr)
        proba_cal = clf.predict_proba(X_cal)
        proba_te = clf.predict_proba(X_te)
        # クラス欠けの補正
        if proba_cal.shape[1] < N_CLASSES:
            full_cal = np.zeros((len(cal), N_CLASSES))
            full_te = np.zeros((len(te), N_CLASSES))
            for j, c in enumerate(clf.classes_):
                full_cal[:, int(c)] = proba_cal[:, j]
                full_te[:, int(c)] = proba_te[:, j]
            proba_cal, proba_te = full_cal, full_te

        s_cal = scores_from_proba(proba_cal, yb_cal)
        cal_weeks = week_idx[cal]
        win_masks = {
            "all": np.ones(len(cal), bool),
            "recent4": cal_weeks >= train_end_week - 4,
            "recent2": cal_weeks >= train_end_week - 2,
        }
        for alpha in ALPHAS:
            for win in WINDOWS:
                sc = s_cal[win_masks[win]]
                tau = conformal_tau(sc, alpha)
                res = evaluate_floor(proba_te, yb_te, tau)
                rows.append({"fold": fold_id, "target": tgt,
                             "method": "conformal", "window": win,
                             "alpha": alpha, "n_cal": int(win_masks[win].sum()),
                             "tau": None if np.isinf(tau) else round(float(tau), 4),
                             **res})
            # 無較正ベースライン: τ=α（モデル確率をそのまま信じる）
            res = evaluate_floor(proba_te, yb_te, alpha)
            rows.append({"fold": fold_id, "target": tgt,
                         "method": "uncalibrated", "window": "-",
                         "alpha": alpha, "n_cal": 0, "tau": alpha, **res})
    meta = {"fold": fold_id, "n_train": len(tr), "n_cal": len(cal),
            "n_test": len(te), "cal_weeks": n_cal_weeks,
            "train_end_week": int(train_end_week)}
    return rows, meta


def main():
    df = load_dedupe()
    y = df[TARGET].values.astype(float)
    base = build_base(df)
    corpus = text_corpus(df)
    week = df["trend_dt"].dt.to_period("W").dt.start_time
    week_idx = ((week - week.min()).dt.days // 7).values
    total_weeks = int(week_idx.max()) + 1
    folds = get_folds(total_weeks)

    os.makedirs(PART_DIR, exist_ok=True)
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg == "assemble":
        rows, metas = [], []
        for fid in range(len(folds)):
            p = os.path.join(PART_DIR, f"fold{fid}.json")
            if not os.path.exists(p):
                print(f"[warn] fold{fid} 未実行"); continue
            with open(p) as f:
                d = json.load(f)
            rows += d["rows"]; metas.append(d["meta"])
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(SCRIPT_DIR, "likes_conformal_folds.csv"),
                   index=False)
        # サマリ: target×method×window×alpha 平均
        summ = (res.groupby(["target", "method", "window", "alpha"])
                [["coverage", "mean_floor", "informative_rate"]]
                .agg(["mean", "min", "max"]).round(3))
        summ.to_csv(os.path.join(SCRIPT_DIR, "likes_conformal_summary.csv"))
        print(summ.to_string())

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, tgt in zip(axes, TARGETS):
            sub = res[(res.target == tgt) & (res.alpha == 0.10)]
            for m, w, style in [("conformal", "all", "-o"),
                                ("conformal", "recent4", "-s"),
                                ("conformal", "recent2", "-^"),
                                ("uncalibrated", "-", "--x")]:
                s = sub[(sub.method == m) & (sub.window == w)].sort_values("fold")
                lab = f"{m}({w})" if m == "conformal" else "無較正(τ=α)"
                ax.plot(s["fold"], s["coverage"], style, label=lab)
            ax.axhline(0.90, color="red", lw=1, ls=":", label="目標90%")
            ax.set_title(f"target={tgt} (α=0.10)")
            ax.set_xlabel("fold"); ax.set_ylim(0.5, 1.02)
        axes[0].set_ylabel("実測カバレッジ")
        axes[0].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, "likes_conformal_plot.png"), dpi=130)
        print("saved: likes_conformal_folds.csv / _summary.csv / _plot.png")
        return

    fold_ids = range(len(folds)) if arg == "all" else [int(arg)]
    for fid in fold_ids:
        ts, te_end = folds[fid]
        rows, meta = run_fold(fid, df, base, corpus, y, week_idx, ts, te_end)
        with open(os.path.join(PART_DIR, f"fold{fid}.json"), "w") as f:
            json.dump({"rows": rows, "meta": meta}, f, ensure_ascii=False)
        print(f"fold{fid} done: n_train={meta['n_train']} n_cal={meta['n_cal']} "
              f"n_test={meta['n_test']}")


if __name__ == "__main__":
    main()
