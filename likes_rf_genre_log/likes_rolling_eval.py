"""
方向性6: rolling-origin 評価基盤（walk-forward evaluation）
================================================================================
背景:
  これまでの評価は「動画重複除去→trending_date順80/20一発分割」のみだった。
  境界の位置次第で結果が変わっていないか、レジーム転換（2018年3月頃のlikesジャンプ）
  をまたぐと何が起きるかを検証するため、学習窓を伸ばしながらtest窓を4週ずつ
  後ろへスライドさせる rolling-origin（expanding window）評価に格上げする。

仕様（改善ロードマップ.md 方向性6 準拠）:
  - 学習開始固定・学習終了(=test開始)を4週ずつ後ろへスライド（expanding window）。
  - 全30週のデータに対し、最小学習窓=10週、test窓=4週で folds を作る
    → fold: train[0:10) test[10:14) / train[0:14) test[14:18) / ... の5 fold。
  - 各foldで v1（学習分位点・現行）と D4（時間相対比・最終水準持ち越し）の
    4段階分類を評価。D4の分母（最終水準）は fold ごとに学習期間のみから再計算
    （週次更新運用のシミュレーション）。
  - fold別の指標を記録し、平均±範囲で報告。fold2(train0-13,test14-17)は
    レジーム転換（2018-03-05週）をまたぐfoldになる想定。

sandbox 45秒対策: v3 と同様に分割実行に対応。
  python3 likes_rolling_eval.py <fold_id>   # fold単体を実行して部品を保存
  python3 likes_rolling_eval.py all         # 全fold逐次実行（時間に余裕があれば）
  python3 likes_rolling_eval.py assemble    # 部品を集めて表・図を作る

出力:
  likes_rolling_eval_folds.csv    fold×method の生指標
  likes_rolling_eval_summary.csv  method別の平均・最小・最大（誤差幅）
  likes_rolling_eval_plot.png     fold別の正解率/±1段階/低F1の推移図
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from likes_4class_v3 import (load_dedupe, build_base, text_corpus,
                             weekly_denominators, make_labels, NUM_FEATURES,
                             N_ESTIMATORS, RANDOM_STATE, TFIDF_MAX, N_CLASSES,
                             CLASS_NAMES, TARGET)

PART_DIR = os.path.join(SCRIPT_DIR, "_rolling_parts")
TRAIN_MIN_WEEKS = 10   # 最小学習窓（週）
TEST_WEEKS = 4         # test窓（週）
METHODS = [("v1", "学習分位点(現状)"),
           ("D4_naive", "時間相対比(最終水準持ち越し)")]


def get_folds(total_weeks):
    """expanding window: train=[0,train_end) / test=[train_end,test_end)"""
    folds = []
    ts = TRAIN_MIN_WEEKS
    while ts < total_weeks:
        te_end = min(ts + TEST_WEEKS, total_weeks)
        if te_end > ts:
            folds.append((ts, te_end))
        ts += TEST_WEEKS
    return folds


def run_fold(fold_id, df, base, corpus, y, week_idx, train_end_week, test_end_week):
    tr = np.where(week_idx < train_end_week)[0]
    te = np.where((week_idx >= train_end_week) & (week_idx < test_end_week))[0]
    n_train, n_test = len(tr), len(te)

    (denom_d1, denom_d2, denom_d3, denom_d4, week, med_by_week, tr_med,
     coef, coef3, last_level) = weekly_denominators(df, y, tr)

    ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()

    def with_channel(idx):
        X = base.iloc[idx].copy()
        m = df.iloc[idx]["channel_title"].map(ch_means).fillna(gmean)
        X["channel_mean_likes_log"] = np.log1p(m.values)
        return X[NUM_FEATURES]

    Xn_tr = with_channel(tr); Xn_te = with_channel(te)
    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr]); Xt_te = vec.transform(corpus[te])
    X_tr = hstack([csr_matrix(Xn_tr.values.astype(float)), Xt_tr]).tocsr()
    X_te = hstack([csr_matrix(Xn_te.values.astype(float)), Xt_te]).tocsr()

    results = {}
    for key, label in METHODS:
        yb_tr, yb_te, edges = make_labels(key, y, tr, te, df, ch_means, gmean,
                                          denom_d1, denom_d2, denom_d3, denom_d4)
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced_subsample")
        clf.fit(X_tr, yb_tr)
        pred = clf.predict(X_te)

        acc = accuracy_score(yb_te, pred)
        adj = float(np.mean(np.abs(pred - yb_te) <= 1))
        f1m = f1_score(yb_te, pred, average="macro")
        p, r, f, s = precision_recall_fscore_support(
            yb_te, pred, labels=list(range(N_CLASSES)), zero_division=0)
        te_counts = np.bincount(yb_te, minlength=N_CLASSES)

        results[key] = {
            "fold": fold_id, "method": key, "label": label,
            "train_end_week": int(train_end_week),
            "test_end_week": int(test_end_week),
            "n_train": int(n_train), "n_test": int(n_test),
            "accuracy": round(float(acc), 3), "adj1": round(adj, 3),
            "macroF1": round(float(f1m), 3),
            "低_support": int(te_counts[0]), "低_f1": round(float(f[0]), 3),
        }
    return results


def main():
    # 使い方: python3 likes_rolling_eval.py <fold_id|all> / assemble
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    df = load_dedupe()
    base = build_base(df)
    corpus = text_corpus(df)
    y = df[TARGET].values
    week = df["trend_dt"].dt.to_period("W").dt.start_time
    week_idx = ((week - week.min()).dt.days / 7).round().astype(int).values
    total_weeks = int(week_idx.max()) + 1
    folds = get_folds(total_weeks)

    os.makedirs(PART_DIR, exist_ok=True)

    if arg != "assemble":
        if arg == "all":
            fold_ids = list(range(len(folds)))
        else:
            fold_ids = [int(arg)]
        for fid in fold_ids:
            train_end_week, test_end_week = folds[fid]
            print(f"--- fold {fid}: train weeks[0,{train_end_week}) "
                  f"test weeks[{train_end_week},{test_end_week}) ---")
            res = run_fold(fid, df, base, corpus, y, week_idx,
                           train_end_week, test_end_week)
            with open(os.path.join(PART_DIR, f"fold{fid}.json"), "w") as fp:
                json.dump(res, fp, ensure_ascii=False)
            for key, r in res.items():
                print(f"  [{r['label']}] acc={r['accuracy']:.3f} "
                      f"adj1={r['adj1']:.3f} macroF1={r['macroF1']:.3f} "
                      f"低F1={r['低_f1']:.3f} (n_test={r['n_test']})")
        if arg != "all":
            print(f"(全fold数={len(folds)}。全部揃ったら"
                  f" 'python3 likes_rolling_eval.py assemble' を実行)")
            return

    # ---- assemble ----
    rows = []
    for fid in range(len(folds)):
        path = os.path.join(PART_DIR, f"fold{fid}.json")
        if not os.path.exists(path):
            print(f"WARNING: fold {fid} の結果が未生成です。"
                  f"'python3 likes_rolling_eval.py {fid}' を先に実行してください。")
            return
        with open(path) as fp:
            res = json.load(fp)
        for key, r in res.items():
            rows.append(r)

    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(SCRIPT_DIR, "likes_rolling_eval_folds.csv"),
                index=False)

    summary = comp.groupby(["method", "label"]).agg(
        n_folds=("fold", "count"),
        acc_mean=("accuracy", "mean"), acc_min=("accuracy", "min"),
        acc_max=("accuracy", "max"),
        adj1_mean=("adj1", "mean"), adj1_min=("adj1", "min"),
        adj1_max=("adj1", "max"),
        macroF1_mean=("macroF1", "mean"),
        low_f1_mean=("低_f1", "mean"), low_f1_min=("低_f1", "min"),
        low_f1_max=("低_f1", "max"),
    ).reset_index().round(3)
    summary.to_csv(os.path.join(SCRIPT_DIR, "likes_rolling_eval_summary.csv"),
                    index=False)

    print("\n" + "=" * 64)
    print(f"rolling-origin評価: {len(folds)} folds × {len(METHODS)} methods")
    print("=" * 64)
    print(comp.to_string(index=False))
    print()
    print(summary.to_string(index=False))

    # --- 図: fold別の指標推移 ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for key, label in METHODS:
        sub = comp[comp.method == key].sort_values("fold")
        axes[0].plot(sub.fold, sub.accuracy, "o-", label=label)
        axes[1].plot(sub.fold, sub.adj1, "o-", label=label)
        axes[2].plot(sub.fold, sub["低_f1"], "o-", label=label)
    for ax, ttl in zip(axes, ["正解率", "±1段階", "低F1"]):
        ax.set_xlabel("fold"); ax.set_title(ttl); ax.legend()
        ax.set_xticks(comp.fold.unique())

    # レジーム転換(2018-03-05週)がどのfoldに入るか注記
    regime_week_idx = int(((pd.Timestamp("2018-03-05") - week.min()).days) / 7)
    regime_fold = None
    for i, (ts, te_) in enumerate(folds):
        if ts <= regime_week_idx < te_:
            regime_fold = i
            break
    if regime_fold is not None:
        for ax in axes:
            ax.axvline(regime_fold, color="red", ls=":", alpha=0.6)
        fig.suptitle(f"rolling-origin評価: fold別指標の推移"
                     f"（赤線=fold{regime_fold}がレジーム転換週をまたぐ）")
    else:
        fig.suptitle("rolling-origin評価: fold別指標の推移")
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, "likes_rolling_eval_plot.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    print("\n【読み方の目安】")
    print(" ・acc_min〜acc_max の幅が広ければ、単一境界の一発評価は運が良い/悪いだけ"
          "の可能性がある。")
    print(" ・regime転換をまたぐfoldでD4/v1がどう動くかに注目"
          "（v3の知見: D2は転換で過小評価、D4は横ばい前提で堅実なはず）。")
    print("-> folds.csv / summary.csv / plot.png を保存しました")


if __name__ == "__main__":
    main()
