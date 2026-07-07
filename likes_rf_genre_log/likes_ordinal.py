"""
方向性7: 順序を使う — ordinal分類 / 分位点回帰への置き換え ★安い割に効くはず
================================================================================
背景:
  現行のD4(時間相対比・最終水準持ち越し)4段階分類はRandomForestClassifierで
  4段階を「名義（順序なし）クラス」として学習している。「低と高を間違える」のと
  「低と中低を間違える」のを同じ重みで学習しており、段階に順序があるという情報を
  捨てている。

比較する3モデル（同一の80/20系列・同一特徴量、ターゲットの扱い方だけ差し替え）:
  D4_baseline: 現行のRandomForestClassifierによる名義4クラス分類（既存ベースライン）。
  ordinal    : 3つの二値分類器 P(段階>低), P(段階>中低), P(段階>中高) に分解し、
               累積確率の差分から4クラスの確率を作り argmax で予測。
  quantile   : GradientBoostingRegressor(loss="quantile")でlog(時間相対比)の
               中央値(50%)を直接回帰予測し、D4と同じ学習分位点の境界で段階に変換。
               さらに10%分位点も回帰し、「下限予測」として最低保証(方向性3)に
               将来接続できる形で保存する。

評価: 方向性6のrolling-origin基盤（fold0〜4）をプールして評価する
  （単一80/20分割は境界依存が大きいと第13回で判明済みのため）。

合格基準: ±1段階とマクロF1・低F1が D4_baseline 比で改善するか。

sandbox 45秒対策: fold毎に分割実行→assemble。
  python3 likes_ordinal.py <fold_id>
  python3 likes_ordinal.py assemble

出力:
  likes_ordinal_folds.csv     fold×method の指標
  likes_ordinal_summary.csv   method別の平均・範囲
  likes_ordinal_floor.csv     quantileモデルの10%分位点による最低保証成立率
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from likes_4class_v3 import (load_dedupe, build_base, text_corpus,
                             weekly_denominators, NUM_FEATURES, N_CLASSES,
                             CLASS_NAMES, TARGET, RANDOM_STATE, TFIDF_MAX)
from likes_rolling_eval import get_folds

PART_DIR = os.path.join(SCRIPT_DIR, "_ordinal_parts")
N_ESTIMATORS_RF = 300
N_ESTIMATORS_GBR = 150
METHODS = ["D4_baseline", "ordinal", "quantile"]
METHOD_LABEL = {
    "D4_baseline": "D4基準(名義4クラスRF)",
    "ordinal": "ordinal(累積二値分類×3)",
    "quantile": "quantile(GBR分位点回帰+閾値切り)",
}


def with_channel(df, idx, ch_means, gmean, base):
    X = base.iloc[idx].copy()
    m = df.iloc[idx]["channel_title"].map(ch_means).fillna(gmean)
    X["channel_mean_likes_log"] = np.log1p(m.values)
    return X[NUM_FEATURES]


def run_fold(fold_id, df, base, corpus, y, week_idx, train_end_week, test_end_week):
    tr = np.where(week_idx < train_end_week)[0]
    te = np.where((week_idx >= train_end_week) & (week_idx < test_end_week))[0]

    (denom_d1, denom_d2, denom_d3, denom_d4, week, med_by_week, tr_med,
     coef, coef3, last_level) = weekly_denominators(df, y, tr)
    ratio = y / np.clip(denom_d4, 1.0, None)

    q = np.linspace(0, 1, N_CLASSES + 1)[1:-1]
    edges = np.quantile(ratio[tr], q)
    yb_tr = np.digitize(ratio[tr], edges)
    yb_te = np.digitize(ratio[te], edges)

    ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()
    Xn_tr = with_channel(df, tr, ch_means, gmean, base)
    Xn_te = with_channel(df, te, ch_means, gmean, base)

    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr]); Xt_te = vec.transform(corpus[te])
    X_tr = hstack([csr_matrix(Xn_tr.values.astype(float)), Xt_tr]).tocsr()
    X_te = hstack([csr_matrix(Xn_te.values.astype(float)), Xt_te]).tocsr()

    def eval_pred(pred, yb_te):
        acc = accuracy_score(yb_te, pred)
        adj = float(np.mean(np.abs(pred - yb_te) <= 1))
        f1m = f1_score(yb_te, pred, average="macro")
        p, r, f, s = precision_recall_fscore_support(
            yb_te, pred, labels=list(range(N_CLASSES)), zero_division=0)
        return {"accuracy": round(float(acc), 3), "adj1": round(adj, 3),
                "macroF1": round(float(f1m), 3),
                "低_support": int(s[0]), "低_f1": round(float(f[0]), 3)}

    results = {}

    # --- D4_baseline: 名義4クラスRF ---
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE,
                                 n_jobs=-1, class_weight="balanced_subsample")
    clf.fit(X_tr, yb_tr)
    pred_base = clf.predict(X_te)
    results["D4_baseline"] = eval_pred(pred_base, yb_te)

    # --- ordinal: P(class>0), P(class>1), P(class>2) の3二値分類 ---
    cum_probs_te = []
    for c in range(N_CLASSES - 1):
        bin_tr = (yb_tr > c).astype(int)
        if len(np.unique(bin_tr)) < 2:
            # 全train行が片側に寄る場合は定数確率で代用
            p_te = np.full(len(te), float(bin_tr.mean()))
        else:
            bclf = RandomForestClassifier(n_estimators=N_ESTIMATORS_RF,
                                          random_state=RANDOM_STATE, n_jobs=-1,
                                          class_weight="balanced_subsample")
            bclf.fit(X_tr, bin_tr)
            p_te = bclf.predict_proba(X_te)[:, list(bclf.classes_).index(1)]
        cum_probs_te.append(p_te)
    # 累積確率 P(y>0)>=P(y>1)>=P(y>2) から各クラス確率を作る（負値はクリップ）
    cum = np.vstack([np.ones(len(te))] + cum_probs_te + [np.zeros(len(te))])  # (5, n_te)
    class_probs = np.clip(cum[:-1] - cum[1:], 0, None).T  # (n_te, 4)
    pred_ord = np.argmax(class_probs, axis=1)
    results["ordinal"] = eval_pred(pred_ord, yb_te)

    # --- quantile: GBR分位点回帰(50%)→D4と同じ学習分位点境界で段階化 ---
    log_ratio_tr = np.log1p(ratio[tr])
    gbr50 = GradientBoostingRegressor(loss="quantile", alpha=0.5,
                                      n_estimators=N_ESTIMATORS_GBR,
                                      max_depth=3, random_state=RANDOM_STATE)
    gbr50.fit(X_tr.toarray() if hasattr(X_tr, "toarray") else X_tr, log_ratio_tr)
    pred_log_ratio_te = gbr50.predict(
        X_te.toarray() if hasattr(X_te, "toarray") else X_te)
    pred_ratio_te = np.clip(np.expm1(pred_log_ratio_te), 0, None)
    pred_q = np.digitize(pred_ratio_te, edges)
    results["quantile"] = eval_pred(pred_q, yb_te)

    # --- 10%分位点回帰（最低保証floorの下ごしらえ・方向性3への接続用）---
    gbr10 = GradientBoostingRegressor(loss="quantile", alpha=0.1,
                                      n_estimators=N_ESTIMATORS_GBR,
                                      max_depth=3, random_state=RANDOM_STATE)
    gbr10.fit(X_tr.toarray() if hasattr(X_tr, "toarray") else X_tr, log_ratio_tr)
    pred_log_floor_te = gbr10.predict(
        X_te.toarray() if hasattr(X_te, "toarray") else X_te)
    pred_floor_ratio_te = np.clip(np.expm1(pred_log_floor_te), 0, None)
    pred_floor_likes_te = pred_floor_ratio_te * np.clip(denom_d4[te], 1.0, None)
    coverage = float(np.mean(y[te] >= pred_floor_likes_te))

    floor_info = {"fold": fold_id, "n_test": int(len(te)),
                  "target_quantile": 0.1, "empirical_coverage": round(coverage, 3)}

    for k in METHODS:
        results[k]["fold"] = fold_id
        results[k]["method"] = k
        results[k]["n_test"] = int(len(te))

    return results, floor_info


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    df = load_dedupe()
    base = build_base(df)
    corpus = text_corpus(df)
    y = df[TARGET].values
    week = df["trend_dt"].dt.to_period("W").dt.start_time
    week_idx = ((week - week.min()).dt.days / 7).round().astype(int).values
    total_weeks = int(week_idx.max()) + 1
    folds = get_folds(total_weeks)
    use_folds = list(range(min(5, len(folds))))  # fold5は方向性6の知見によりn小・参考値のため除外

    os.makedirs(PART_DIR, exist_ok=True)

    print("=" * 64)
    print("方向性7: ordinal分類 / 分位点回帰（D4ターゲット・rolling-origin基盤）")
    print("=" * 64)

    if arg != "assemble":
        fold_ids = use_folds if arg == "all" else [int(arg)]
        for fid in fold_ids:
            train_end_week, test_end_week = folds[fid]
            print(f"--- fold {fid}: train weeks[0,{train_end_week}) "
                  f"test weeks[{train_end_week},{test_end_week}) ---")
            results, floor_info = run_fold(fid, df, base, corpus, y, week_idx,
                                           train_end_week, test_end_week)
            with open(os.path.join(PART_DIR, f"fold{fid}.json"), "w") as fp:
                json.dump({"results": results, "floor": floor_info}, fp,
                          ensure_ascii=False)
            for k in METHODS:
                r = results[k]
                print(f"  [{METHOD_LABEL[k]}] acc={r['accuracy']:.3f} "
                      f"adj1={r['adj1']:.3f} macroF1={r['macroF1']:.3f} "
                      f"低F1={r['低_f1']:.3f}")
            print(f"  [10%分位点floor] 実測カバレッジ={floor_info['empirical_coverage']:.3f}"
                  f" (目標90%)")
        if arg != "all":
            print(f"(対象fold={use_folds}。全部揃ったら "
                  f"'python3 likes_ordinal.py assemble' を実行)")
            return

    # ---- assemble ----
    all_rows, floor_rows = [], []
    for fid in use_folds:
        path = os.path.join(PART_DIR, f"fold{fid}.json")
        if not os.path.exists(path):
            print(f"WARNING: fold {fid} 未実行。"
                  f"'python3 likes_ordinal.py {fid}' を先に実行してください。")
            return
        with open(path) as fp:
            d = json.load(fp)
        for k in METHODS:
            all_rows.append(d["results"][k])
        floor_rows.append(d["floor"])

    comp = pd.DataFrame(all_rows)
    comp["label"] = comp["method"].map(METHOD_LABEL)
    comp.to_csv(os.path.join(SCRIPT_DIR, "likes_ordinal_folds.csv"), index=False)

    summary = comp.groupby(["method", "label"]).agg(
        n_folds=("fold", "count"),
        acc_mean=("accuracy", "mean"), acc_min=("accuracy", "min"), acc_max=("accuracy", "max"),
        adj1_mean=("adj1", "mean"), adj1_min=("adj1", "min"), adj1_max=("adj1", "max"),
        macroF1_mean=("macroF1", "mean"),
        low_f1_mean=("低_f1", "mean"), low_f1_min=("低_f1", "min"), low_f1_max=("低_f1", "max"),
    ).reset_index().round(3)
    summary.to_csv(os.path.join(SCRIPT_DIR, "likes_ordinal_summary.csv"), index=False)

    floor_df = pd.DataFrame(floor_rows)
    floor_df.to_csv(os.path.join(SCRIPT_DIR, "likes_ordinal_floor.csv"), index=False)

    print("\n" + "=" * 64)
    print(f"3手法 × {len(use_folds)}fold の比較（pooled対象fold={use_folds}）")
    print("=" * 64)
    print(comp[["fold", "label", "n_test", "accuracy", "adj1", "macroF1",
               "低_f1"]].to_string(index=False))
    print()
    print(summary.to_string(index=False))

    print("\n【合格基準チェック】D4_baseline比でadj1・低F1が改善するか")
    base_row = summary[summary.method == "D4_baseline"].iloc[0]
    for k in ("ordinal", "quantile"):
        r = summary[summary.method == k].iloc[0]
        ok_adj = r["adj1_mean"] >= base_row["adj1_mean"]
        ok_low = r["low_f1_mean"] > base_row["low_f1_mean"]
        print(f"  {METHOD_LABEL[k]}: adj1 {base_row['adj1_mean']}->{r['adj1_mean']} "
              f"({'○' if ok_adj else '×'}) / 低F1 {base_row['low_f1_mean']}->"
              f"{r['low_f1_mean']} ({'○' if ok_low else '×'})")

    print("\n10%分位点floorの実測カバレッジ（目標90%）:")
    print(floor_df.to_string(index=False))
    print(f"平均カバレッジ = {floor_df['empirical_coverage'].mean():.3f}")

    print("\n-> folds.csv / summary.csv / floor.csv を保存しました")


if __name__ == "__main__":
    main()
