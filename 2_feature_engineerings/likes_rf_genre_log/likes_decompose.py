"""
方向性1: log-likes の加法分解モデル ★最優先・本命
================================================================================
狙い:
  log(likes) = 時代効果 + チャンネル効果 + 内容効果 + ノイズ を正面から分解し、
  「時代・発信者・内容が各何%を説明するか」を1枚の分散分解表にする。

  これまでのC(チャンネル相対比)とD(時間相対比)は、実はこの構造の部分修正を
  別々にやっていただけ。統一すると、これまでの知見（発信者支配・新規ch予測不能・
  分布シフト）が全部この1つの表に統合される。

段階的残差回帰（すべて各foldの学習期間のみから統計量を算出。testには一切触れない）:
  1. 時代効果 = D4方式の週次水準（学習期間の週次中央値、test週は最終水準持ち越し）
     を log で引く。D4は `likes_4class_v3.weekly_denominators` の denom_d4 を再利用。
  2. チャンネル効果 = 残差(r1)に対する経験ベイズ収縮チャンネル平均
     （方向性2 `likes_shrinkage.py` と同じ式・同じk=1。対象がlikesの生値ではなく
     r1=時代効果を引いた後の残差である点だけが違う）。
  3. 内容効果 = 残った残差(r2)を投稿時点特徴（タイトル・タグのTF-IDF、title統計、
     publish_hour/dow、category_id）でRF回帰。

【重要な設計判断】単一の80/20分割ではなく方向性6のrolling-origin基盤（6fold）に乗せる:
  単一test期間だと「時代効果」はtest内で一定値（D4=最終水準の持ち越し）になるため、
  定数を引いても test内分散は数学的に不変（Var(X-c)=Var(X)）で、時代効果の寄与率が
  常に0%と出てしまう（実際に試して確認済み）。これは分解手法の欠陥ではなく
  「1回きりのtest期間では時代水準は定数」という当然の帰結。
  → 方向性6で作った6foldをプールし、fold間で時代水準が変動する状態で分散分解する。
    これにより時代効果の寄与が正しく可視化される。

sandbox 45秒対策: fold毎に分割実行→assemble（v3/rolling_evalと同じ方式）。
  python3 likes_decompose.py <fold_id>   # fold単体の3段階分解を実行して部品を保存
  python3 likes_decompose.py assemble    # 全fold分をプールして寄与率表・図を作る

評価: 各段階でtestの分散をどれだけ削ったかを記録し、寄与率(%)の表を作る
  （時代/チャンネル/内容/未説明 の4つで合計100%になるように正規化。全fold pooled）。
  カテゴリ別にも同じ分解を出す（Sports易/Gaming難の「なぜ」を確認）。

合格基準: 寄与率の表が出ること。内容効果の残差R²(test, out-of-sample, pooled)が0を
  超えるか（超えなければ「内容は時代とチャンネルを統制すると説明力ゼロ」という
  強い結論として書く）。

出力:
  likes_decompose_contrib.csv       全体の寄与率テーブル（段階別・累積R²、pooled）
  likes_decompose_by_category.csv   カテゴリ別の寄与率テーブル（pooled）
  likes_decompose_plot.png          カテゴリ別寄与率の積み上げ棒グラフ
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from likes_4class_v3 import (load_dedupe, build_base, text_corpus,
                             weekly_denominators, TARGET)
from likes_rolling_eval import get_folds

PART_DIR = os.path.join(SCRIPT_DIR, "_decompose_parts")
RANDOM_STATE = 42
N_ESTIMATORS = 300
TFIDF_MAX = 300
SHRINK_K = 1  # 方向性2の内部検証で選ばれたkをそのまま流用

CONTENT_FEATURES = [
    "title_len", "title_word_count", "title_upper_ratio",
    "title_has_excl", "title_has_question", "tag_count",
    "publish_hour", "publish_dow",
    "comments_disabled", "ratings_disabled", "category_id",
]

CATEGORY_NAMES = {
    1: "Film&Animation", 2: "Autos&Vehicles", 10: "Music", 15: "Pets&Animals",
    17: "Sports", 19: "Travel&Events", 20: "Gaming", 22: "People&Blogs",
    23: "Comedy", 24: "Entertainment", 25: "News&Politics", 26: "Howto&Style",
    27: "Education", 28: "Science&Technology", 29: "Nonprofits&Activism",
    43: "Shows",
}


def shrink_ref_and_apply(df, tr, values_tr, k):
    """train期間の値(values_tr、r1など)からチャンネル別・カテゴリ別平均を作り、
       経験ベイズ収縮特徴を全行(df全体)に適用して返す。"""
    tmp = df.iloc[tr][["channel_title", "category_id"]].copy()
    tmp["_v"] = values_tr
    ch_count = tmp.groupby("channel_title").size()
    ch_mean = tmp.groupby("channel_title")["_v"].mean()
    cat_mean = tmp.groupby("category_id")["_v"].mean()
    gmean = float(np.mean(values_tr))

    n_c = df["channel_title"].map(ch_count).fillna(0).values
    m_c = df["channel_title"].map(ch_mean).fillna(0).values
    cat_m = df["category_id"].map(cat_mean).fillna(gmean).values
    return (n_c * m_c + k * cat_m) / (n_c + k)


def run_fold(fold_id, df, base, corpus, y, ylog, week_idx,
            train_end_week, test_end_week):
    tr = np.where(week_idx < train_end_week)[0]
    te = np.where((week_idx >= train_end_week) & (week_idx < test_end_week))[0]

    # --- 段階1: 時代効果（D4） ---
    (denom_d1, denom_d2, denom_d3, denom_d4, week, med_by_week, tr_med,
     coef, coef3, last_level) = weekly_denominators(df, y, tr)
    log_time_effect = np.log(np.clip(denom_d4, 1.0, None))
    r1 = ylog - log_time_effect

    # --- 段階2: チャンネル効果（残差r1への経験ベイズ収縮）---
    channel_effect = shrink_ref_and_apply(df, tr, r1[tr], SHRINK_K)
    r2 = r1 - channel_effect

    # --- 段階3: 内容効果（残差r2をRF回帰）---
    Xb = base[CONTENT_FEATURES].values.astype(float)
    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr]); Xt_te = vec.transform(corpus[te])
    X_tr = hstack([csr_matrix(Xb[tr]), Xt_tr]).tocsr()
    X_te = hstack([csr_matrix(Xb[te]), Xt_te]).tocsr()

    reg = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    reg.fit(X_tr, r2[tr])
    pred_content_te = reg.predict(X_te)
    residual3_te = r2[te] - pred_content_te

    return {
        "fold": fold_id,
        "te_idx": te.tolist(),
        "category_id": df.iloc[te]["category_id"].astype(int).tolist(),
        "y0": ylog[te].tolist(),
        "r1": r1[te].tolist(),
        "r2": r2[te].tolist(),
        "residual3": residual3_te.tolist(),
        "pred_content": pred_content_te.tolist(),
    }


def main():
    # 使い方: python3 likes_decompose.py <fold_id|all> / assemble
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    df = load_dedupe()
    base = build_base(df)
    corpus = text_corpus(df)
    y = df[TARGET].values
    ylog = np.log1p(y)
    week = df["trend_dt"].dt.to_period("W").dt.start_time
    week_idx = ((week - week.min()).dt.days / 7).round().astype(int).values
    total_weeks = int(week_idx.max()) + 1
    folds = get_folds(total_weeks)
    # 最終foldはn=40程度と小さく不安定なため、方向性6と同様に参考値として除外候補にするが
    # ここでは分解の対象を「安定して評価できるfold0〜4」に絞る（方向性6の知見を踏襲）。
    use_folds = list(range(min(5, len(folds))))

    os.makedirs(PART_DIR, exist_ok=True)

    print("=" * 64)
    print("方向性1: log-likes 加法分解（時代 + チャンネル + 内容 + ノイズ）")
    print(f"rolling-origin基盤（方向性6）の fold0〜{use_folds[-1]} をプールして評価")
    print("=" * 64)

    if arg != "assemble":
        if arg == "all":
            fold_ids = use_folds
        else:
            fold_ids = [int(arg)]
        for fid in fold_ids:
            train_end_week, test_end_week = folds[fid]
            print(f"--- fold {fid}: train weeks[0,{train_end_week}) "
                  f"test weeks[{train_end_week},{test_end_week}) ---")
            res = run_fold(fid, df, base, corpus, y, ylog, week_idx,
                           train_end_week, test_end_week)
            with open(os.path.join(PART_DIR, f"fold{fid}.json"), "w") as fp:
                json.dump(res, fp)
            v0 = np.var(res["y0"]); v1 = np.var(res["r1"])
            v2 = np.var(res["r2"]); v3 = np.var(res["residual3"])
            print(f"  var0={v0:.3f} var1={v1:.3f} var2={v2:.3f} var3={v3:.3f} "
                  f"n_test={len(res['te_idx'])}")
        if arg != "all":
            print(f"(対象fold={use_folds}。全部揃ったら "
                  f"'python3 likes_decompose.py assemble' を実行)")
            return

    # ---- assemble: 全foldのtest行をプールして分散分解 ----
    all_parts = []
    for fid in use_folds:
        path = os.path.join(PART_DIR, f"fold{fid}.json")
        if not os.path.exists(path):
            print(f"WARNING: fold {fid} 未実行。"
                  f"'python3 likes_decompose.py {fid}' を先に実行してください。")
            return
        with open(path) as fp:
            all_parts.append(json.load(fp))

    y0 = np.concatenate([np.array(p["y0"]) for p in all_parts])
    r1 = np.concatenate([np.array(p["r1"]) for p in all_parts])
    r2 = np.concatenate([np.array(p["r2"]) for p in all_parts])
    residual3 = np.concatenate([np.array(p["residual3"]) for p in all_parts])
    pred_content = np.concatenate([np.array(p["pred_content"]) for p in all_parts])
    cat_ids = np.concatenate([np.array(p["category_id"]) for p in all_parts])
    fold_ids_arr = np.concatenate([np.full(len(p["y0"]), p["fold"]) for p in all_parts])

    var0 = np.var(y0); var1 = np.var(r1); var2 = np.var(r2); var3 = np.var(residual3)
    time_pct = (var0 - var1) / var0 * 100
    channel_pct = (var1 - var2) / var0 * 100
    content_pct = (var2 - var3) / var0 * 100
    unexplained_pct = var3 / var0 * 100
    content_r2 = r2_score(r2, pred_content)

    contrib = pd.DataFrame([
        {"stage": "時代効果(D4)", "累積R2": round(1 - var1 / var0, 3),
         "寄与率(%)": round(time_pct, 1)},
        {"stage": "+チャンネル効果(収縮推定)", "累積R2": round(1 - var2 / var0, 3),
         "寄与率(%)": round(channel_pct, 1)},
        {"stage": "+内容効果(RF)", "累積R2": round(1 - var3 / var0, 3),
         "寄与率(%)": round(content_pct, 1)},
        {"stage": "未説明(ノイズ)", "累積R2": None,
         "寄与率(%)": round(unexplained_pct, 1)},
    ])
    contrib.to_csv(os.path.join(SCRIPT_DIR, "likes_decompose_contrib.csv"),
                    index=False)

    print("\n" + "=" * 64)
    print(f"全体の分散分解（fold{use_folds}をプール・n_test={len(y0)}・"
          f"log(likes)のVarに対する割合）")
    print("=" * 64)
    print(contrib.to_string(index=False))
    print(f"\n内容効果の残差R²（out-of-sample, 段階2残差r2に対して, pooled）"
          f"= {content_r2:.4f}")
    if content_r2 > 0:
        print("-> 合格基準達成: 時代・チャンネルを統制した後も内容特徴に説明力が残る。")
    else:
        print("-> 合格基準未達: 時代・チャンネルを統制すると内容特徴の説明力はゼロ以下"
              "（=『何を言うか』はほぼ効かず、『いつ・誰が』でほぼ決まる、という強い結論）。")

    # --- fold別の寄与率（誤差幅の参考として）---
    fold_rows = []
    for fid in use_folds:
        m = fold_ids_arr == fid
        v0f = np.var(y0[m]); v1f = np.var(r1[m]); v2f = np.var(r2[m]); v3f = np.var(residual3[m])
        fold_rows.append({
            "fold": fid, "n_test": int(m.sum()),
            "時代(%)": round((v0f - v1f) / v0f * 100, 1) if v0f > 0 else None,
            "チャンネル(%)": round((v1f - v2f) / v0f * 100, 1) if v0f > 0 else None,
            "内容(%)": round((v2f - v3f) / v0f * 100, 1) if v0f > 0 else None,
            "未説明(%)": round(v3f / v0f * 100, 1) if v0f > 0 else None,
        })
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(SCRIPT_DIR, "likes_decompose_by_fold.csv"), index=False)
    print("\nfold別の寄与率（参考・誤差幅の目安）:")
    print(fold_df.to_string(index=False))

    # --- カテゴリ別の分散分解（pooled）---
    cat_rows = []
    for cat, cnt in pd.Series(cat_ids).value_counts().items():
        if cnt < 30:
            continue
        m = cat_ids == cat
        v0 = np.var(y0[m]); v1 = np.var(r1[m]); v2 = np.var(r2[m]); v3 = np.var(residual3[m])
        if v0 == 0:
            continue
        cat_rows.append({
            "category_id": int(cat),
            "category_name": CATEGORY_NAMES.get(int(cat), f"cat{cat}"),
            "n_test": int(cnt),
            "時代(%)": round((v0 - v1) / v0 * 100, 1),
            "チャンネル(%)": round((v1 - v2) / v0 * 100, 1),
            "内容(%)": round((v2 - v3) / v0 * 100, 1),
            "未説明(%)": round(v3 / v0 * 100, 1),
            "内容R2": round(r2_score(r2[m], pred_content[m]), 3),
        })
    cat_df = pd.DataFrame(cat_rows).sort_values("内容R2", ascending=False)
    cat_df.to_csv(os.path.join(SCRIPT_DIR, "likes_decompose_by_category.csv"),
                  index=False)

    print("\n" + "=" * 64)
    print("カテゴリ別の分散分解（pooled、test件数30以上のみ）")
    print("=" * 64)
    print(cat_df.to_string(index=False))

    # --- 図: カテゴリ別寄与率の積み上げ棒グラフ ---
    fig, ax = plt.subplots(figsize=(12, 6))
    cat_plot = cat_df.sort_values("内容R2", ascending=False)
    x = np.arange(len(cat_plot))
    bottoms = np.zeros(len(cat_plot))
    for col, color in [("時代(%)", "#8DA0CB"), ("チャンネル(%)", "#FC8D62"),
                       ("内容(%)", "#66C2A5"), ("未説明(%)", "#CCCCCC")]:
        vals = cat_plot[col].values
        ax.bar(x, vals, bottom=bottoms, label=col, color=color)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(cat_plot["category_name"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("寄与率 (%)")
    ax.set_title("カテゴリ別 log-likes 分散分解（rolling-origin, fold0-4 pooled）\n"
                 "(時代効果 + チャンネル効果(収縮推定) + 内容効果(RF) + 未説明)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, "likes_decompose_plot.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    print("\n-> contrib.csv / by_fold.csv / by_category.csv / plot.png を保存しました")


if __name__ == "__main__":
    main()
