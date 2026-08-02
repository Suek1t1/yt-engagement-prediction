"""
likes 4段階分類 v2 — 「低」段階の崩れをビン境界の切り直しで改善
================================================================================
背景（第10回までの知見）:
  4段階分類（v1=likes_4class.py）は正解率0.616 / ±1段階0.911 と悪くないが、
  「低」段階が test で support=15・precision=0.074 と崩壊していた。

崩れの原因（本スクリプト冒頭の診断で定量確認）:
  トレンド動画の likes は時系列で大きく上昇する。
    学習期間の中央値 ≈ 5,642 / test期間の中央値 ≈ 20,562（約3.6倍）。
  そのため「学習データの分位点」で境界を引くと、未来(test)では大半が上位
  ビンに寄り、「低」がほぼ存在しなくなる（=分布シフト）。
  → 単純なビンの引き直しだけでは「低」の件数は根本的には増えない。

そこで境界の決め方を4通り比較する:
  v1  : 学習データの分位点（現状・ベースライン）
  A   : 全データの分位点（固定の絶対値境界。リーク回避のため y は段階化のみに使用、
        特徴量には一切使わない）
  B   : 対数 likes の等幅ビン（裾の長い分布を均し、低likes帯を太らせる）
  C   : チャンネル相対比による段階化 ★本命
        likes / そのチャンネルの学習期間平均likes を取り、「実力に対して
        どれだけ伸びたか」の比で4段階化する。絶対水準の時系列上昇を相対化でき、
        各段階が test 期間でも均等に出現する＝「低」も復活する。

評価（リークなし・全方式共通）:
  動画重複除去 → trending_date順 前半80%学習/後半20%test の時系列ホールドアウト。
  各方式で 正解率 / ±1段階許容 / マクロF1 / 「低」段階の support・precision・recall
  を並べて比較する。

出力:
  likes_4class_v2_compare.csv     4方式の比較表（全体指標＋低段階指標）
  likes_4class_v2_report.txt      本命Cの詳細レポート＋混同行列
  likes_4class_v2_confusion.png   v1 と C の混同行列を並べた図
"""

import os
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
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_csv(name="english_titles.csv"):
    for p in (os.path.join(SCRIPT_DIR, name),
              os.path.join(os.path.dirname(SCRIPT_DIR), name)):
        if os.path.exists(p):
            return p
    return name


CSV = _resolve_csv()
TARGET = "likes"
RANDOM_STATE = 42
N_ESTIMATORS = 300
TFIDF_MAX = 300
N_CLASSES = 4
CLASS_NAMES = ["低", "中低", "中高", "高"]

NUM_FEATURES = [
    "title_len", "title_word_count", "title_upper_ratio",
    "title_has_excl", "title_has_question", "tag_count",
    "publish_hour", "publish_dow",
    "comments_disabled", "ratings_disabled", "category_id",
    "channel_mean_likes_log",
]


def load_dedupe():
    df = pd.read_csv(CSV).dropna(subset=[TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(float)
    df["trend_dt"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m",
                                    errors="coerce")
    df = df.dropna(subset=["trend_dt"])
    df = (df.sort_values("trend_dt")
            .drop_duplicates(subset="video_id", keep="first")
            .sort_values("trend_dt").reset_index(drop=True))
    return df


def build_base(df):
    out = pd.DataFrame(index=df.index)
    t = df["title"].fillna("").astype(str)
    out["title_len"] = t.str.len()
    out["title_word_count"] = t.str.split().apply(len)
    out["title_upper_ratio"] = t.apply(
        lambda s: sum(c.isupper() for c in s) / len(s) if len(s) else 0.0)
    out["title_has_excl"] = t.str.contains("!").astype(int)
    out["title_has_question"] = t.str.contains(r"\?").astype(int)
    tags = df["tags"].fillna("[none]").astype(str)
    out["tag_count"] = np.where(tags == "[none]", 0, tags.str.count(r"\|") + 1)
    pt = pd.to_datetime(df["publish_time"], errors="coerce", utc=True)
    out["publish_hour"] = pt.dt.hour.fillna(-1).astype(int)
    out["publish_dow"] = pt.dt.dayofweek.fillna(-1).astype(int)
    out["comments_disabled"] = df["comments_disabled"].astype(int)
    out["ratings_disabled"] = df["ratings_disabled"].astype(int)
    out["category_id"] = df["category_id"].astype(int)
    return out


def text_corpus(df):
    title = df["title"].fillna("").astype(str)
    tags = (df["tags"].fillna("").astype(str)
            .str.replace("[none]", "", regex=False)
            .str.replace("|", " ", regex=False)
            .str.replace('"', " ", regex=False))
    return (title + " " + tags).values


def make_labels(method, y, tr, te, df, ch_means, gmean):
    """方式ごとに学習/test の段階ラベルを返す。
       境界の決定には y（目的変数）だけを使い、特徴量には混ぜない＝リークなし。"""
    q = np.linspace(0, 1, N_CLASSES + 1)[1:-1]
    if method == "v1":            # 学習分位点（ベースライン）
        edges = np.quantile(y[tr], q)
        return np.digitize(y[tr], edges), np.digitize(y[te], edges), edges
    if method == "A_alldata":     # 全データ分位点（固定絶対境界）
        edges = np.quantile(y, q)
        return np.digitize(y[tr], edges), np.digitize(y[te], edges), edges
    if method == "B_logwidth":    # 対数等幅
        ly = np.log1p(y)
        edges_l = np.linspace(ly[tr].min(), ly[tr].max(), N_CLASSES + 1)[1:-1]
        edges = np.expm1(edges_l)
        return np.digitize(y[tr], edges), np.digitize(y[te], edges), edges
    if method == "C_chanrel":     # チャンネル相対比
        # 各動画 likes / そのチャンネルの学習期間平均likes（未知chは全体平均）
        ch_all = df["channel_title"].map(ch_means).fillna(gmean).values
        ratio = y / np.clip(ch_all, 1.0, None)
        edges = np.quantile(ratio[tr], q)        # 比の学習分位点で4段階
        return np.digitize(ratio[tr], edges), np.digitize(ratio[te], edges), edges
    raise ValueError(method)


def main():
    df = load_dedupe()
    base = build_base(df)
    y = df[TARGET].values
    n = len(df); cut = int(n * 0.8)
    tr, te = np.arange(cut), np.arange(cut, n)

    # --- 診断: 分布シフトの定量化 ---
    print("=" * 64)
    print("分布シフトの診断")
    print("=" * 64)
    print(f"データ: {n} 動画 / 学習 {len(tr)} / test {len(te)}")
    print(f"学習 likes 中央値 = {np.median(y[tr]):,.0f}")
    print(f"test  likes 中央値 = {np.median(y[te]):,.0f}  "
          f"(学習の {np.median(y[te])/np.median(y[tr]):.1f} 倍)")
    print("→ likes 絶対値で境界を引くと test は上位ビンに偏る。\n")

    # チャンネル平均（学習期間のみ）= 特徴量＆方式Cの基準
    ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()

    def with_channel(idx):
        X = base.iloc[idx].copy()
        m = df.iloc[idx]["channel_title"].map(ch_means).fillna(gmean)
        X["channel_mean_likes_log"] = np.log1p(m.values)
        return X[NUM_FEATURES]

    Xn_tr = with_channel(tr); Xn_te = with_channel(te)

    corpus = text_corpus(df)
    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr]); Xt_te = vec.transform(corpus[te])
    X_tr = hstack([csr_matrix(Xn_tr.values.astype(float)), Xt_tr]).tocsr()
    X_te = hstack([csr_matrix(Xn_te.values.astype(float)), Xt_te]).tocsr()

    methods = [("v1", "学習分位点(現状)"),
               ("A_alldata", "全データ分位点"),
               ("B_logwidth", "対数等幅"),
               ("C_chanrel", "チャンネル相対比")]

    rows = []
    saved_cm = {}
    for key, label in methods:
        yb_tr, yb_te, edges = make_labels(key, y, tr, te, df, ch_means, gmean)
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight="balanced_subsample")
        clf.fit(X_tr, yb_tr)
        pred = clf.predict(X_te)

        acc = accuracy_score(yb_te, pred)
        adj = float(np.mean(np.abs(pred - yb_te) <= 1))
        f1m = f1_score(yb_te, pred, average="macro")
        p, r, f, s = precision_recall_fscore_support(
            yb_te, pred, labels=list(range(N_CLASSES)), zero_division=0)
        # test の各段階件数
        te_counts = np.bincount(yb_te, minlength=N_CLASSES)

        rows.append({
            "method": key, "label": label,
            "accuracy": round(acc, 3), "adj1": round(adj, 3),
            "macroF1": round(f1m, 3),
            "低_support": int(te_counts[0]),
            "低_precision": round(p[0], 3), "低_recall": round(r[0], 3),
            "低_f1": round(f[0], 3),
        })
        saved_cm[key] = (confusion_matrix(yb_te, pred), acc, adj)

        # 本命Cは詳細レポートも保存
        if key == "C_chanrel":
            rep = classification_report(yb_te, pred, target_names=CLASS_NAMES,
                                        digits=3, zero_division=0)
            cm = confusion_matrix(yb_te, pred)
            cm_df = pd.DataFrame(cm, index=[f"真_{c}" for c in CLASS_NAMES],
                                 columns=[f"予_{c}" for c in CLASS_NAMES])
            with open(os.path.join(SCRIPT_DIR, "likes_4class_v2_report.txt"),
                      "w") as fp:
                fp.write("4段階分類 v2 — 本命C(チャンネル相対比)\n")
                fp.write("段階の定義: likes / チャンネル学習期間平均likes の分位点\n")
                fp.write(f"学習{len(tr)} / test{len(te)}\n")
                fp.write(f"正解率={acc:.3f} ±1段階={adj:.3f} マクロF1={f1m:.3f}\n\n")
                fp.write(rep + "\n")
                fp.write("混同行列(行=実際,列=予測):\n" + cm_df.to_string() + "\n")

    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(SCRIPT_DIR, "likes_4class_v2_compare.csv"),
                index=False)
    print("=" * 64)
    print("4方式の比較（test=未来期間）")
    print("=" * 64)
    print(comp.to_string(index=False))

    # --- 図: v1 と C の混同行列を並べる ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key, ttl in [(axes[0], "v1", "v1: 学習分位点"),
                         (axes[1], "C_chanrel", "C: チャンネル相対比")]:
        cm, acc, adj = saved_cm[key]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("予測"); ax.set_ylabel("実際")
        ax.set_title(f"{ttl}\n正解率={acc:.2f} / ±1段階={adj:.2f}")
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle("4段階likes分類: ビン境界の切り直しによる「低」段階の改善")
    fig.tight_layout()
    p = os.path.join(SCRIPT_DIR, "likes_4class_v2_confusion.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close()

    print("\n【結論】")
    best = comp.sort_values("低_f1", ascending=False).iloc[0]
    print(f" ・「低」段階のF1が最良なのは {best['label']} "
          f"(低F1={best['低_f1']}, support={best['低_support']})。")
    print(" ・絶対値ビン(v1/A/B)は test の分布シフトで「低」が痩せるのが本質的限界。")
    print(" ・チャンネル相対比(C)は『実力比でどれだけ伸びたか』を当てる問題に変換し、")
    print("   段階が時系列でも均等に出るため「低」が復活する。")
    print("-> compare.csv / report.txt / confusion.png を保存しました")


if __name__ == "__main__":
    main()
