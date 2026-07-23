"""
likes 4段階分類モデル（投稿時点・事前確定情報のみ／正式版）
================================================================================
問題設定:
  likes の絶対数を当てるのは事前情報だけでは難しい（真のR²≈0.4）。そこで
  likes を等頻度で4段階（低 / 中低 / 中高 / 高）に分け、投稿前に「どの段階か」
  を当てる多クラス分類にする。前回の粒度探索で4段階は正解率≈0.62、
  ±1ビン許容≈0.91 と、刻みと精度のバランスが良いことを確認済み。

使う情報（すべて投稿前に確定）:
  - タイトル表面特徴（長さ・語数・大文字率・記号有無）
  - タグ数
  - 投稿時刻（時・曜日）
  - コメント/評価の許可設定
  - カテゴリ
  - チャンネル過去平均likes（※学習期間のみで算出してリーク回避）
  - タイトル/タグの TF-IDF（300語・学習データで語彙構築）
  ※ views/dislikes/comment_count は「投稿後」の量なので一切使わない。

評価（リークなし）:
  動画重複を除去 → trending_date順に前半80%学習/後半20%test の時系列ホールドアウト。
  正解率・±1段階許容正解率・マクロF1・混同行列・段階別precision/recall。

出力:
  likes_4class_report.txt        分類レポート
  likes_4class_confusion.csv     混同行列
  likes_4class_importance.csv    特徴量重要度（数値特徴のみ抜粋）
  likes_4class_confusion.png     混同行列ヒートマップ
  likes_4class_bins.csv          各段階のlikes範囲（境界）
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
                             confusion_matrix)

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


def main():
    df = load_dedupe()
    base = build_base(df)
    y = df[TARGET].values
    n = len(df); cut = int(n * 0.8)
    tr, te = np.arange(cut), np.arange(cut, n)
    print(f"データ: {n} 動画 / 学習 {len(tr)} / test {len(te)} "
          f"(時系列ホールドアウト)")

    # --- 4段階ビン（学習データの分位点で境界決定）---
    edges = np.quantile(y[tr], np.linspace(0, 1, N_CLASSES + 1)[1:-1])
    yb_tr = np.digitize(y[tr], edges)
    yb_te = np.digitize(y[te], edges)
    bins_info = []
    lo = 0
    for i, e in enumerate(list(edges) + [y.max()]):
        bins_info.append({"class": i, "name": CLASS_NAMES[i],
                          "likes_from": int(lo), "likes_to": int(e)})
        lo = e
    bins_df = pd.DataFrame(bins_info)
    print("\n各段階のlikes範囲（学習データ基準）:")
    print(bins_df.to_string(index=False))
    bins_df.to_csv(os.path.join(SCRIPT_DIR, "likes_4class_bins.csv"), index=False)

    # --- チャンネル平均（学習期間のみ）---
    ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()

    def with_channel(idx):
        X = base.iloc[idx].copy()
        m = df.iloc[idx]["channel_title"].map(ch_means).fillna(gmean)
        X["channel_mean_likes_log"] = np.log1p(m.values)
        return X[NUM_FEATURES]

    Xn_tr = with_channel(tr); Xn_te = with_channel(te)

    # --- TF-IDF（学習データで語彙構築）---
    corpus = text_corpus(df)
    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr]); Xt_te = vec.transform(corpus[te])

    X_tr = hstack([csr_matrix(Xn_tr.values.astype(float)), Xt_tr]).tocsr()
    X_te = hstack([csr_matrix(Xn_te.values.astype(float)), Xt_te]).tocsr()

    # --- 学習 ---
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
        n_jobs=-1, class_weight="balanced_subsample")
    clf.fit(X_tr, yb_tr)
    pred = clf.predict(X_te)

    # --- 評価 ---
    acc = accuracy_score(yb_te, pred)
    adj = np.mean(np.abs(pred - yb_te) <= 1)
    f1m = f1_score(yb_te, pred, average="macro")
    chance = 1.0 / N_CLASSES
    print("\n" + "=" * 60)
    print("=== 4段階分類 評価（test=未来期間）===")
    print("=" * 60)
    print(f"正解率           : {acc:.3f}  (ランダム={chance:.3f})")
    print(f"±1段階許容正解率 : {adj:.3f}  (隣の段階まで許容)")
    print(f"マクロF1         : {f1m:.3f}")

    report = classification_report(yb_te, pred, target_names=CLASS_NAMES,
                                   digits=3)
    print("\n段階別 precision / recall / f1:")
    print(report)

    cm = confusion_matrix(yb_te, pred)
    cm_df = pd.DataFrame(cm, index=[f"真_{c}" for c in CLASS_NAMES],
                         columns=[f"予_{c}" for c in CLASS_NAMES])
    print("混同行列（行=実際, 列=予測）:")
    print(cm_df.to_string())

    # --- 保存 ---
    with open(os.path.join(SCRIPT_DIR, "likes_4class_report.txt"), "w") as f:
        f.write(f"4段階likes分類レポート\n学習{len(tr)} / test{len(te)}\n")
        f.write(f"正解率={acc:.3f} (ランダム={chance:.3f})\n")
        f.write(f"±1段階許容={adj:.3f}\nマクロF1={f1m:.3f}\n\n")
        f.write(report + "\n")
        f.write("混同行列（行=実際, 列=予測）:\n" + cm_df.to_string() + "\n")
    cm_df.to_csv(os.path.join(SCRIPT_DIR, "likes_4class_confusion.csv"))

    # 特徴量重要度（数値特徴のみ抜粋）
    imp_num = clf.feature_importances_[:len(NUM_FEATURES)]
    imp_df = (pd.DataFrame({"feature": NUM_FEATURES, "importance": imp_num})
              .sort_values("importance", ascending=False).reset_index(drop=True))
    tfidf_total = clf.feature_importances_[len(NUM_FEATURES):].sum()
    print(f"\n特徴量重要度（数値特徴）/ TF-IDF全体の合計={tfidf_total:.3f}:")
    print(imp_df.to_string(index=False))
    imp_df.to_csv(os.path.join(SCRIPT_DIR, "likes_4class_importance.csv"),
                  index=False)

    # 混同行列ヒートマップ
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("予測"); ax.set_ylabel("実際")
    ax.set_title(f"4段階likes分類 混同行列\n正解率={acc:.2f} / ±1段階={adj:.2f}")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    p = os.path.join(SCRIPT_DIR, "likes_4class_confusion.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
    print(f"\n-> レポート/混同行列/重要度/図 を保存しました")


if __name__ == "__main__":
    main()
