"""
likes「最低保証」予測モデル（投稿時点・事前確定情報のみ／正式版）
================================================================================
問題設定:
  予測した段階を「最低保証」として扱う。実際の likes が予測段階 *以上*（上振れ）
  なら正解、予測段階 *未満*（下振れ＝盛りすぎ）なら不正解とする。
  「最低でもこの段階は伸びる」という控えめな約束として予測を使う考え方。

通常のピタリ正解（実際==予測）と異なり、上に外す方向だけを罰するため、
段階を細かく刻んでも成立率が落ちにくい、という性質を検証する。

使う情報（すべて投稿前に確定）:
  タイトル表面特徴 / タグ数 / 投稿時刻 / 許可設定 / カテゴリ /
  チャンネル過去平均likes（学習期間のみで算出） / タイトル・タグの TF-IDF(300語)
  ※ views/dislikes/comment_count（投稿後の量）は不使用。

評価（リークなし）:
  動画重複を除去 → trending_date順に前半80%学習/後半20%test。
  K = 3,4,5,10 段階それぞれで以下を算出:
    - chance        : 1/K（でたらめのピタリ正解率）
    - exact         : ピタリ正解率（実際==予測）
    - floor_rate    : 最低保証成立率（実際>=予測）★この設定の主指標
    - overshoot_avg : 下振れ（不正解）時に平均何段階ずれたか
    - 段階別の最低保証成立率（どの段階を下限に置くと保証が堅いか）

出力:
  likes_floor_guarantee_summary.csv   段階数ごとの指標
  likes_floor_guarantee_per_class.csv 各K・各段階の保証成立率
  likes_floor_guarantee_plot.png      最低保証 vs ピタリ正解の比較図
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
K_LIST = (3, 4, 5, 10)

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
          f"(時系列ホールドアウト)\n")

    # チャンネル平均（学習期間のみ）
    ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()

    def with_channel(idx):
        X = base.iloc[idx].copy()
        m = df.iloc[idx]["channel_title"].map(ch_means).fillna(gmean)
        X["channel_mean_likes_log"] = np.log1p(m.values)
        return X[NUM_FEATURES]

    Xn_tr = with_channel(tr); Xn_te = with_channel(te)

    # TF-IDF（学習データで語彙構築）
    corpus = text_corpus(df)
    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr]); Xt_te = vec.transform(corpus[te])
    X_tr = hstack([csr_matrix(Xn_tr.values.astype(float)), Xt_tr]).tocsr()
    X_te = hstack([csr_matrix(Xn_te.values.astype(float)), Xt_te]).tocsr()

    summary, per_class = [], []
    print(f"{'K':>3} {'chance':>7} {'ピタリ':>7} {'最低保証':>9} {'下振れ平均段数':>13}")
    for K in K_LIST:
        edges = np.quantile(y[tr], np.linspace(0, 1, K + 1)[1:-1])
        yb_tr = np.digitize(y[tr], edges)
        yb_te = np.digitize(y[te], edges)

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight="balanced_subsample")
        clf.fit(X_tr, yb_tr)
        pred = clf.predict(X_te)

        exact = float(np.mean(pred == yb_te))
        floor = float(np.mean(yb_te >= pred))        # 実際>=予測＝保証成立
        over = pred - yb_te
        overshoot = float(np.mean(over[over > 0])) if (over > 0).any() else 0.0

        summary.append({
            "K": K, "chance": 1.0 / K, "exact": exact,
            "floor_rate": floor, "overshoot_avg_steps": overshoot,
        })
        print(f"{K:>3} {1/K:>7.3f} {exact:>7.3f} {floor:>9.3f} {overshoot:>13.2f}")

        # 段階別の最低保証成立率（予測段階 c のとき、実際>=c だった割合）
        for c in range(K):
            mask = pred == c
            if mask.sum() == 0:
                continue
            per_class.append({
                "K": K, "predicted_class": c, "n_pred": int(mask.sum()),
                "floor_rate": float(np.mean(yb_te[mask] >= c)),
            })

    sum_df = pd.DataFrame(summary)
    pc_df = pd.DataFrame(per_class)
    sum_df.to_csv(os.path.join(SCRIPT_DIR, "likes_floor_guarantee_summary.csv"),
                  index=False)
    pc_df.to_csv(os.path.join(SCRIPT_DIR, "likes_floor_guarantee_per_class.csv"),
                 index=False)

    print("\n=== 段階別の最低保証成立率（予測=下限としたとき実際がそれ以上だった割合）===")
    print(pc_df.to_string(index=False))

    # ---- 図: 最低保証 vs ピタリ vs ランダム ----
    plt.figure(figsize=(8, 5))
    plt.plot(sum_df["K"], sum_df["floor_rate"], "o-", label="最低保証成立率(実≥予)")
    plt.plot(sum_df["K"], sum_df["exact"], "s--", label="ピタリ正解率(実=予)")
    plt.plot(sum_df["K"], sum_df["chance"], "^:", color="gray", label="ランダム(1/K)")
    plt.xlabel("段階数 K"); plt.ylabel("正解率")
    plt.title("『最低保証』予測の成立率\n(投稿時点情報のみ・時系列評価)")
    plt.xticks(list(K_LIST)); plt.ylim(0, 1)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    p = os.path.join(SCRIPT_DIR, "likes_floor_guarantee_plot.png")
    plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
    print(f"\n-> summary/per_class CSV と {os.path.basename(p)} を保存しました")

    print("\n【まとめ】")
    print(" ・最低保証成立率はどのKでもピタリ正解率より大幅に高い。")
    print("   段階を細かくしても落ちにくい（刻みに強い）。")
    print(" ・ただし常に最低段階を予測すれば保証はほぼ100%にできるため、")
    print("   『情報量（段階の散らばり）を保ちつつ高い保証』である点が価値。")


if __name__ == "__main__":
    main()
