"""
チャンネル知名度（トレンド入り回数）で層別した likes 予測の比較
================================================================================
問い:
  「有名チャンネル（トレンド常連）」と「無名チャンネル（トレンド初・少数）」では
  投稿前 likes の予測しやすさはどれだけ違うか。無名群では内容特徴（タイトル/タグ
  /TF-IDF）はどこまで効くか。

知名度の定義（リーク回避）:
  「学習期間内」に何本トレンド入りしたか（ユニーク動画数）。全期間で数えると
  未来情報リークになるため、学習期間のカウントのみを使う。
  分布（学習期間1832ch）: 64%が1本、中央値1本、上位9%が6本以上。これを踏まえ
    famous : 学習期間に 6 本以上（トレンド常連）
    mid    : 2〜5 本
    obscure: 1 本のみ（実績ほぼなし）
    cold   : 学習期間に未出現の完全新規チャンネル（test専用・コールドスタート）

評価（リークなし・全群共通）:
  動画重複除去 → trending_date順 前半80%学習/後半20%test の時系列ホールドアウト。
  チャンネル平均likesは学習期間のみから算出。
  test を上記の群に振り分け、群ごとに以下を測る:
    - n_test            : test件数
    - spearman          : 予測 likes と実 likes の順位相関（粒度に依らない予測力）
    - auc_median        : 「その群の学習中央値より上か」の二値分類AUC
    - r2_log            : log1p(likes) の R²
  併せて 2 つのモデルを比較し「内容の正味の効き」を切り分ける:
    - full    : チャンネル平均likes + 内容特徴(表面+TF-IDF)
    - content : 内容特徴のみ（チャンネル平均likesを抜く）
  ※学習は全データで1つのモデル、評価だけ群別。これにより「同じ予測器が
    群ごとにどれだけ当たるか」を公平に比較できる。

出力:
  likes_fame_split_results.csv   群×モデルの指標表
  likes_fame_split_plot.png      群別の予測力（Spearman/AUC）棒グラフ
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
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, roc_auc_score

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

BASE_FEATURES = [
    "title_len", "title_word_count", "title_upper_ratio",
    "title_has_excl", "title_has_question", "tag_count",
    "publish_hour", "publish_dow",
    "comments_disabled", "ratings_disabled", "category_id",
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


def fame_group(count):
    """学習期間のトレンド入り本数 -> 群名。count=0 は完全新規(cold)。"""
    if count == 0:
        return "cold"
    if count >= 6:
        return "famous"
    if count >= 2:
        return "mid"
    return "obscure"


def main():
    df = load_dedupe()
    base = build_base(df)
    y = df[TARGET].values
    ylog = np.log1p(y)
    n = len(df); cut = int(n * 0.8)
    tr, te = np.arange(cut), np.arange(cut, n)

    # --- 学習期間のチャンネル別トレンド入り本数で群を決める ---
    train_counts = df.iloc[tr].groupby("channel_title").size()
    te_counts = df.iloc[te]["channel_title"].map(train_counts).fillna(0).astype(int)
    te_group = te_counts.map(fame_group).values

    print("=" * 64)
    print("チャンネル知名度（学習期間トレンド入り本数）で層別")
    print("=" * 64)
    grp_order = ["famous", "mid", "obscure", "cold"]
    grp_label = {"famous": "有名(6本+)", "mid": "中堅(2-5本)",
                 "obscure": "無名(1本)", "cold": "新規(学習期間に無し)"}
    for g in grp_order:
        print(f"  {grp_label[g]:<22}: test {int((te_group==g).sum()):>4} 本")

    # --- チャンネル平均likes（学習期間のみ）---
    ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()
    ch_feat_all = df["channel_title"].map(ch_means).fillna(gmean)
    chan_log = np.log1p(ch_feat_all.values)

    # --- TF-IDF（学習データで語彙構築）---
    corpus = text_corpus(df)
    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr]); Xt_te = vec.transform(corpus[te])

    Xb = base[BASE_FEATURES].values.astype(float)

    # tr/te 用の特徴量をまとめてスタック
    def stack(idx, xt, mode):
        parts = [csr_matrix(Xb[idx])]
        if mode == "full":
            parts.append(csr_matrix(chan_log[idx].reshape(-1, 1)))
        parts.append(xt)
        return hstack(parts).tocsr()

    rows = []
    preds_store = {}
    for mode in ("full", "content"):
        X_tr = stack(tr, Xt_tr, mode)
        X_te = stack(te, Xt_te, mode)
        reg = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        reg.fit(X_tr, ylog[tr])
        pred_log = reg.predict(X_te)
        preds_store[mode] = pred_log

        # 全体 & 群別の指標
        for g in ["ALL"] + grp_order:
            if g == "ALL":
                m = np.ones(len(te), dtype=bool)
            else:
                m = te_group == g
            if m.sum() < 10:
                rows.append({"group": g, "model": mode, "n_test": int(m.sum()),
                             "spearman": None, "auc_median": None, "r2_log": None})
                continue
            yt = y[te][m]; pl = pred_log[m]
            # Spearman（順位相関）
            sp = spearmanr(pl, yt).correlation
            # その群の「学習中央値」より上か の二値AUC
            thr = np.median(ylog[tr])
            ybin = (np.log1p(yt) > thr).astype(int)
            try:
                auc = roc_auc_score(ybin, pl) if ybin.min() != ybin.max() else None
            except Exception:
                auc = None
            r2l = r2_score(np.log1p(yt), pl)
            rows.append({"group": g, "model": mode, "n_test": int(m.sum()),
                         "spearman": round(float(sp), 3),
                         "auc_median": round(float(auc), 3) if auc is not None else None,
                         "r2_log": round(float(r2l), 3)})

    res = pd.DataFrame(rows)
    res["group_label"] = res["group"].map(lambda g: grp_label.get(g, "全体"))
    res.to_csv(os.path.join(SCRIPT_DIR, "likes_fame_split_results.csv"),
               index=False)

    print("\n" + "=" * 64)
    print("群 × モデル の予測力（test=未来期間）")
    print("=" * 64)
    show = res[["group_label", "model", "n_test", "spearman",
                "auc_median", "r2_log"]]
    print(show.to_string(index=False))

    # --- 図: 群別 Spearman（full vs content）---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = [("spearman", "Spearman順位相関"), ("auc_median", "二値分類AUC")]
    x = np.arange(len(grp_order))
    w = 0.35
    for ax, (col, ttl) in zip(axes, metrics):
        for i, mode in enumerate(("full", "content")):
            vals = []
            for g in grp_order:
                r = res[(res.group == g) & (res.model == mode)]
                vals.append(r[col].values[0] if len(r) and r[col].values[0]
                            is not None else 0)
            ax.bar(x + (i - 0.5) * w, vals, w,
                   label="full(実績+内容)" if mode == "full" else "content(内容のみ)")
        ax.set_xticks(x)
        ax.set_xticklabels([grp_label[g] for g in grp_order], rotation=20,
                           ha="right", fontsize=9)
        ax.set_title(ttl); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=9)
    fig.suptitle("チャンネル知名度別の likes 予測力\n"
                 "（投稿時点情報のみ・時系列評価）")
    fig.tight_layout()
    p = os.path.join(SCRIPT_DIR, "likes_fame_split_plot.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close()

    print("\n【まとめ】")
    print(" ・有名群ほど full モデルの予測力が高く、無名/新規群ほど落ちるはず。")
    print(" ・content(内容のみ)モデルが有名群でも無名群でもどれだけ当たるかで、")
    print("   『内容の正味の効き』が知名度に依らないかが分かる。")
    print("-> results.csv / plot.png を保存しました")


if __name__ == "__main__":
    main()
