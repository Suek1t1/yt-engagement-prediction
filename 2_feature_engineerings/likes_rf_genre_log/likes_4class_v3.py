"""
likes 4段階分類 v3 — 「D: 時間相対比」による分布シフトの根治を検証
================================================================================
背景（第11回までの知見）:
  「低」段階の崩れの正体は分布シフト（test期間の likes 中央値が学習期間の約3.6倍）。
  v2 で試した絶対値ビン(v1/A/B)は根治せず、チャンネル相対比(C)は「低」を復活させる
  が問題が難化して全体正解率0.344に落ちた。

そこで新方式 D「時間相対比」を追加する:
  likes を「その動画がトレンド入りした週の likes 中央値」で割った比でビンを切る。
  トレンド全体のインフレ（時間方向のシフト）が分母に吸収されるため、
  段階が期間をまたいで安定し「低」が復活するはず。
  C との違い: C は「チャンネルの実力比」なので発信者情報を目的変数に持ち込み難化する。
  D は「その時代の水準比」なので問題の中身は絶対値ビンとほぼ同じまま、
  スケールだけを補正する。

D は2変種を用意する:
  D1_week  : 分母 = その週の likes 中央値（全データから週ごとに算出）。
             ※分母は同時期の他動画から決まる量なので「週が終わるまで確定しない」。
             ターゲット定義（その時代の水準に対してどの段階か）としては正当だが、
             投稿前に絶対値へ逆変換するには分母の見積もりが要る点に注意。
  D2_trend : 分母 = 学習期間の週次中央値に log 線形トレンドを当てはめ、
             test 週へ外挿した予測中央値。完全リークフリー（未来情報ゼロ）で、
             投稿前でも分母が計算できる実運用版。

比較対象（v2 と同一条件・同一特徴量・同一分割）:
  v1 / A_alldata / C_chanrel / D1_week / D2_trend
  （B_logwidth は v2 で「低」が消滅済みのため割愛）

評価: 動画重複除去 → trending_date順 80/20 時系列ホールドアウト。
出力:
  likes_4class_v3_compare.csv     比較表（全体指標＋低段階指標）
  likes_4class_v3_report.txt      D1/D2 の詳細レポート＋混同行列
  likes_4class_v3_confusion.png   v1 / C / D2 の混同行列を並べた図
  likes_4class_v3_trend.png       週次中央値とトレンド外挿の図
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


RECENT_WEEKS = 8


def weekly_denominators(df, y, tr):
    """D1: 各動画の属す週の likes 中央値（全データ・週ごと）。
       D2: 学習週の中央値に log 線形トレンドを当てはめ全週へ外挿した予測中央値。
       D3: 学習期間の「直近 RECENT_WEEKS 週」だけで fit した外挿
           （インフレが2018年3月頃に構造的に加速するため、全期間 fit の D2 は
             過小評価する。直近だけ使えば新レジームに追従できるか検証）。"""
    week = df["trend_dt"].dt.to_period("W").dt.start_time
    wk_idx = (week - week.min()).dt.days / 7.0        # 週番号(実数)

    # D1: 週ごとの実中央値
    med_by_week = pd.Series(y).groupby(week.values).median()
    denom_d1 = week.map(med_by_week).values

    # D2: 学習期間の週次中央値 → log線形回帰で外挿
    tr_weeks = week.iloc[tr]
    tr_med = pd.Series(y[tr]).groupby(tr_weeks.values).median()
    xw = np.array([(w - week.min()).days / 7.0 for w in tr_med.index])
    coef = np.polyfit(xw, np.log(tr_med.values), 1)   # log(median) ~ a*week + b
    denom_d2 = np.exp(np.polyval(coef, wk_idx.values))

    # D3: 直近週のみで fit（学習期間内はその値、test週は外挿）
    coef3 = np.polyfit(xw[-RECENT_WEEKS:],
                       np.log(tr_med.values[-RECENT_WEEKS:]), 1)
    denom_d3 = np.exp(np.polyval(coef3, wk_idx.values))

    # D4: ナイーブ持ち越し。学習期間内は各週の実中央値（過去なので既知）、
    #     test週は「学習最終4週の中央値の中央値」を定数で持ち越す。
    #     レジーム転換後に水準が横ばいならトレンド外挿より当たる（random walk仮説）。
    last_level = float(np.median(tr_med.values[-4:]))
    denom_d4 = np.where(np.arange(len(week)) < len(tr),
                        week.map(med_by_week).values, last_level)

    return (denom_d1, denom_d2, denom_d3, denom_d4, week, med_by_week,
            tr_med, coef, coef3, last_level)


def make_labels(method, y, tr, te, df, ch_means, gmean,
                denom_d1, denom_d2, denom_d3, denom_d4):
    """境界の決定には y だけを使い、特徴量には混ぜない＝特徴量リークなし。"""
    q = np.linspace(0, 1, N_CLASSES + 1)[1:-1]

    def cut(ratio):
        edges = np.quantile(ratio[tr], q)
        return np.digitize(ratio[tr], edges), np.digitize(ratio[te], edges), edges

    if method == "v1":
        edges = np.quantile(y[tr], q)
        return np.digitize(y[tr], edges), np.digitize(y[te], edges), edges
    if method == "A_alldata":
        edges = np.quantile(y, q)
        return np.digitize(y[tr], edges), np.digitize(y[te], edges), edges
    if method == "C_chanrel":
        ch_all = df["channel_title"].map(ch_means).fillna(gmean).values
        return cut(y / np.clip(ch_all, 1.0, None))
    if method == "D1_week":
        return cut(y / np.clip(denom_d1, 1.0, None))
    if method == "D2_trend":
        return cut(y / np.clip(denom_d2, 1.0, None))
    if method == "D3_recent":
        return cut(y / np.clip(denom_d3, 1.0, None))
    if method == "D4_naive":
        return cut(y / np.clip(denom_d4, 1.0, None))
    raise ValueError(method)


ALL_METHODS = [("v1", "学習分位点(現状)"),
               ("A_alldata", "全データ分位点"),
               ("C_chanrel", "チャンネル相対比"),
               ("D1_week", "時間相対比(週中央値)"),
               ("D2_trend", "時間相対比(トレンド外挿)"),
               ("D3_recent", f"時間相対比(直近{RECENT_WEEKS}週外挿)"),
               ("D4_naive", "時間相対比(最終水準持ち越し)")]
PART_DIR = os.path.join(SCRIPT_DIR, "_v3_parts")


def main():
    # 使い方: python3 likes_4class_v3.py <method|all>  /  assemble
    #   sandbox の45秒制限対策として方式ごとに分割実行できる。
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    df = load_dedupe()
    base = build_base(df)
    y = df[TARGET].values
    n = len(df); cut_i = int(n * 0.8)
    tr, te = np.arange(cut_i), np.arange(cut_i, n)

    (denom_d1, denom_d2, denom_d3, denom_d4, week, med_by_week, tr_med,
     coef, coef3, last_level) = weekly_denominators(df, y, tr)

    print("=" * 64)
    print("v3: 時間相対比(D)による分布シフト補正")
    print("=" * 64)
    print(f"データ: {n} 動画 / 学習 {len(tr)} / test {len(te)}")
    print(f"学習 likes 中央値 = {np.median(y[tr]):,.0f} / "
          f"test = {np.median(y[te]):,.0f} "
          f"({np.median(y[te])/np.median(y[tr]):.1f}倍)")
    print(f"週次中央値トレンド: log(median) = {coef[0]:.4f}*week + {coef[1]:.2f} "
          f"(週あたり x{np.exp(coef[0]):.3f})")
    # 外挿の当たり具合（test週）
    te_weeks = week.iloc[te]
    te_med_true = pd.Series(y[te]).groupby(te_weeks.values).median()
    xw_te = np.array([(w - week.min()).days / 7.0 for w in te_med_true.index])
    preds = [("D2全期間", np.exp(np.polyval(coef, xw_te))),
             ("D3直近", np.exp(np.polyval(coef3, xw_te))),
             ("D4持ち越し", np.full(len(xw_te), last_level))]
    for nm, pv in preds:
        ratio_err = pv / te_med_true.values
        print(f"test週の外挿中央値/実中央値 [{nm}] = "
              f"{np.min(ratio_err):.2f}〜{np.max(ratio_err):.2f} "
              f"(中央値 {np.median(ratio_err):.2f})")
    print()

    ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()

    def with_channel(idx):
        X = base.iloc[idx].copy()
        m = df.iloc[idx]["channel_title"].map(ch_means).fillna(gmean)
        X["channel_mean_likes_log"] = np.log1p(m.values)
        return X[NUM_FEATURES]

    if arg != "assemble":
        Xn_tr = with_channel(tr); Xn_te = with_channel(te)
        corpus = text_corpus(df)
        vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english",
                              min_df=5)
        Xt_tr = vec.fit_transform(corpus[tr]); Xt_te = vec.transform(corpus[te])
        X_tr = hstack([csr_matrix(Xn_tr.values.astype(float)), Xt_tr]).tocsr()
        X_te = hstack([csr_matrix(Xn_te.values.astype(float)), Xt_te]).tocsr()

        os.makedirs(PART_DIR, exist_ok=True)
        methods = [m for m in ALL_METHODS if arg in ("all", m[0])]
        for key, label in methods:
            yb_tr, yb_te, edges = make_labels(key, y, tr, te, df, ch_means,
                                              gmean, denom_d1, denom_d2,
                                              denom_d3, denom_d4)
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

            row = {
                "method": key, "label": label,
                "accuracy": round(acc, 3), "adj1": round(adj, 3),
                "macroF1": round(f1m, 3),
                "低_support": int(te_counts[0]),
                "低_precision": round(p[0], 3), "低_recall": round(r[0], 3),
                "低_f1": round(f[0], 3),
            }
            rep = classification_report(yb_te, pred, target_names=CLASS_NAMES,
                                        digits=3, zero_division=0)
            cm = confusion_matrix(yb_te, pred)
            with open(os.path.join(PART_DIR, f"{key}.json"), "w") as fp:
                json.dump({"row": row, "cm": cm.tolist(), "report": rep},
                          fp, ensure_ascii=False)
            print(f"[{label}] acc={acc:.3f} adj1={adj:.3f} macroF1={f1m:.3f} "
                  f"低F1={f[0]:.3f} (support={te_counts[0]})")
        if arg != "all":
            return  # 分割実行時はここまで。全部そろったら assemble を呼ぶ。

    # ---- assemble: 部品を集めて比較表・レポート・図を作る ----
    rows, saved_cm, reports = [], {}, {}
    for key, label in ALL_METHODS:
        with open(os.path.join(PART_DIR, f"{key}.json")) as fp:
            d = json.load(fp)
        rows.append(d["row"])
        saved_cm[key] = (np.array(d["cm"]), d["row"]["accuracy"],
                         d["row"]["adj1"])
        reports[key] = d["report"]

    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(SCRIPT_DIR, "likes_4class_v3_compare.csv"),
                index=False)
    print("\n" + "=" * 64)
    print(f"{len(ALL_METHODS)}方式の比較（test=未来期間）")
    print("=" * 64)
    print(comp.to_string(index=False))

    with open(os.path.join(SCRIPT_DIR, "likes_4class_v3_report.txt"), "w") as fp:
        fp.write("4段階分類 v3 — D: 時間相対比\n")
        fp.write("D1: likes / その週のlikes中央値（実測） / "
                 "D2: likes / トレンド外挿中央値（リークフリー）\n")
        fp.write(f"週次トレンド: log(median) = {coef[0]:.4f}*week + {coef[1]:.2f}\n")
        fp.write(f"学習{len(tr)} / test{len(te)}\n\n")
        for key in ("D1_week", "D2_trend", "D3_recent", "D4_naive"):
            cm, acc, adj = saved_cm[key]
            cm_df = pd.DataFrame(cm, index=[f"真_{c}" for c in CLASS_NAMES],
                                 columns=[f"予_{c}" for c in CLASS_NAMES])
            fp.write(f"--- {key} ---\n正解率={acc:.3f} ±1段階={adj:.3f}\n")
            fp.write(reports[key] + "\n混同行列(行=実際,列=予測):\n")
            fp.write(cm_df.to_string() + "\n\n")

    # --- 図1: v1 / C / D2 の混同行列 ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key, ttl in [(axes[0], "v1", "v1: 学習分位点"),
                         (axes[1], "C_chanrel", "C: チャンネル相対比"),
                         (axes[2], "D4_naive",
                          "D4: 時間相対比(最終水準持ち越し)")]:
        cm, acc, adj = saved_cm[key]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("予測"); ax.set_ylabel("実際")
        ax.set_title(f"{ttl}\n正解率={acc:.2f} / ±1段階={adj:.2f}")
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle("4段階likes分類 v3: 時間相対比(D)による分布シフト補正")
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, "likes_4class_v3_confusion.png"),
                dpi=200, bbox_inches="tight"); plt.close()

    # --- 図2: 週次中央値とトレンド外挿 ---
    fig, ax = plt.subplots(figsize=(9, 5))
    all_weeks = med_by_week.index
    xw_all = np.array([(w - week.min()).days / 7.0 for w in all_weeks])
    ax.plot(all_weeks, med_by_week.values, "o-", label="週次中央値(実測)")
    ax.plot(all_weeks, np.exp(np.polyval(coef, xw_all)), "--",
            label="D2: log線形トレンド(学習全期間でfit)")
    ax.plot(all_weeks, np.exp(np.polyval(coef3, xw_all)), "-.",
            label=f"D3: 同(学習の直近{RECENT_WEEKS}週のみでfit)")
    te_weeks_u = [w for w in all_weeks if w >= week.iloc[te[0]]]
    ax.plot(te_weeks_u, [last_level] * len(te_weeks_u), ls=":",
            lw=2.5, label="D4: 最終水準の持ち越し")
    cut_dt = week.iloc[te[0]]
    ax.axvline(cut_dt, color="gray", ls=":", label="学習/test 境界")
    ax.set_yscale("log")
    ax.set_ylabel("週の likes 中央値 (log)"); ax.set_xlabel("週")
    ax.set_title("トレンド動画 likes のインフレと外挿")
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, "likes_4class_v3_trend.png"),
                dpi=200, bbox_inches="tight"); plt.close()

    print("\n【結論の目安】")
    print(" ・D で 低F1 が v1/A より回復し、かつ全体正解率が C より高ければ、")
    print("   「スケールだけ補正して問題は難化させない」という狙いが成立。")
    print(" ・D2 (外挿) が D1 (実測) に近ければ、リークフリー運用でも成立する。")
    print("-> compare.csv / report.txt / confusion.png / trend.png を保存しました")


if __name__ == "__main__":
    main()
