"""
ローカル実行用: 事前学習モデルによるマルチモーダル特徴量（フェーズ3・ロマン枠）
================================================================================
sandboxはHuggingFace/ytimgへのネットワークが遮断されているため、このスクリプトは
**ユーザーのMacでローカル実行する**。`likes_dl_text.py`（スクラッチDAN）の上位互換の
検証: 事前学習の意味表現なら内容効果を拾えるか？

事前準備（初回のみ）:
  pip install sentence-transformers torch pillow requests scikit-learn scipy pandas

実行手順（likes_rf_genre_log/ で実行。所要目安: ダウンロード20分 + 埋め込み10分）:
  python3 run_local_multimodal.py text_embed   # タイトル+タグ+概要欄 → MiniLM埋め込み
  python3 run_local_multimodal.py download     # サムネイル5,817枚を _thumbs/ へ
  python3 run_local_multimodal.py clip_embed   # CLIP画像埋め込み
  python3 run_local_multimodal.py evaluate     # rolling-origin全foldでRF比較
  （text_embed → evaluate だけでも動く。CLIPは任意）

比較構成（evaluate、ターゲットD4・4段階、likes_dl_text.py と同一プロトコル）:
  ST_content   : MiniLM文埋め込み + 基本特徴（channelなし）
  ST_CLIP_content: + CLIP画像埋め込み（あれば）
  ST_full      : ST_content + channel_mean_likes_log
判定基準: RF_content_tt(acc 0.454/低F1 0.455/新規ch ρ 0.332, fold0-4平均)と
  DAN(likes_dl_text_summary.csv)を上回るか。特に新規ch群ρが焦点。

注意（リーク鉄則, CLAUDE.md参照）:
  - 埋め込みは投稿前情報(タイトル/タグ/概要欄/サムネイル)のみ。likes等は入れない。
  - 事前学習モデルは2017-18より後の知識を含むが、個別動画のlikesは知らないので
    「投稿時点で使える情報」の範疇として扱う（報告書に1行注記すること）。
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from likes_4class_v3 import (load_dedupe, build_base, weekly_denominators,
                             NUM_FEATURES, N_CLASSES, TARGET, RANDOM_STATE)
from likes_rolling_eval import get_folds
from likes_dl_text import corpus_ttd, metrics, subgroup, CONTENT_FEATURES

THUMB_DIR = os.path.join(SCRIPT_DIR, "_thumbs")
ST_NPZ = os.path.join(SCRIPT_DIR, "_st_embed.npz")
CLIP_NPZ = os.path.join(SCRIPT_DIR, "_clip_embed.npz")
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"


def stage_text_embed(df):
    from sentence_transformers import SentenceTransformer
    texts = corpus_ttd(df)
    model = SentenceTransformer(ST_MODEL)
    emb = model.encode(list(texts), batch_size=128, show_progress_bar=True,
                       normalize_embeddings=True)
    np.savez_compressed(ST_NPZ, emb=emb, video_id=df["video_id"].values)
    print("saved", ST_NPZ, emb.shape)


def stage_download(df):
    import requests
    os.makedirs(THUMB_DIR, exist_ok=True)
    ok = ng = 0
    for vid, url in zip(df["video_id"], df["thumbnail_link"].fillna("")):
        path = os.path.join(THUMB_DIR, f"{vid}.jpg")
        if os.path.exists(path) or not url:
            continue
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
                ok += 1
            else:
                ng += 1
        except Exception:
            ng += 1
    print(f"downloaded {ok}, failed {ng}, total files "
          f"{len(os.listdir(THUMB_DIR))}")


def stage_clip_embed(df):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL)
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    embs, ids = [], []
    batch, bids = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            inp = proc(images=batch, return_tensors="pt")
            e = model.get_image_features(**inp)
            e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.numpy()); ids.extend(bids)
        batch.clear(); bids.clear()

    for vid in df["video_id"]:
        p = os.path.join(THUMB_DIR, f"{vid}.jpg")
        if not os.path.exists(p):
            continue
        try:
            batch.append(Image.open(p).convert("RGB")); bids.append(vid)
        except Exception:
            continue
        if len(batch) == 64:
            flush()
            print(f"  {len(ids)} done", end="\r")
    flush()
    emb = np.vstack(embs)
    np.savez_compressed(CLIP_NPZ, emb=emb, video_id=np.array(ids))
    print("saved", CLIP_NPZ, emb.shape)


def stage_evaluate(df):
    from sklearn.ensemble import RandomForestClassifier
    d = np.load(ST_NPZ, allow_pickle=True)
    st = pd.DataFrame(d["emb"], index=d["video_id"])
    ST = st.reindex(df["video_id"]).fillna(0).values
    has_clip = os.path.exists(CLIP_NPZ)
    if has_clip:
        d2 = np.load(CLIP_NPZ, allow_pickle=True)
        cl = pd.DataFrame(d2["emb"], index=d2["video_id"])
        CL = cl.reindex(df["video_id"]).fillna(0).values

    y = df[TARGET].values.astype(float)
    base = build_base(df)
    week = df["trend_dt"].dt.to_period("W").dt.start_time
    week_idx = ((week - week.min()).dt.days // 7).values
    folds = get_folds(int(week_idx.max()) + 1)

    rows = []
    for fid, (ts, te_end) in enumerate(folds):
        tr = np.where(week_idx < ts)[0]
        te = np.where((week_idx >= ts) & (week_idx < te_end))[0]
        (_, _, _, denom_d4, *_rest) = weekly_denominators(df, y, tr)
        ratio = y / np.clip(denom_d4, 1.0, None)
        edges = np.quantile(ratio[tr], np.linspace(0, 1, N_CLASSES + 1)[1:-1])
        yb = np.digitize(ratio, edges)
        ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
        gmean = df.iloc[tr][TARGET].mean()
        seen = set(df.iloc[tr]["channel_title"])
        new_mask = ~df.iloc[te]["channel_title"].isin(seen).values

        def numeric(idx, content_only):
            X = base.iloc[idx].copy()
            if not content_only:
                m = df.iloc[idx]["channel_title"].map(ch_means).fillna(gmean)
                X["channel_mean_likes_log"] = np.log1p(m.values)
                return X[NUM_FEATURES].values.astype(float)
            return X[CONTENT_FEATURES].values.astype(float)

        configs = [("ST_content", [ST], True), ("ST_full", [ST], False)]
        if has_clip:
            configs.insert(1, ("ST_CLIP_content", [ST, CL], True))
        for cfg, mats, content_only in configs:
            X_tr = np.hstack([numeric(tr, content_only)] + [M[tr] for M in mats])
            X_te = np.hstack([numeric(te, content_only)] + [M[te] for M in mats])
            rf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                        random_state=RANDOM_STATE,
                                        class_weight="balanced_subsample")
            rf.fit(X_tr, yb[tr])
            proba = np.zeros((len(te), N_CLASSES))
            pr = rf.predict_proba(X_te)
            for j, c in enumerate(rf.classes_):
                proba[:, int(c)] = pr[:, j]
            pred = proba.argmax(1)
            esc = proba @ np.arange(N_CLASSES)
            rows.append({"fold": fid, "config": cfg,
                         **metrics(pred, yb[te]),
                         **subgroup(pred, yb[te], esc, new_mask, "newch")})
            print(fid, cfg, rows[-1]["accuracy"])
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(SCRIPT_DIR, "likes_multimodal_folds.csv"), index=False)
    cols = ["accuracy", "adj1", "macroF1", "低_f1", "newch_acc", "newch_rho"]
    print(res.groupby("config")[cols].agg(["mean", "min", "max"]).round(3))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    df = load_dedupe()
    {"text_embed": stage_text_embed, "download": stage_download,
     "clip_embed": stage_clip_embed, "evaluate": stage_evaluate}[stage](df)
