"""
サムネイル Vision メタデータ 全件取得スクリプト（ローカル実行用）
================================================================================
目的:
  `thumbnail_features_1000.csv`（811件）を全5,817件（重複除去後のユニーク動画）へ拡張し、
  `likes_thumb_meta.py` を rolling-origin 全fold評価へ切り替える。

  ※このスクリプトはユーザーのMacで実行する。sandboxでは ytimg・Google Vision API が
    ネットワーク遮断（403）のため取得不可。列フォーマットは班員の既存CSVと完全一致させる。

出力: thumbnail_features_full.csv
  列: video_id,likes,thumbnail_link,extracted_text,labels,web_entities
  （likes_thumb_meta.py の THUMB_CSV をこのファイルに差し替えれば、コード改修なしで全fold評価）

--------------------------------------------------------------------------------
事前準備:
  pip install google-cloud-vision pandas requests
  # Google Cloud で Vision API を有効化し、サービスアカウントJSONを取得して:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

実行:
  python3 fetch_thumbnail_meta.py
  # 途中中断しても再実行で続きから（既取得分はスキップ）。5,817件で数十分＋API課金に注意。

リーク鉄則（CLAUDE.md準拠）:
  - likes を Vision のプロンプト/処理に一切渡さない（投稿前情報だけで特徴を作る）。
  - video_id で重複除去済みの5,817件に対して取得（同一動画の複数行を作らない）。
--------------------------------------------------------------------------------
Vision API を使わない場合の代替:
  APIコストを避けるなら、ローカルの画像キャプション/物体検出モデル（例: BLIP, YOLO,
  OpenCLIP zero-shotラベリング）で labels/web_entities 相当を生成してもよい。
  その場合も列名と「likesを見せない」鉄則は厳守。extracted_text はOCR（pytesseract等）で代替。
"""

import ast
import os
import sys
import time
import io
import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from likes_4class_v3 import load_dedupe, TARGET

OUT_CSV = os.path.join(SCRIPT_DIR, "..", "thumbnail_features_full.csv")
EXISTING = os.path.join(SCRIPT_DIR, "..", "thumbnail_features_1000.csv")
HIRES = False          # True で hqdefault（高解像度）を使う。既存CSVは default.jpg
MAX_LABELS = 10
MAX_ENTITIES = 10
SLEEP = 0.05           # レート制御（必要に応じて調整）


def thumb_url(video_id):
    kind = "hqdefault" if HIRES else "default"
    return f"https://i.ytimg.com/vi/{video_id}/{kind}.jpg"


def vision_annotate(client, content):
    """Google Vision の Web Detection + Label + OCR を1回で取得。"""
    from google.cloud import vision
    image = vision.Image(content=content)
    feats = [vision.Feature(type_=vision.Feature.Type.WEB_DETECTION),
             vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
             vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)]
    resp = client.annotate_image({"image": image, "features": feats})
    ocr = resp.text_annotations[0].description.strip().replace("\n", " ") \
        if resp.text_annotations else ""
    labels = [l.description for l in resp.label_annotations[:MAX_LABELS]]
    entities = [e.description for e in resp.web_detection.web_entities
                if e.description][:MAX_ENTITIES]
    return ocr, labels, entities


def main():
    df = load_dedupe()[["video_id", TARGET, ]].drop_duplicates("video_id")
    df["thumbnail_link"] = df["video_id"].map(thumb_url)

    # 既存の811件を再利用（重複取得を避ける）
    done = {}
    for path in [OUT_CSV, EXISTING]:
        if os.path.exists(path):
            prev = pd.read_csv(path)
            for _, r in prev.iterrows():
                done.setdefault(r["video_id"], r)

    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    rows, n_new, n_skip, n_fail = [], 0, 0, 0
    for i, r in df.iterrows():
        vid = r["video_id"]
        if vid in done:
            rows.append(done[vid]); n_skip += 1; continue
        try:
            img = requests.get(r["thumbnail_link"], timeout=15)
            ocr, labels, entities = vision_annotate(client, img.content)
            rows.append({"video_id": vid, "likes": r[TARGET],
                         "thumbnail_link": r["thumbnail_link"],
                         "extracted_text": ocr,
                         "labels": str(labels), "web_entities": str(entities)})
            n_new += 1
        except Exception as e:
            rows.append({"video_id": vid, "likes": r[TARGET],
                         "thumbnail_link": r["thumbnail_link"],
                         "extracted_text": "", "labels": "[]", "web_entities": "[]"})
            n_fail += 1
        time.sleep(SLEEP)
        if (n_new + n_fail) % 100 == 0 and (n_new + n_fail) > 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)   # 途中保存（再開用）
            print(f"  進捗: 新規{n_new} 失敗{n_fail} 既存{n_skip} / 全{len(df)}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"完了: {OUT_CSV}  新規{n_new} 失敗{n_fail} 既存{n_skip} 合計{len(rows)}")
    print("次: likes_thumb_meta.py の THUMB_CSV を thumbnail_features_full.csv に差し替えて実行")


if __name__ == "__main__":
    main()
