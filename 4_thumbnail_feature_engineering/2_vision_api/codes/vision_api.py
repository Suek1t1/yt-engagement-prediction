# thumbnail_features_1000.csvのサムネイル画像をGoogle Cloud Vision APIで解析し、OCR、ラベル、Web検出の結果を取得するコード
import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from google.cloud import vision

# --- 初期設定 ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "thumbnail-analysis-research-ad4fe8627de1.json"

def analyze_thumbnail_from_url(image_url, client):
    """
    URLから画像をメモリにダウンロードし、1回のAPI呼び出しでOCR、ラベル、Web検出を行う
    """
    try:
        # --- 1. 画像をメモリ上に取得 ---
        response = requests.get(image_url, timeout=10)
        
        # 404 Not Found などのエラー（動画削除済みなど）の場合はここで例外を発生させる
        response.raise_for_status() 
        content = response.content

        # --- 2. Vision APIの準備 ---
        image = vision.Image(content=content)
        features = [
            vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
            vision.Feature(type_=vision.Feature.Type.WEB_DETECTION),
        ]
        request = vision.AnnotateImageRequest(image=image, features=features)
        
        # --- 3. APIリクエスト実行 ---
        api_response = client.annotate_image(request=request)

        # --- 4. 結果の抽出 ---
        extracted_text = ""
        if api_response.text_annotations:
            extracted_text = api_response.text_annotations[0].description.replace('\n', ' ')

        labels = []
        if api_response.label_annotations:
            labels = [label.description for label in api_response.label_annotations]

        web_entities = []
        if api_response.web_detection and api_response.web_detection.web_entities:
            web_entities = [entity.description for entity in api_response.web_detection.web_entities if entity.description]

        return extracted_text, labels, web_entities

    except requests.exceptions.RequestException as e:
        # リンク切れや通信エラーの場合
        print(f"\n[スキップ] 画像取得エラー ({image_url})")
        return None, None, None
    except Exception as e:
        # API側のエラーの場合
        print(f"\n[スキップ] API解析エラー ({image_url})")
        return None, None, None

def main():
    print("データを読み込んでいます...")
    df_all = pd.read_csv("datasets/2_feature_engineerings/english_titles.csv") 
    
    # --------------------------------------------------
    # 【追加】事前フィルター：サムネイルURLがない行を除外
    # --------------------------------------------------
    # 1. 完全な欠損値 (NaN, None) を落とす
    df_all = df_all.dropna(subset=['thumbnail_link'])
    # 2. 空文字（""）や空白だけの行を落とす
    df_all = df_all[df_all['thumbnail_link'].astype(str).str.strip() != '']
    
    print(f"有効なURLを持つデータ件数: {len(df_all)}件")

    # ランダムに1000件抽出
    df_sample = df_all.sample(n=1000, random_state=42).copy()
    print("1000件のサンプリング完了。解析を開始します...")

    client = vision.ImageAnnotatorClient()
    
    results_text = []
    results_labels = []
    results_entities = []

    for index, row in tqdm(df_sample.iterrows(), total=1000, desc="Analyzing Images"):
        image_url = row['thumbnail_link'] 
        
        text, labels, entities = analyze_thumbnail_from_url(image_url, client)
        
        results_text.append(text)
        results_labels.append(labels)
        results_entities.append(entities)
        
        time.sleep(0.1)

    # 結果をデータフレームに結合
    df_sample['extracted_text'] = results_text
    df_sample['labels'] = results_labels
    df_sample['web_entities'] = results_entities

    columns_to_save = ['video_id', 'likes', 'thumbnail_link', 'extracted_text', 'labels', 'web_entities']
    df_final = df_sample[columns_to_save]
    
    # --------------------------------------------------
    # 【追加】事後フィルター：画像取得や解析に失敗した行を除外
    # --------------------------------------------------
    # エラー時は None を返しているので、None が含まれる行を削除する
    # （※ OCRで文字がなかっただけの行は "" が入るため削除されません）
    df_final = df_final.dropna(subset=['extracted_text'])

    output_filename = "datasets/4_thumbnail_feature_engineering/2_vision_api/thumbnail_features_1000.csv"
    df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n解析完了！")
    print(f"エラーを除外した最終的なデータ件数: {len(df_final)}件")
    print(f"結果を {output_filename} に保存しました。")

if __name__ == "__main__":
    main()