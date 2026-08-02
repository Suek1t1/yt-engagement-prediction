# 画像認識APIを使って、画像内のラベルや文字を検出するテストコード
import os
from google.cloud import vision

# 1. 認証キーのパスを指定
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "thumbnail-analysis-research-ad4fe8627de1.json"

def detect_labels(path):
    """画像内のラベル（何が写っているか）を検出する"""
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # ラベル検出を実行
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print('検出されたラベル:')
    for label in labels:
        print(f"{label.description} (信頼度: {label.score:.2%})")

def detect_text(path):
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as f:
        content = f.read()
    image = vision.Image(content=content)

    # text_detection を使う
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        print("検出された文字:")
        print(texts[0].description) # 最初に画像全体の一括テキストが出る
    else:
        print("文字は検出されませんでした")
# 実行
thumbnail_path = 'test_thumbnail5.png'
detect_labels(thumbnail_path)
detect_text(thumbnail_path)