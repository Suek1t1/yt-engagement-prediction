# データセットからランダムにn件データを取り、それらのサムネイルをViTでベクトル化して保存するコード
import pandas as pd
import requests
import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time

# 1. モデルのロード (メモリ効率を考えてモデルはループ外で一度だけ定義)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
model.eval()

# 2. データ読み込みとランダム抽出
df = pd.read_csv('datasets/english_titles.csv')
sample_df = df.sample(n=15000, random_state=42) # random_stateで再現性を確保

# 3. データ格納用リスト
processed_data = []

# 4. ループ処理
for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    url = row['thumbnail_link']
    
    try:
        # 画像取得
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            continue # 画像がなければスキップ
        
        # ViTでのベクトル化処理
        img = Image.open(BytesIO(response.content)).convert('RGB')
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # [CLS]トークンのベクトル抽出
        vec = outputs.last_hidden_state[:, 0, :].numpy().flatten()
        
        # 結果を保存
        processed_data.append({
            'likes': row['likes'],
            'vector': vec
        })
        
    except Exception as e:
        print(f"Error processing {url}: {e}")
        continue

# 5. 保存 (ベクトルデータはpickleが便利です)
final_df = pd.DataFrame(processed_data)
final_df.to_pickle("datasets/sample_15000_data.pkl")
print("15,000件の処理が完了しました。")