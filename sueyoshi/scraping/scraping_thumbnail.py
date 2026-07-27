import pandas as pd
import requests
import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# =========================
# 設定
# =========================
BATCH_SIZE = 32

# Apple SiliconならGPU(MPS)を使う
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# =========================
# モデルロード
# =========================
processor = ViTImageProcessor.from_pretrained(
    'google/vit-base-patch16-224-in21k'
)
model = ViTModel.from_pretrained(
    'google/vit-base-patch16-224-in21k'
).to(device)
model.eval()

# =========================
# データ読み込み
# =========================
df = pd.read_csv('info-dm-g5/english_titles.csv')
#sample_df = df.sample(n=30, random_state=42)
sample_df = df.sample(n=10000, random_state=42).reset_index(drop=True)

processed_data = []

# =========================
# バッチ処理
# =========================
for start in tqdm(range(0, len(sample_df), BATCH_SIZE)):
    batch_df = sample_df.iloc[start:start+BATCH_SIZE]

    images = []
    rows = []

    # 画像取得
    for _, row in batch_df.iterrows():
        try:
            response = requests.get(row['thumbnail_link'], timeout=5)
            if response.status_code != 200:
                continue

            img = Image.open(BytesIO(response.content)).convert('RGB')
            images.append(img)
            rows.append(row)

        except Exception:
            continue

    if len(images) == 0:
        continue

    # まとめて前処理
    inputs = processor(images=images, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # まとめて推論
    with torch.no_grad():
        outputs = model(**inputs)

    vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # 保存
    for row, vec in zip(rows, vectors):
        processed_data.append({
            'video_id': row['video_id'],
            'title': row['title'],
            'channel_title': row['channel_title'],
            'category_id': row['category_id'],
            'publish_time': row['publish_time'],
            'tags': row['tags'],
            'thumbnail_link': row['thumbnail_link'],
            'likes': row['likes'],
            'vector': vec
        })

# =========================
# 保存
# =========================
final_df = pd.DataFrame(processed_data)
final_df.to_pickle('info-dm-g5/sample_10000_data.pkl')

print(f'完了: {len(final_df)}件保存しました')