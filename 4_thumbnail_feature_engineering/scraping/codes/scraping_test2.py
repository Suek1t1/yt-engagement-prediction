# データセットにあるサムネイルURLから画像を5枚ダウンロードして保存するテストコード
import pandas as pd
import requests
from pathlib import Path
import time

# 1. データセットの読み込み
df = pd.read_csv('english_titles.csv')

# 2. 保存先フォルダの作成
save_dir = Path("thumbnails")
save_dir.mkdir(exist_ok=True)

# 3. サムネイルの列名を指定
url_column = 'thumbnail_link' 

# 4. 上から5件だけ取得してループ処理
for i, row in df.head(5).iterrows():
    url = row[url_column]
    save_path = save_dir / f"thumbnail_{i}.jpg"
    
    print(f"ダウンロード中... ({i+1}/5): {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # サーバー負荷軽減のための待機時間（マナー）
        time.sleep(1)
        
    except Exception as e:
        print(f"失敗しました: {url} - {e}")

print("完了しました！")