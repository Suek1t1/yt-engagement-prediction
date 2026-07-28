# URLから画像をダウンロードして保存するテストコード
import requests
from pathlib import Path

# 保存先フォルダを作成
save_dir = Path("downloaded_images")
save_dir.mkdir(exist_ok=True)

def download_image(url, save_path):
    try:
        # 画像データを取得
        response = requests.get(url, stream=True)
        # ステータスコードが200(成功)か確認
        response.raise_for_status()
        
        # バイナリモードでファイルに書き込み
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"保存完了: {save_path}")
        
    except Exception as e:
        print(f"失敗しました: {url} - {e}")

# サンプルURL
image_url = "https://i.ytimg.com/vi/1ZAPwfrtAFY/default.jpg"
file_name = save_dir / "thumbnail.jpg"

download_image(image_url, file_name)