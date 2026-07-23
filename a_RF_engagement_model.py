import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt

#cd Gread3_1/info-dm-g5

#使用アルゴリズム: LightGBM
#viewなし
#word-transformerを使ったタイトルの意味ベクトル化
#画像データを使ったサムネイル特徴量の追加


# ==========================================
# ⚙️ 設定・モデルの準備
# ==========================================
# 英語の軽量版BERT（DistilBERT）を使用
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_bert = AutoModel.from_pretrained(MODEL_NAME)

# パソコンにGPUがあれば使う（処理が速くなります）
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model_bert = model_bert.to(device)
model_bert.eval() # 評価モードに設定

# ==========================================
# 1. CSV読み込みとスエキチの画像データの結合
# ==========================================
print("--- データを読み込み中 ---")
df = pd.read_csv("english_titles.csv")
print(f"元のデータの件数: {len(df)} 件")
# 友達の画像データ
df_vision = pd.read_csv("thumbnail_features_1000.csv") # ※実際のファイル名にしてください
print(f"友達のデータの件数: {len(df_vision)} 件")
# 🚨 how="inner" から how="left" に変更！（あなたの元のデータを絶対に消さない設定）
df = pd.merge(df, df_vision, on="video_id", how="left")
print(f"結合後のデータの件数: {len(df)} 件")
# 🚨 デバッグ用：本当にlikesが存在するか、データが空になっていないか画面に出す
print("現在のデータフレームの列名一覧:", df.columns.tolist())
if len(df) == 0:
    print("⚠️ 警告: データが0件になっています！video_idが一致していません。")

# ==========================================
# 2. 特徴量エンジニアリング（サムネイル特徴量の追加）
# ==========================================
# 1. サムネイルに文字が含まれているか？（文字数カウント、欠損値は0に）
df['vision_text_len'] = df['extracted_text'].fillna('').astype(str).str.len()
# 2. サムネイルに写っているオブジェクト（labels）の文字列の長さ（タグが多いほど長くなる）
df['vision_label_len'] = df['labels'].fillna('').astype(str).str.len()
# 3. ネット上の関連ワード（web_entities）の文字列の長さ
df['vision_web_len'] = df['web_entities'].fillna('').astype(str).str.len()
# ---- 元々あったあなたの特徴量作成コード（例えば視聴回数とか投稿日数の処理など）をここに残す ----
# 例: X = df[['views', 'vision_text_len', 'vision_label_len', 'vision_web_len', ...]]

# 3. 新しい特徴量の作成（これまでのメタデータ）
df["title_len"] = df["title"].str.len()
df["desc_len"] = df["description"].str.len()
df["tag_count"] = df["tags"].str.count(r"\|") + 1
df["tag_len"] = df["tags"].str.len()

df["has_exclamation"] = df["title"].str.contains(r"!").astype(int)
df["has_question"] = df["title"].str.contains(r"\?").astype(int)
df["is_uppercase"] = df["title"].str.isupper().astype(int)

df["publish_time"] = pd.to_datetime(df["publish_time"])
df["publish_hour"] = df["publish_time"].dt.hour
df["publish_dayofweek"] = df["publish_time"].dt.dayofweek

# ==========================================
# 🔥 Word Transformer (BERT) でタイトルをベクトル化
# ==========================================
print("--- タイトルの意味を分析中 (BERT処理) ---")
title_features = []

# データが多いと時間がかかるため、1件ずつテキストを数値に変換します
with torch.no_grad():
    for title in tqdm(df["title"]):
        # テキストをトークン（単語の細切れ）に分解してID化
        inputs = tokenizer(title, padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
        # BERTに通して特徴を取り出す
        outputs = model_bert(**inputs)
        # 文全体の意味を表す「[CLS]トークン」のベクトル（768次元）を抽出
        #「文全体の要約・意味」を詰め込んだ特別な数字の箱（CLSトークン）を作ってくれます。そこから、タイトルの意味がギッシリ詰まった 768個の数字の羅列（ベクトル） を引っこ抜いています。
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        title_features.append(embeddings)

# 変換したBERTの特徴（768列のデータ）をデータフレームにする
df_bert = pd.DataFrame(title_features, columns=[f"bert_{i}" for i in range(768)])
# ==========================================

# 4. 特徴量(X)とターゲット(y)の選定
# メタデータ
X_meta = df[[
    "title_len", "desc_len", "tag_count", "tag_len", 
    "publish_hour", "publish_dayofweek", "category_id", 
    "has_exclamation", "has_question", "is_uppercase"
]]

# メタデータ(10列)とBERTが作った768列のデータを横にガッチャンコ（結合）する
X = pd.concat([X_meta, df_bert], axis=1)

# ターゲット（対数変換）
y = np.log1p(df["likes"])

# 5. データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 6. モデル学習（特徴量が700個以上に増えたので少し時間がかかります）
print("--- ランダムフォレスト学習中 ---")
model = RandomForestRegressor(random_state=0, n_jobs=-1) # n_jobs=-1で全コア使って高速化
model.fit(X_train, y_train)

# 7. 予測と評価
y_pred = model.predict(X_test)

y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)

mask = y_test_original >= 10
y_test_filtered = y_test_original[mask]
y_pred_filtered = y_pred_original[mask]

print("\n=== 予測結果 ===")
print(f"平均エラー（何いいねズレているか）: {mean_absolute_error(y_test_original, y_pred_original):.1f}")
print(f"R2スコア（予測の正確さ 0~1）: {r2_score(y_test, y_pred):.4f}")
print(f"平均二乗誤差 (MSE): {mean_squared_error(y_test_original, y_pred_original):.4f}")
print(f"平均パーセント誤差: {mean_absolute_percentage_error(y_test_filtered, y_pred_filtered):.4f}")

# 📊 グラフ化
plt.scatter(y_test, y_pred, alpha=0.5)
max_val = max(max(y_test), max(y_pred))
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')
plt.xlabel("Real Likes (Log)")
plt.ylabel("Predicted Likes (Log)")
plt.show()