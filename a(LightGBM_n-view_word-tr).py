import numpy as np
import pandas as pd
import torch
import lightgbm as lgb  # 👈 LightGBMをインポート！
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import re

# ==========================================
# ⚙️ 設定・モデルの準備
# ==========================================
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_bert = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model_bert = model_bert.to(device)
model_bert.eval()

# 1. CSV読み込み
df = pd.read_csv("english_titles.csv")

# 2. 欠損値の事前処理
df["description"] = df["description"].fillna("")
df["tags"] = df["tags"].fillna("")
df["title"] = df["title"].fillna("")

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

with torch.no_grad():
    for title in tqdm(df["title"]):
        inputs = tokenizer(title, padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
        outputs = model_bert(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        title_features.append(embeddings)

df_bert = pd.DataFrame(title_features, columns=[f"bert_{i}" for i in range(768)])

# 4. 特徴量(X)とターゲット(y)の選定
X_meta = df[[
    "title_len", "desc_len", "tag_count", "tag_len", 
    "publish_hour", "publish_dayofweek", "category_id", 
    "has_exclamation", "has_question", "is_uppercase"
]]

X = pd.concat([X_meta, df_bert], axis=1)

# 💡 LightGBM用のエラー対策：列名に変な記号が入らないように綺麗にする
X.columns = [re.sub(r'[\[\]<>\s,:]', '_', str(col)) for col in X.columns]

# ターゲット（対数変換）
y = np.log1p(df["likes"])

import gc  # 👈 メモリを強制的に掃除するライブラリをインポート

# ==========================================
# 5. 特徴量の結合とメモリ解放（重複列対策版）
# ==========================================
print("--- 特徴量を結合中 ---")

# メタデータ（X）と BERTデータ（df_bert）を合体
X_final = pd.concat([X.reset_index(drop=True), pd.DataFrame(df_bert).reset_index(drop=True)], axis=1)

# ⚠️【超重要】もう使わない古いデータを即座に削除してメモリ解放
del X
del df_bert
if 'df' in locals(): del df  
gc.collect()  # ゴミ箱を空にする

# 列名をすべて文字列に変える
X_final.columns = X_final.columns.astype(str)

# 🚨【新機能】もし列名が重複していたら、2回目以降に登場した重複列を削除する
if X_final.columns.duplicated().any():
    print("⚠️ 重複した列名（bert_0など）を発見したため、重複分を削除します。")
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=0
)

# ⚠️ 分割し終わったら、合体データも削除してさらにメモリを空ける！
del X_final
gc.collect()


# ==========================================
# 👑 6. LightGBMモデルの学習（超省エネ・絶対完走版）
# ==========================================
print("--- LightGBM学習中（省エネモード） ---")
import lightgbm as lgb

model = lgb.LGBMRegressor(
    random_state=0,
    n_estimators=300,       # 👈 500回だとまだ重かったので、300回に調整（これでも超強いです）
    learning_rate=0.05,     # 👈 その分、少し学習の歩幅を大きく
    max_depth=4,            # 👈 木の深さを「4」に絞る（これでメモリ消費が劇的に減ります！）
    num_leaves=15,          # 👈 深さ4に合わせて、葉っぱの数も15に制限
    n_jobs=1,               # 👈 使うコアを完全に「1つ」に固定（並列処理によるメモリ爆発を防ぐ最強の防御策）
    min_child_samples=20
)

# 学習を実行！
model.fit(X_train, y_train)
print("--- 学習完了！ ---")
# 7. 予測と評価
y_pred = model.predict(X_test)

y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)

mask = y_test_original >= 10
y_test_filtered = y_test_original[mask]
y_pred_filtered = y_pred_original[mask]

print("\n=== LightGBM 予測結果 ===")
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

