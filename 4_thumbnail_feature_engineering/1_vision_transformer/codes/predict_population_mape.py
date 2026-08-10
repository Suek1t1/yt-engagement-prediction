# 母集団のMAPEを予測するコード
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import japanize_matplotlib

# 1. 提示いただいたデータを入力
sizes = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 7710])
mapes = np.array([650.26, 756.58, 548.17, 519.84, 482.72, 547.03, 405.39, 327.30])

# 2. 予測用の関数（べき乗モデル: データが増えるほど誤差が減少して底を打つ）
def power_law(x, a, b, c):
    # c が「無限にデータを増やしたときのMAPEの限界値（底）」になります
    return a * (x ** b) + c

# 3. カーブフィッティングを実行
# ※初期値 p0 は、計算が収束しやすくなるように設定する「アタリ」です
popt, _ = curve_fit(power_law, sizes, mapes, p0=[10000, -0.5, 300], maxfev=10000)

# 4. 母集団（例: 37,000件）でのMAPEを予測
population_size = 37000
predicted_mape = power_law(population_size, *popt)

print(f"データ数が {population_size} 件になったときの予測MAPE: {predicted_mape:.2f}%")
print(f"無限にデータを増やしたときの限界MAPE (底): {popt[2]:.2f}%")

# 予測曲線をグラフ化
x_pred = np.linspace(1000, 40000, 100)
y_pred = power_law(x_pred, *popt)

plt.figure(figsize=(8, 5))
plt.scatter(sizes, mapes, color='blue', label='実際のデータ (Actual)')
plt.plot(x_pred, y_pred, color='red', linestyle='--', label='予測曲線 (Trend)')
plt.axvline(x=population_size, color='green', linestyle=':', label=f'母集団 ({population_size}件)')
plt.xlabel('データ数 (Data Size)')
plt.ylabel('MAPE (%)')
plt.title('母集団MAPEの予測')
plt.legend()
plt.grid(True)

# プロットの保存
plt.savefig("4_thumbnail_feature_engineering/1_vision_transformer/figures/predict_population_mape.png")

# 画面への表示
plt.show()
plt.close()

