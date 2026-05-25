# 「dislikes」と、「views」の関係を散布図で可視化するコードです。
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd

df = pd.read_csv('USvideos.csv - Sheet1.csv')

# それぞれの散布図を
plt.scatter(df['views'], df['dislikes'], alpha=0.3, color='red')
plt.title('Views vs Dislikes')
plt.xlabel('Views')
plt.ylabel('Dislikes')
plt.show()
