import numpy as np
"""
基本的な線形回帰分析授業でやったものそのまま
"""
class LinearRegression:
    x = None
    theta = None
    y= None

    def predict(self, x):#予測値（定義的）
        return np.dot(x,self.theta)

    def score(self, x, y):#誤差計算
        error = self.predict(x)-y
        return(error**2).sum()
    
    alpha = None

    def __init__(self,alpha=0.1):
        self.alpha = alpha

    def fit(self,input,output):#θ
        xTx = np.dot(input.T,input)
        I = np.eye(len(xTx))
        self.theta = np.dot(np.dot(np.linalg.inv(xTx + self.alpha*I),input.T),output)
    

    