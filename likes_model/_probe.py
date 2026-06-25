import numpy as np, pandas as pd, lightgbm as lgb
n=5794; X=np.random.rand(n,250).astype(np.float32); y=np.random.rand(n)
m=lgb.LGBMRegressor(n_estimators=2000,learning_rate=0.03,num_leaves=63,n_jobs=2,verbose=-1)
m.fit(X[:4600],y[:4600],eval_set=[(X[4600:],y[4600:])],callbacks=[lgb.early_stopping(100,verbose=False)])
print("probe ok best_iter", m.best_iteration_)
