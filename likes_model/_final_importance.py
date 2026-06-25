import numpy as np,pandas as pd,json,lightgbm as lgb
Xs=np.load("likes_model/Xs.npy");Xt=np.load("likes_model/Xt.npy");y=np.load("likes_model/y.npy")
meta=pd.read_csv("likes_model/meta.csv");ch=meta["channel_title"].values;gm=float(y.mean())
agg=pd.DataFrame({"c":ch,"y":y}).groupby("c")["y"].agg(["mean","count"])
agg["e"]=(agg["mean"]*agg["count"]+gm*10)/(agg["count"]+10);em=agg["e"].to_dict()
ce=np.array([em.get(c,gm) for c in ch],dtype=np.float32)
X=np.hstack([Xs,Xt,ce[:,None]]).astype(np.float32)
fn=json.load(open("likes_model/featnames.json"))+[f"text_svd_{i}" for i in range(Xt.shape[1])]+["channel_enc"]
p=dict(objective="regression",metric="rmse",n_estimators=800,learning_rate=0.05,num_leaves=47,
       subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.5,
       min_child_samples=20,random_state=42,n_jobs=1,verbose=-1)
m=lgb.LGBMRegressor(**p);m.fit(X,y)
imp=pd.Series(m.feature_importances_,index=fn).sort_values(ascending=False)
imp.to_csv("likes_model/feature_importance.csv",header=["importance"])
print(imp.head(15).to_string())
