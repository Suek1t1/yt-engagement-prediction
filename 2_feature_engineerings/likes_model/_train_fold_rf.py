import sys,numpy as np,pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
fold=int(sys.argv[1]); RS=42; N=5
Xs=np.load("likes_model/Xs.npy");Xt=np.load("likes_model/Xt.npy");y=np.load("likes_model/y.npy")
meta=pd.read_csv("likes_model/meta.csv");channel=meta["channel_title"].values;gmean=float(y.mean())
kf=KFold(n_splits=N,shuffle=True,random_state=RS);tr,va=list(kf.split(Xs))[fold]
tdf=pd.DataFrame({"c":channel[tr],"y":y[tr]});agg=tdf.groupby("c")["y"].agg(["mean","count"])
agg["e"]=(agg["mean"]*agg["count"]+gmean*10)/(agg["count"]+10);em=agg["e"].to_dict()
ce=np.array([em.get(c,gmean) for c in channel],dtype=np.float32)
Xtr=np.hstack([Xs[tr],Xt[tr][:,:40],ce[tr,None]]).astype(np.float32)
Xva=np.hstack([Xs[va],Xt[va][:,:40],ce[va,None]]).astype(np.float32)
m=RandomForestRegressor(n_estimators=150,max_features=0.33,min_samples_leaf=3,max_depth=20,
                        n_jobs=1,random_state=RS)
m.fit(Xtr,y[tr]);pred=m.predict(Xva)
np.save(f"likes_model/rf_idx_{fold}.npy",va);np.save(f"likes_model/rf_pred_{fold}.npy",pred)
print(f"rf fold {fold} done",flush=True)
