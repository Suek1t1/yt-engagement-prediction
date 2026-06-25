import sys,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.model_selection import KFold
fold=int(sys.argv[1]); RS=42; N=5
Xs=np.load("likes_model/Xs.npy");Xt=np.load("likes_model/Xt.npy");y=np.load("likes_model/y.npy")
meta=pd.read_csv("likes_model/meta.csv");channel=meta["channel_title"].values;gmean=float(y.mean())
kf=KFold(n_splits=N,shuffle=True,random_state=RS);splits=list(kf.split(Xs));tr,va=splits[fold]
tdf=pd.DataFrame({"c":channel[tr],"y":y[tr]});agg=tdf.groupby("c")["y"].agg(["mean","count"])
agg["e"]=(agg["mean"]*agg["count"]+gmean*10)/(agg["count"]+10);em=agg["e"].to_dict()
ce=np.array([em.get(c,gmean) for c in channel],dtype=np.float32)
Xtr=np.hstack([Xs[tr],Xt[tr],ce[tr,None]]).astype(np.float32)
Xva=np.hstack([Xs[va],Xt[va],ce[va,None]]).astype(np.float32)
p=dict(objective="regression",metric="rmse",n_estimators=1500,learning_rate=0.05,num_leaves=47,
       subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.5,
       min_child_samples=20,random_state=RS,n_jobs=1,verbose=-1)
m=lgb.LGBMRegressor(**p);m.fit(Xtr,y[tr],eval_set=[(Xva,y[va])],callbacks=[lgb.early_stopping(100,verbose=False)])
pred=m.predict(Xva,num_iteration=m.best_iteration_)
np.save(f"likes_model/oof_idx_{fold}.npy",va);np.save(f"likes_model/oof_pred_{fold}.npy",pred)
print(f"fold {fold} best_iter={m.best_iteration_}",flush=True)
