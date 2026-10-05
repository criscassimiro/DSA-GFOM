import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score
import base_preliminar as b

FEATURES=["ear_pct_sub","cref_amarela","gap_below","intensidade","gt_hist_mwmed","mes_num","mes_sin","mes_cos"]
from pathlib import Path
DF=pd.read_csv(Path(__file__).resolve().parent.parent/"dados"/"base_modelagem.csv",parse_dates=["mes_dt"]).sort_values(["mes_dt","subsistema"]).reset_index(drop=True)
MESES=sorted(DF.mes_dt.unique())
LIMIAR_SEM_GATILHO=1.0  # MWmed: limiar fixo a priori para classificar ocorrencia no modelo sem gatilho

PRELIMINAR={
 "RF":dict(n_estimators=500,max_depth=8,min_samples_leaf=2,min_samples_split=4),
 "GB":dict(n_estimators=300,learning_rate=0.05,max_depth=3,min_samples_leaf=2,min_samples_split=4,subsample=1.0),
 "XGB":dict(),  # padrao da biblioteca (sem configuracao na etapa preliminar)
}

def make(modelo,params):
    p=dict(params)
    if modelo=="RF": return RandomForestRegressor(random_state=42,n_jobs=4,**p)
    if modelo=="GB": return GradientBoostingRegressor(random_state=42,**p)
    if modelo=="XGB": return XGBRegressor(random_state=42,n_jobs=4,objective="reg:squarederror",verbosity=0,**p)

def walk_forward(modelo,params,inicio,fim):
    out=[]
    for m in MESES:
        if m<pd.Timestamp(inicio) or m>pd.Timestamp(fim): continue
        tr=DF[DF.mes_dt<m]; te=DF[DF.mes_dt==m]
        mdl=make(modelo,params); mdl.fit(tr[FEATURES],tr["share_hist"])
        o=te[["mes_dt","subsistema","gfom_hist_mwmed","gt_hist_mwmed","gatilho"]].copy()
        o["share_pred"]=np.clip(mdl.predict(te[FEATURES]),0,1)
        out.append(o)
    o=pd.concat(out,ignore_index=True)
    o["pred_sem_gatilho"]=o.share_pred*o.gt_hist_mwmed
    o["pred_com_gatilho"]=np.where(o.gatilho==1,o.pred_sem_gatilho,0.0)
    return o

def metricas(y,p,limiar):
    y=np.asarray(y,float); p=np.asarray(p,float)
    yb=(y>0).astype(int); pb=(p>limiar).astype(int)
    return dict(n=len(y),eventos=int(yb.sum()),mae=mean_absolute_error(y,p),rmse=float(np.sqrt(mean_squared_error(y,p))),
        r2=r2_score(y,p) if len(np.unique(y))>1 else np.nan, smape=b.smape(pd.Series(y),pd.Series(p)),
        precision=precision_score(yb,pb,zero_division=0),recall=recall_score(yb,pb,zero_division=0),f1=f1_score(yb,pb,zero_division=0),
        tp=int(((yb==1)&(pb==1)).sum()),fp=int(((yb==0)&(pb==1)).sum()),fn=int(((yb==1)&(pb==0)).sum()))

def avaliar(o,agregado_sin=False):
    """retorna metricas com e sem gatilho"""
    r={}
    r["com_gatilho"]=metricas(o.gfom_hist_mwmed,o.pred_com_gatilho,0.0)
    r["sem_gatilho"]=metricas(o.gfom_hist_mwmed,o.pred_sem_gatilho,LIMIAR_SEM_GATILHO)
    return r
