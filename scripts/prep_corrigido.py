"""Base corrigida: GFOM total do subsistema = soma das usinas em cada hora, media das horas do mes.
(Na etapa preliminar, a GFOM foi agregada pela media das linhas usina x hora, i.e., media por usina.)"""
import pandas as pd, numpy as np
from pathlib import Path
import os
D=Path(os.environ.get("GFOM_DADOS_BRUTOS","dados_brutos"))
b=pd.read_csv("dados/base_modelagem_preliminar.csv",parse_dates=["mes_dt"])
df=pd.read_csv(D/"gt_historico.csv",sep=";",low_memory=False,usecols=["data_hora","subsistema","val_verifgarantiaenergetica"])
df["data_hora"]=pd.to_datetime(df.data_hora,utc=True).dt.tz_localize(None)
df["v"]=pd.to_numeric(df.val_verifgarantiaenergetica,errors="coerce")
h=df.groupby(["subsistema","data_hora"],as_index=False).v.sum()
h["mes_dt"]=h.data_hora.dt.to_period("M").dt.to_timestamp()
m=h.groupby(["mes_dt","subsistema"],as_index=False).v.mean().rename(columns={"v":"gfom_total"})
b=b.merge(m,on=["mes_dt","subsistema"],how="left")
b["gfom_media_por_usina"]=b.gfom_hist_mwmed
b["gfom_hist_mwmed"]=b.gfom_total; b=b.drop(columns="gfom_total")
b["share_hist"]=(b.gfom_hist_mwmed/b.gt_hist_mwmed.replace(0,np.nan)).fillna(0).clip(0,1)
print("max razao gfom/gt:",(b.gfom_hist_mwmed/b.gt_hist_mwmed).max())
print("ocorrencia igual:",((b.gfom_hist_mwmed>0)==(b.gfom_media_por_usina>0)).all())
print(b.groupby("subsistema")[["gfom_hist_mwmed","gfom_media_por_usina"]].agg(["mean","max"]).round(1))
b.to_csv("dados/base_modelagem.csv",index=False)
