import sys, pandas as pd
from pathlib import Path
from dataclasses import replace
import base_preliminar as b
import os
D=Path(os.environ.get("GFOM_DADOS_BRUTOS","dados_brutos"))
cfg=b.Config(base_dir=D,gt_historico_csv=D/"gt_historico.csv",geracao_historico_csv=D/"geracao_historico.csv",ear_historico_csv=D/"ear_historico.csv",ear_2026_csv=D/"ear.csv",gt_2026_csv=D/"ute.csv",cref_xlsx=D/"cref_2021_2026.xlsx",out_dir=Path("out"))
tr=b.build_train(cfg)
tr.to_csv("dados/base_modelagem_preliminar.csv",index=False)
print(tr.shape, tr.mes_dt.min(), tr.mes_dt.max())
tr["ano"]=pd.to_datetime(tr.mes_dt).dt.year
tr["ev"]=tr.gfom_hist_mwmed>0
print(tr.groupby("ano").agg(n=("ev","size"),eventos=("ev","sum"),gatilho=("gatilho","sum")))
print(pd.crosstab(tr.ev,tr.gatilho))
print(tr[tr.ev & (tr.ano>=2022)][["mes_dt","subsistema","gfom_hist_mwmed","gatilho","gap_below"]].to_string())
