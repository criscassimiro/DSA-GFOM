"""Etapa 2: avaliacao fora da amostra das configuracoes selecionadas.
Walk-forward mensal de 2022-01 a 2025-12; metricas em 2023-2025 (teste puro, nao usado na selecao)
e em 2022-2025 (janela da etapa preliminar; 2022 sobrepoe a janela de selecao)."""
import json, os, sys, time
import pandas as pd, numpy as np
from comum import *
T0=time.time(); BUDGET=float(os.environ.get("BUDGET","160"))
d=pd.read_csv("resultados/busca_hiperparametros.csv")
d["met"]=d.metodo.str.replace("_com_gatilho","").str.replace("_sem_gatilho","")
confs=[]
for v in ["com_gatilho","sem_gatilho"]:
    x=d[~d.metodo.str.startswith("bayes") | d.metodo.str.endswith(v)]
    for (mod,met),g in x.groupby(["modelo","met"]):
        r=g.loc[g[f"mae_{v}"].idxmin()]
        confs.append(dict(variante=v,modelo=mod,metodo=met,params=r.params,mae_validacao=r[f"mae_{v}"]))
confs=pd.DataFrame(confs); confs.to_csv("resultados/configuracoes_selecionadas.csv",index=False)
os.makedirs("resultados/pred",exist_ok=True)
for _,c in confs.drop_duplicates(["modelo","params"]).iterrows():
    key=f"{c.modelo}_{__import__("hashlib").md5(c.params.encode()).hexdigest()[:8]}"
    f=f"resultados/pred/{key}.csv"
    if os.path.exists(f): continue
    if time.time()-T0>BUDGET: print("PAUSA"); sys.exit()
    o=walk_forward(c.modelo,json.loads(c.params),"2022-01-01","2025-12-01"); o["params"]=c.params; o["modelo"]=c.modelo
    o.to_csv(f,index=False); print("ok",key,flush=True)
print("FIM")
