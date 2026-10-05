import json, glob, hashlib
import pandas as pd, numpy as np
from comum import metricas, LIMIAR_SEM_GATILHO
c=pd.read_csv("resultados/configuracoes_selecionadas.csv")
rows=[]; 
for _,r in c.iterrows():
    key=f"{r.modelo}_{hashlib.md5(r.params.encode()).hexdigest()[:8]}"
    o=pd.read_csv(f"resultados/pred/{key}.csv",parse_dates=["mes_dt"])
    col="pred_com_gatilho" if r.variante=="com_gatilho" else "pred_sem_gatilho"; lim=0.0 if r.variante=="com_gatilho" else LIMIAR_SEM_GATILHO
    for jan,ini in [("teste_2023_2025","2023-01-01"),("preliminar_2022_2025","2022-01-01")]:
        x=o[o.mes_dt>=ini]; m=metricas(x.gfom_hist_mwmed,x[col],lim)
        # media e dp por ano
        anual=[metricas(g.gfom_hist_mwmed,g[col],lim)["mae"] for _,g in x.groupby(x.mes_dt.dt.year)]
        rows.append(dict(variante=r.variante,modelo=r.modelo,metodo=r.metodo,janela=jan,mae_validacao=r.mae_validacao,**m,mae_anual_media=np.mean(anual),mae_anual_dp=np.std(anual,ddof=1),params=r.params))
# baseline zero
o=pd.read_csv(glob.glob("resultados/pred/*.csv")[0],parse_dates=["mes_dt"])
for jan,ini in [("teste_2023_2025","2023-01-01"),("preliminar_2022_2025","2022-01-01")]:
    x=o[o.mes_dt>=ini]; rows.append(dict(variante="-",modelo="Zero",metodo="ingenuo",janela=jan,**metricas(x.gfom_hist_mwmed,np.zeros(len(x)),0.0)))
t=pd.DataFrame(rows); t.to_csv("resultados/metricas_teste.csv",index=False)
pd.set_option("display.width",250)
print(t.drop(columns=["params"]).round(3).to_string())
