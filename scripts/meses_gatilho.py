import pandas as pd, hashlib
c=pd.read_csv("resultados/configuracoes_selecionadas.csv"); c=c[c.variante=="com_gatilho"]
out=None
for _,r in c.iterrows():
    o=pd.read_csv(f"resultados/pred/{r.modelo}_{hashlib.md5(r.params.encode()).hexdigest()[:8]}.csv")
    o=o[o.gatilho==1][["mes_dt","subsistema","gfom_hist_mwmed","pred_com_gatilho"]].rename(columns={"pred_com_gatilho":f"{r.modelo}_{r.metodo}"})
    out=o if out is None else out.merge(o,on=["mes_dt","subsistema","gfom_hist_mwmed"])
out.round(1).to_csv("resultados/meses_gatilho_previsoes.csv",index=False); print(out.round(0).T.to_string())
