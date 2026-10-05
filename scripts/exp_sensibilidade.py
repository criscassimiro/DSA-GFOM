"""Etapa 3: analise de sensibilidade (um hiperparametro por vez, demais no valor da etapa preliminar).
Janela 2022-01 a 2025-12 (mesma da etapa preliminar, N=192)."""
import json, time, os, sys
T0=time.time(); BUDGET=float(os.environ.get("BUDGET","160"))
import pandas as pd
from comum import *
GRADE={
 "RF":{"n_estimators":[50,100,300,500,1000],"max_depth":[2,3,5,8,12,None],"min_samples_leaf":[1,2,5,10],"max_features":[0.3,0.5,0.8,1.0]},
 "GB":{"n_estimators":[50,100,300,500,1000],"learning_rate":[0.01,0.03,0.05,0.1,0.2],"max_depth":[1,2,3,5,8],"subsample":[0.5,0.8,1.0]},
 "XGB":{"n_estimators":[50,100,300,500,1000],"learning_rate":[0.01,0.03,0.05,0.1,0.3],"max_depth":[1,2,3,6,8],"subsample":[0.5,0.8,1.0]},
}
BASE={"RF":PRELIMINAR["RF"],"GB":PRELIMINAR["GB"],"XGB":dict(n_estimators=100,learning_rate=0.3,max_depth=6,subsample=1.0)}
CSV="resultados/sensibilidade.csv"
linhas=pd.read_csv(CSV,keep_default_na=False).to_dict("records") if os.path.exists(CSV) else []
feitos={(l["modelo"],l["hiperparametro"],str(l["valor"])) for l in linhas}
for modelo,g in GRADE.items():
    for hp,vals in g.items():
        for v in vals:
            if (modelo,hp,str(v)) in feitos: continue
            if time.time()-T0>BUDGET: print("PAUSA"); sys.exit()
            p=dict(BASE[modelo]); p[hp]=v
            o=walk_forward(modelo,p,"2022-01-01","2025-12-01"); r=avaliar(o)
            lin=dict(modelo=modelo,hiperparametro=hp,valor=str(v))
            for var in r:
                for m in ["mae","rmse","r2","precision","recall","f1"]: lin[f"{m}_{var}"]=r[var][m]
            linhas.append(lin); pd.DataFrame(linhas).to_csv("resultados/sensibilidade.csv",index=False)
            print(modelo,hp,v,round(lin["mae_com_gatilho"],3),round(lin["mae_sem_gatilho"],3),flush=True)
print("FIM",flush=True)
