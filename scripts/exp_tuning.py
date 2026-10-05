"""Etapa 1: busca de hiperparametros (grid, random, bayesiana/TPE) com validacao walk-forward.
Janela de selecao: 2021-07 a 2022-12. Criterio: MAE (MWmed)."""
import itertools, json, time, sys, os
T0=time.time(); BUDGET=float(os.environ.get("BUDGET","150"))
class Pare(Exception): pass
import numpy as np, pandas as pd, optuna
from comum import *
optuna.logging.set_verbosity(optuna.logging.WARNING)
VAL_INI,VAL_FIM="2021-07-01","2022-12-01"
ORC=36  # orcamento de configuracoes por metodo
rng=np.random.default_rng(42)

GRID={
 "RF":dict(n_estimators=[100,500],max_depth=[3,8,None],min_samples_leaf=[1,2,5],max_features=[1.0,0.5]),
 "GB":dict(n_estimators=[100,300],learning_rate=[0.01,0.05,0.1],max_depth=[2,3,5],subsample=[0.8,1.0]),
 "XGB":dict(n_estimators=[100,300],learning_rate=[0.01,0.05,0.1],max_depth=[2,3,5],subsample=[0.8,1.0]),
}
def amostra(modelo,trial=None):
    s=(lambda n,a,b_,log=False: trial.suggest_float(n,a,b_,log=log)) if trial else (lambda n,a,b_,log=False: float(np.exp(rng.uniform(np.log(a),np.log(b_))) if log else rng.uniform(a,b_)))
    i=(lambda n,a,b_: trial.suggest_int(n,a,b_)) if trial else (lambda n,a,b_: int(rng.integers(a,b_+1)))
    if modelo=="RF":
        md=i("max_depth",2,16)
        return dict(n_estimators=i("n_estimators",50,1000),max_depth=md,min_samples_leaf=i("min_samples_leaf",1,10),max_features=round(s("max_features",0.2,1.0),3))
    p=dict(n_estimators=i("n_estimators",50,1000),learning_rate=round(s("learning_rate",0.005,0.3,log=True),4),max_depth=i("max_depth",1,8),subsample=round(s("subsample",0.5,1.0),3))
    if modelo=="GB": p["min_samples_leaf"]=i("min_samples_leaf",1,10)
    else: p["min_child_weight"]=round(s("min_child_weight",0.5,10,log=True),3); p["reg_lambda"]=round(s("reg_lambda",0.01,10,log=True),3)
    return p

CSV="resultados/busca_hiperparametros.csv"
linhas=pd.read_csv(CSV).to_dict("records") if os.path.exists(CSV) else []
feitos={(l["modelo"],l["metodo"],int(l["iter"])) for l in linhas}
def checa():
    if time.time()-T0>BUDGET: raise Pare()
def roda(modelo,metodo,params,k):
    checa()
    t=time.time(); o=walk_forward(modelo,params,VAL_INI,VAL_FIM); r=avaliar(o)
    lin=dict(modelo=modelo,metodo=metodo,iter=k,params=json.dumps(params),seg=round(time.time()-t,1))
    for v in r:
        for m in ["mae","rmse","r2","precision","recall","f1"]: lin[f"{m}_{v}"]=r[v][m]
    linhas.append(lin); pd.DataFrame(linhas).to_csv("resultados/busca_hiperparametros.csv",index=False)
    print(modelo,metodo,k,lin["mae_com_gatilho"],lin["mae_sem_gatilho"],flush=True)
    return lin

MODELOS=sys.argv[1:] or ["RF","GB","XGB"]
try:
  for modelo in MODELOS:
    rng=np.random.default_rng(42)
    rand=[amostra(modelo) for _ in range(ORC)]
    if (modelo,"referencia_preliminar",0) not in feitos: roda(modelo,"referencia_preliminar",PRELIMINAR[modelo],0)
    g=GRID[modelo]; combos=[dict(zip(g,c)) for c in itertools.product(*g.values())]
    for k,p in enumerate(combos,1):
        if (modelo,"grid",k) not in feitos: roda(modelo,"grid",p,k)
    for k,p in enumerate(rand,1):
        if (modelo,"random",k) not in feitos: roda(modelo,"random",p,k)
    for alvo in ["com_gatilho","sem_gatilho"]:
        st=optuna.create_study(study_name=f"{modelo}_{alvo}",storage="sqlite:///"+os.path.expanduser("~/optuna_gfom_v2.db")+"",load_if_exists=True,direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
        done=len([t for t in st.trials if t.state==optuna.trial.TrialState.COMPLETE])
        def obj(trial):
            p=amostra(modelo,trial); return roda(modelo,f"bayes_{alvo}",p,trial.number+1)[f"mae_{alvo}"]
        if done<ORC: st.optimize(obj,n_trials=ORC-done,catch=())
  print("FIM",flush=True)
except Pare:
  print("PAUSA",len(linhas),flush=True)
