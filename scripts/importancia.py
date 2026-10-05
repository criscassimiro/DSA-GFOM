import json, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import estilo_br  # virgula decimal
from sklearn.inspection import permutation_importance
from comum import *
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":200})
CM={"RF":"#2a78d6","GB":"#eb6834","XGB":"#1baf7a"}
c=pd.read_csv("resultados/configuracoes_selecionadas.csv"); c=c[c.variante=="com_gatilho"]
X,y=DF[FEATURES],DF.share_hist
import os
if os.path.exists("resultados/importancia_variaveis.csv") and not os.environ.get("REFAZER"):
    t=pd.read_csv("resultados/importancia_variaveis.csv")  # REFAZER=1 recalcula
else:
    rows=[]
    for _,r in c[c.metodo.isin(["referencia_preliminar","random"])].iterrows():
        m=make(r.modelo,json.loads(r.params)).fit(X,y)
        pi=permutation_importance(m,X,y,n_repeats=30,random_state=42,scoring="neg_mean_absolute_error")
        for f,a,b_,s in zip(FEATURES,m.feature_importances_,pi.importances_mean,pi.importances_std): rows.append(dict(modelo=r.modelo,metodo=r.metodo,variavel=f,imp_impureza=a,imp_permutacao=b_,imp_perm_dp=s))
    t=pd.DataFrame(rows); t.to_csv("resultados/importancia_variaveis.csv",index=False)
p=t[t.metodo=="referencia_preliminar"].pivot(index="variavel",columns="modelo",values="imp_impureza").loc[FEATURES]
print(p.round(3)); print(t[t.metodo=="random"].pivot(index="variavel",columns="modelo",values="imp_impureza").round(3))
p=p.sort_values("RF")
fig,ax=plt.subplots(figsize=(6,3.2)); w=0.27
for k,m in enumerate(["RF","GB","XGB"]): ax.barh(np.arange(len(p))+(k-1)*w,p[m],height=w-0.03,color=CM[m],label=m)
ax.set_yticks(range(len(p))); ax.set_yticklabels(p.index); ax.set_xlabel("Importância relativa (redução de impureza)"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig("figuras/fig11_importancia.png")
