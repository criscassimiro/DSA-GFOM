import json, hashlib, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import estilo_br  # virgula decimal
from scipy.stats import spearmanr
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#e6e6e6","grid.linewidth":0.6,"axes.edgecolor":"#888","savefig.dpi":200,"legend.frameon":False})
CS={"Nordeste":"#2a78d6","Norte":"#eb6834","Sudeste":"#1baf7a","Sul":"#eda100"}
CM={"RF":"#2a78d6","GB":"#eb6834","XGB":"#1baf7a"}
MET={"referencia_preliminar":"Referência","grid":"Grade","random":"Aleatória","bayes":"Bayesiana"}
df=pd.read_csv("dados/base_modelagem.csv",parse_dates=["mes_dt"])
# 1 serie
fig,ax=plt.subplots(figsize=(7,2.8))
for s,c in CS.items():
    d=df[df.subsistema==s]; ax.plot(d.mes_dt,d.gfom_hist_mwmed,color=c,lw=1.6,label=s)
ax.set_ylabel("GFOM (MW médios)"); ax.legend(ncol=4,loc="upper right"); fig.tight_layout(); fig.savefig("figuras/fig01_serie_gfom.png")
# 2 EAR x CRef
fig,axs=plt.subplots(2,2,figsize=(7,4.8),sharex=True,sharey=True)
for ax,(s,c) in zip(axs.flat,CS.items()):
    d=df[df.subsistema==s]; ax.plot(d.mes_dt,d.ear_pct_sub,color=c,lw=1.6,label="EAR do subsistema (%)"); ax.plot(d.mes_dt,d.cref_amarela,color="#555",lw=1.1,ls="--",label="CRef amarela (%)")
    ev=d[d.gfom_hist_mwmed>0]; ax.scatter(ev.mes_dt,np.full(len(ev),3),marker="|",color="#e34948",s=40,label="mês com GFOM > 0")
    ax.set_title(s,fontsize=9,loc="left")
from matplotlib.lines import Line2D
h=[Line2D([],[],color="#333",lw=1.6,label="EAR do subsistema (%) - cor do painel"),Line2D([],[],color="#555",lw=1.1,ls="--",label="CRef amarela (%)"),Line2D([],[],color="#e34948",marker="|",ls="",ms=8,mew=1.5,label="mês com GFOM > 0")]
fig.legend(handles=h,ncol=3,loc="lower center",fontsize=8); fig.tight_layout(rect=(0,0.06,1,1)); fig.savefig("figuras/fig02_ear_cref.png")
# 3 zeros
g=df.groupby("subsistema").gfom_hist_mwmed.apply(lambda s:100*(s==0).mean()).sort_values()
fig,ax=plt.subplots(figsize=(5,2.4)); ax.barh(g.index,g.values,color="#2a78d6",height=0.55)
for i,v in enumerate(g.values): ax.text(v+1,i,f"{v:.1f}%".replace(".",","),va="center",fontsize=8)
ax.set_xlim(0,100); ax.set_xlabel("Meses com GFOM = 0 (%)"); ax.grid(axis="y",visible=False); fig.tight_layout(); fig.savefig("figuras/fig03_zeros.png")
# 4 esquema validacao
fig,ax=plt.subplots(figsize=(7,1.9)); ax.grid(False); ax.set_yticks([]); 
for sp in ["left"]: ax.spines[sp].set_visible(False)
import matplotlib.dates as mdates
d0,d1,d2,d3=[pd.Timestamp(x) for x in ["2021-01-01","2021-07-01","2023-01-01","2026-01-01"]]
ax.barh(2,(d1-d0).days,left=d0,color="#c9c9c9",height=0.5); ax.barh(2,(d2-d1).days,left=d1,color="#2a78d6",height=0.5); ax.barh(2,(d3-d2).days,left=d2,color="#eb6834",height=0.5)
ax.text(d0+(d1-d0)/2,2,"treino\ninicial",ha="center",va="center",fontsize=7); ax.text(d1+(d2-d1)/2,2,"seleção de hiperparâmetros\n(jul/2021–dez/2022)",ha="center",va="center",fontsize=7,color="white"); ax.text(d2+(d3-d2)/2,2,"teste fora da amostra (jan/2023–dez/2025)",ha="center",va="center",fontsize=7,color="white")
ax.barh(1,(pd.Timestamp("2026-01-01")-pd.Timestamp("2022-01-01")).days,left=pd.Timestamp("2022-01-01"),color="#1baf7a",height=0.4); ax.text(pd.Timestamp("2024-01-01"),1,"janela de comparação com os resultados preliminares (2022–2025)",ha="center",va="center",fontsize=7,color="white")
ax.set_ylim(0.5,2.6); ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); fig.tight_layout(); fig.savefig("figuras/fig04_esquema_validacao.png")
# 5 distribuicao MAE validacao (com gatilho)
b=pd.read_csv("resultados/busca_hiperparametros.csv"); b["met"]=b.metodo.str.replace("_com_gatilho","").str.replace("_sem_gatilho","")
for v,fn in [("com_gatilho","fig05_mae_validacao_com_gatilho.png"),("sem_gatilho","fig06_mae_validacao_sem_gatilho.png")]:
    x=b[~b.metodo.str.startswith("bayes")|b.metodo.str.endswith(v)]
    fig,axs=plt.subplots(1,3,figsize=(7,2.8),sharey=True)
    for ax,m in zip(axs,["RF","GB","XGB"]):
        for j,me in enumerate(["grid","random","bayes"]):
            y=x[(x.modelo==m)&(x.met==me)][f"mae_{v}"]; ax.scatter(np.full(len(y),j)+np.random.default_rng(1).uniform(-.15,.15,len(y)),y,s=10,color=CM[m],alpha=.6,lw=0)
            ax.plot([j-.25,j+.25],[y.min()]*2,color="#222",lw=1.2)
        r=x[(x.modelo==m)&(x.met=="referencia_preliminar")][f"mae_{v}"].iloc[0]; ax.axhline(r,color="#e34948",ls="--",lw=1)
        ax.set_xticks([0,1,2]); ax.set_xticklabels(["Grade","Aleatória","Bayesiana"]); ax.set_title(m,loc="left",fontsize=9); ax.grid(axis="x",visible=False)
    axs[0].set_ylabel("MAE na seleção (MW médios)"); fig.tight_layout(); fig.savefig("figuras/"+fn)
# 7 convergencia (melhor ate a iteracao) com gatilho
fig,axs=plt.subplots(1,3,figsize=(7,2.6),sharey=True)
ls={"grid":":","random":"--","bayes":"-"}
for ax,m in zip(axs,["RF","GB","XGB"]):
    for me in ["grid","random","bayes_com_gatilho"]:
        y=b[(b.modelo==m)&(b.metodo==me)].sort_values("iter").mae_com_gatilho.cummin().values
        ax.plot(range(1,len(y)+1),y,color=CM[m],ls=ls[me.split("_")[0]],lw=1.6,label=MET[me.split("_")[0]])
    r=b[(b.modelo==m)&(b.metodo=="referencia_preliminar")].mae_com_gatilho.iloc[0]; ax.axhline(r,color="#e34948",lw=1,ls="--",label="Referência")
    ax.set_title(m,loc="left",fontsize=9); ax.set_xlabel("Configurações avaliadas")
axs[0].set_ylabel("Melhor MAE até a\niteração (MW médios)"); axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig("figuras/fig07_convergencia.png")
# 8 validacao x teste
t=pd.read_csv("resultados/metricas_teste.csv"); tt=t[(t.variante=="com_gatilho")&(t.janela=="teste_2023_2025")]
mk={"referencia_preliminar":"o","grid":"s","random":"^","bayes":"D"}
fig,ax=plt.subplots(figsize=(6,3.4))
for _,r in tt.iterrows(): ax.scatter(r.mae_validacao,r.mae,color=CM[r.modelo],marker=mk[r.metodo],s=40,edgecolor="white",lw=.8)
z=t[(t.modelo=="Zero")&(t.janela=="teste_2023_2025")].mae.iloc[0]; ax.axhline(z,color="#555",ls=":",lw=1); ax.text(tt.mae_validacao.min(),z+0.01,"modelo ingênuo (zero)",fontsize=7,va="bottom")
from matplotlib.lines import Line2D
hd=[Line2D([],[],marker="o",ls="",color=c,label=m) for m,c in CM.items()]+[Line2D([],[],marker=mk[k],ls="",color="#777",label=MET[k]) for k in mk]
ax.legend(handles=hd,fontsize=7,ncol=1,loc="upper left",bbox_to_anchor=(1.01,1)); ax.set_xlabel("MAE na seleção, jul/2021–dez/2022 (MW médios)"); ax.set_ylabel("MAE no teste, 2023–2025 (MW médios)")
rho,p=spearmanr(tt.mae_validacao,tt.mae); print("spearman val x teste",rho,p)
fig.tight_layout(); fig.savefig("figuras/fig08_validacao_vs_teste.png")
# 9 sensibilidade
s=pd.read_csv("resultados/sensibilidade.csv",keep_default_na=False)
hps={"RF":["n_estimators","max_depth","min_samples_leaf","max_features"],"GB":["n_estimators","learning_rate","max_depth","subsample"],"XGB":["n_estimators","learning_rate","max_depth","subsample"]}
fig,axs=plt.subplots(3,4,figsize=(7.2,6),sharey=True)
for i,m in enumerate(["RF","GB","XGB"]):
    for j,hp in enumerate(hps[m]):
        ax=axs[i,j]; x=s[(s.modelo==m)&(s.hiperparametro==hp)]
        ax.plot(range(len(x)),x.mae_com_gatilho,color=CM[m],marker="o",ms=4,lw=1.6)
        ax.set_xticks(range(len(x))); ax.set_xticklabels([v if v!="None" else "∞" for v in x.valor],fontsize=7,rotation=0)
        ax.set_title(f"{m}: {hp}",fontsize=8,loc="left")
    axs[i,0].set_ylabel("MAE 2022–2025\n(MW médios)")
fig.tight_layout(); fig.savefig("figuras/fig09_sensibilidade.png")
print(s.groupby("modelo").mae_com_gatilho.agg(["min","max"]))
# 10 meses de gatilho
g=pd.read_csv("resultados/meses_gatilho_previsoes.csv")
lab=[f"{pd.Timestamp(a):%m/%Y}\n{b_}" for a,b_ in zip(g.mes_dt,g.subsistema)]
cols=[("gfom_hist_mwmed","Observado","#444"),("RF_referencia_preliminar","RF referência","#2a78d6"),("RF_random","RF aleatória","#9bc1ef"),("XGB_random","XGB aleatória","#1baf7a")]
fig,ax=plt.subplots(figsize=(7,2.8)); w=0.2
for k,(c,n,col) in enumerate(cols): ax.bar(np.arange(len(g))+(k-1.5)*w,g[c],width=w-0.02,color=col,label=n)
ax.set_xticks(range(len(g))); ax.set_xticklabels(lab,fontsize=7); ax.set_ylabel("GFOM (MW médios)"); ax.legend(fontsize=7,ncol=4); ax.grid(axis="x",visible=False); fig.tight_layout(); fig.savefig("figuras/fig10_meses_gatilho.png")
