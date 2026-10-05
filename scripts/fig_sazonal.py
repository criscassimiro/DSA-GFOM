import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import estilo_br  # virgula decimal
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#e6e6e6","savefig.dpi":200,"legend.frameon":False})
CS={"Nordeste":"#2a78d6","Norte":"#eb6834","Sudeste":"#1baf7a","Sul":"#eda100"}
df=pd.read_csv("dados/base_modelagem.csv",parse_dates=["mes_dt"]); df["m"]=df.mes_dt.dt.month
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8),sharex=True)
for s,c in CS.items():
    d=df[df.subsistema==s]
    a=d.groupby("m").gfom_hist_mwmed.mean(); axs[0].plot(a.index,a.values,color=c,lw=1.6,marker="o",ms=3,label=s)
    f=d.groupby("m").gfom_hist_mwmed.apply(lambda x:(x>0).sum()); axs[1].plot(f.index,f.values,color=c,lw=1.6,marker="o",ms=3,label=s)
axs[0].set_ylabel("GFOM média (MW médios)"); axs[1].set_ylabel("Nº de meses com GFOM > 0")
for ax in axs: ax.set_xticks(range(1,13)); ax.set_xticklabels(list("JFMAMJJASOND"))
axs[1].set_yticks([0,1,2]); axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig("figuras/fig12_sazonalidade.png")
t=df.groupby(["subsistema","m"]).gfom_hist_mwmed.agg(["mean",lambda x:(x>0).sum()]).round(1); t.to_csv("resultados/sazonalidade.csv"); print(t.unstack(0).to_string())
