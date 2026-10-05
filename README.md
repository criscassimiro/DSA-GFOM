# DSA-GFOM — Modelo de Machine Learning para Previsão de GFOM no SIN

Código e resultados do TCC do MBA em Data Science e Analytics (USP/Esalq):
**"Modelo de Machine Learning para Previsão de GFOM no Sistema Interligado Nacional"** — Cristiane Aparecida Cassimiro de Araujo, 2026.

Dá continuidade à etapa de resultados preliminares do TCC, na qual foi desenvolvido um modelo híbrido
(gatilho EAR ≤ CRef + regressor de comitê), e compara:

- **Algoritmos:** Random Forest, Gradient Boosting (scikit-learn) e XGBoost
- **Estratégias de busca:** grade, aleatória e bayesiana (Optuna/TPE), 36 configurações cada, contra a configuração fixa dos resultados preliminares
- **Validação:** origem móvel, janela expansiva, reestimação mensal; seleção em jul/2021–dez/2022, teste em 2023–2025

## Correção da base

No código da etapa preliminar, a GFOM mensal era a média das linhas usina × hora (GFOM **média por usina**).
Aqui ela é o **total do subsistema**: soma das usinas em cada hora, depois média das horas do mês
(`scripts/prep_corrigido.py`). A ocorrência (GFOM > 0) e o gatilho não mudam; a escala dos erros, sim.

## Estrutura

```
dados/        base_modelagem.csv (corrigida) e base_modelagem_preliminar.csv (agregação da etapa preliminar), 240 obs. mensais 2021–2025
scripts/      código (executar a partir da raiz do repositório)
resultados/   saídas numéricas (CSV)
figuras/      figuras do TCC
```

## Como reproduzir

```bash
pip install -r requirements.txt
python scripts/t_base.py            # reproduz o RF de referência
python scripts/exp_tuning.py        # busca de hiperparâmetros (retomável; BUDGET=segundos por execução)
python scripts/exp_teste.py         # walk-forward 2022–2025 das configurações selecionadas
python scripts/resumo_teste.py      # métricas por janela -> resultados/metricas_teste.csv
python scripts/exp_sensibilidade.py # sensibilidade um-a-um
python scripts/meses_gatilho.py
python scripts/importancia.py       # REFAZER=1 recalcula as importâncias
python scripts/figuras.py; python scripts/fig_sazonal.py   # estilo_br.py: vírgula decimal nos eixos
```

`resumo_teste.py`, `exp_teste.py` e `figuras.py` leem/escrevem em `resultados/` e `figuras/` relativos à raiz.
A reconstrução da base a partir dos dados brutos do ONS (`scripts/prep.py` e `scripts/prep_corrigido.py`)
exige os arquivos brutos (não versionados) em uma pasta indicada por `GFOM_DADOS_BRUTOS`.

Versões usadas: Python 3.13, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1, optuna 5.0.0. Semente: 42.

## Fonte dos dados

ONS — Dados Abertos (https://dados.ons.org.br): geração térmica por motivo de despacho (garantia energética),
geração por fonte e energia armazenada; CRef: NT-ONS DPL 0156/2021.
