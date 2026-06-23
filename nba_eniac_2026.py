# -*- coding: utf-8 -*-
"""NBA-ENIAC-2026.ipynb


## **Previsão de Campeão e Finalistas da NBA (2000–2025)**

**Objetivo**: Avaliar diferentes algoritmos de classificação supervisionada para identificar quais apresentam melhor desempenho na previsão dos finalistas e do campeão da NBA, utilizando estatísticas da temporada regular. Após a etapa de comparação e validação, o modelo com melhor resultado é utilizado para gerar as previsões da temporada de 2026.

**Targets**:

- reached_finals → 1 se o time chegou às Finais da NBA (2 times por temporada)
- champion → 1 se o time foi campeão da NBA (1 time por temporada)

**Dataset**: NBA Team Summaries — Kaggle (sumitrodatta/nba-aba-baa-stats)

### **Configuração**

Instalar bibliotecas necessárias:
"""

!pip install basketball-reference-scraper requests beautifulsoup4 lxml -q

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import json
import joblib
import warnings
import random, os
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore')
plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")

"""Configurar credenciais do Kaggle para baixar o dataset:"""

# Configurar token do Kaggle
os.environ["KAGGLE_API_TOKEN"] = '{"username":"joaovitorarroyo","key":"KGAT_c09328b129391d106fc75988fca39a94"}'

# Configurar para o kaggle CLI encontrar as credenciais
token = json.loads(os.environ["KAGGLE_API_TOKEN"])
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump(token, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

print("Token configurado!")

"""### **Carregamento do dataset e pré-processamento dos dados**"""

# Baixar dataset NBA
!kaggle datasets download -d sumitrodatta/nba-aba-baa-stats -p nba_data/ --unzip -q
import glob

print("Dataset baixado!")

# Carregar CSV de Team Summaries

df = pd.read_csv("nba_data/Team Summaries.csv")

stats = df[
    (df["season"] >= 2000) &
    (df["season"] <= 2025) &
    (df["lg"] == "NBA")
].copy()

stats = stats[stats["team"] != "League Average"].copy()

print(f"{len(stats)} exemplos de {stats['season'].nunique()} temporadas")
print(f"{len(stats.columns.tolist())} atributos disponíveis: {stats.columns.tolist()}")

"""Criar os atributos alvo: **reached_finals** e **champion**"""

# Criar duas colunas novas no dataset:
#   reached_finals = 1 se o time chegou às finais da NBA
#   champion       = 1 se o time foi campeão
# São essas são as perguntas que nossos modelos vai responder.

champions_data = {
    2000: ("Los Angeles Lakers",    "Indiana Pacers"), # isso quer dizer: 2000: los angeles lakers foi o campeão, Indiana Pacers foi o segundo colocado. No dataset o lakers vai ficar com 1 em "champion" E "reached_finals" e o pacers só com 1 em "reached_finals"
    2001: ("Los Angeles Lakers",    "Philadelphia 76ers"),
    2002: ("Los Angeles Lakers",    "New Jersey Nets"),
    2003: ("San Antonio Spurs",     "New Jersey Nets"),
    2004: ("Detroit Pistons",       "Los Angeles Lakers"),
    2005: ("San Antonio Spurs",     "Detroit Pistons"),
    2006: ("Miami Heat",            "Dallas Mavericks"),
    2007: ("San Antonio Spurs",     "Cleveland Cavaliers"),
    2008: ("Boston Celtics",        "Los Angeles Lakers"),
    2009: ("Los Angeles Lakers",    "Orlando Magic"),
    2010: ("Los Angeles Lakers",    "Boston Celtics"),
    2011: ("Dallas Mavericks",      "Miami Heat"),
    2012: ("Miami Heat",            "Oklahoma City Thunder"),
    2013: ("Miami Heat",            "San Antonio Spurs"),
    2014: ("San Antonio Spurs",     "Miami Heat"),
    2015: ("Golden State Warriors", "Cleveland Cavaliers"),
    2016: ("Cleveland Cavaliers",   "Golden State Warriors"),
    2017: ("Golden State Warriors", "Cleveland Cavaliers"),
    2018: ("Golden State Warriors", "Cleveland Cavaliers"),
    2019: ("Toronto Raptors",       "Golden State Warriors"),
    2020: ("Los Angeles Lakers",    "Miami Heat"),
    2021: ("Milwaukee Bucks",       "Phoenix Suns"),
    2022: ("Golden State Warriors", "Boston Celtics"),
    2023: ("Denver Nuggets",        "Miami Heat"),
    2024: ("Boston Celtics",        "Dallas Mavericks"),
    2025: ("Oklahoma City Thunder", "Indiana Pacers"),
}

# Dataframe das finais
playoffs_df = pd.DataFrame([
    {"season": s, "champion": v[0], "runner_up": v[1]}
    for s, v in champions_data.items()
])

# Alguns times mudaram de nome ou cidade ao longo dos anos
# normalizar para garantir que o merge funcione corretamente
name_map = {
    "New Jersey Nets":                   "Brooklyn Nets",
    "Seattle SuperSonics":               "Oklahoma City Thunder",
    "New Orleans Hornets":               "New Orleans Pelicans",
    "Charlotte Bobcats":                 "Charlotte Hornets",
    "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
}
def normalize(name):
    name = str(name).replace("*", "").strip()
    return name_map.get(name, name)

stats["team_norm"]       = stats["team"].apply(normalize)
playoffs_df["champion"]  = playoffs_df["champion"].apply(normalize)
playoffs_df["runner_up"] = playoffs_df["runner_up"].apply(normalize)

# Merge e criação dos targets
stats = stats.merge(playoffs_df, on="season", how="left")

# É champion OU runner_up
stats["reached_finals"] = (
    (stats["team_norm"] == stats["champion"]) |
    (stats["team_norm"] == stats["runner_up"])
).astype(int)

# É champion
stats["champion"] = (stats["team_norm"] == stats["champion"]).astype(int)

stats.drop(columns=["runner_up"], inplace=True)

# Verificação de integridade
print(f"Campeões no dataset  : {stats['champion'].sum()} (esperado: 26)")
print(f"Finalistas no dataset: {stats['reached_finals'].sum()} (esperado: 52)")
check = stats.groupby("season")["champion"].sum()
print(f"Temporadas sem campeão definido: {(check != 1).sum()} (esperado: 0)")

print("Verificação - Últimos 10 finalistas e quem foi campeão:\n")
print(stats[stats["reached_finals"] == 1][["season", "team_norm", "champion"]].head(10).sort_values("season"))

"""Padronizar nomes das colunas:"""

# Padronizar os nomes das colunas para facilitar a leitura do código nos blocos seguintes.

stats.rename(columns={
    "season":       "Season",
    "team":         "Team",
    "team_norm":    "Team_norm",
    "w":            "W",
    "l":            "L",
    "pw":           "W/L%",
    "srs":          "SRS",
    "o_rtg":        "ORtg",
    "d_rtg":        "DRtg",
    "n_rtg":        "NRtg",
    "pace":         "Pace",
    "e_fg_percent": "eFG%",
    "tov_percent":  "TOV%",
    "orb_percent":  "ORB%",
    "f_tr":      "FT/FGA",
}, inplace=True)

stats.to_csv("nba_raw_stats.csv", index=False)

print(f"{stats.shape[0]} exemplos e {stats.shape[1]} atributos\n")
print(stats[["Season","Team","W","L","SRS","NRtg","champion","reached_finals", "FT/FGA"]].head(8).to_string())

"""### **Análise e seleção de atributos**

Criar features derivadas:
"""

# Criar novas features derivadas das existentes.
# Isso porque o modelo aprende melhor com features que capturam o
# desempenho relativo de cada time dentro da sua temporada:
#   • NRtg_zscore   → quão bom é o Net Rating do time comparado aos outros times DAQUELA temporada.
#   • WinPct_zscore → mesmo conceito para o aproveitamento (W/L%)

stats = pd.read_csv("nba_raw_stats.csv")

if "NRtg" not in stats.columns:
    stats["NRtg"] = stats["ORtg"] - stats["DRtg"]

stats["NRtg_zscore"] = stats.groupby("Season")["NRtg"].transform(
    lambda x: (x - x.mean()) / x.std()
)
stats["WinPct_zscore"] = stats.groupby("Season")["W/L%"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Pool inicial de candidatas — inclui brutas e derivadas
FEATURE_COLS = [c for c in [
    "W/L%", "SRS", "ORtg", "DRtg", "NRtg", "Pace",
    "eFG%", "TOV%", "ORB%", "FT/FGA",
    "NRtg_zscore", "WinPct_zscore",
] if c in stats.columns]

stats.dropna(subset=FEATURE_COLS, inplace=True)
print(f"Pool inicial: {len(FEATURE_COLS)} features, {len(stats)} exemplos")
print(f"Distribuição reached_finals: {stats['reached_finals'].value_counts().to_dict()}")
print(f"Distribuição champion      : {stats['champion'].value_counts().to_dict()}")

"""Observa-se que o dataset apresenta um **desbalanceamento significativo** entre as classes, o que pode impactar negativamente o desempenho e a capacidade de generalização dos modelos de aprendizado de máquina. Enquanto a classe 0 possui 749 amostras, a classe 1 contém apenas 26, evidenciando uma distribuição bastante desigual dos dados.

Análise da distribuição das features por classe:
"""

# Gráficos comparando a distribuição das principais features entre times que chegaram às Finais e os que não.
# Queremos visualizar se as features realmente separam as classes.

PLOT_FEATS = [c for c in FEATURE_COLS if c in stats.columns]
n_cols = 4
n_rows = math.ceil(len(PLOT_FEATS) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
fig.suptitle("Distribuição: Finalistas vs Não-Finalistas", fontsize=13, fontweight="bold")
axes = axes.flatten()

for ax, feat in zip(axes, PLOT_FEATS):
    for val, label, color in [(0, "Não finalista", "#4C72B0"), (1, "Finalista", "#DD8452")]:
        subset = stats[stats["reached_finals"] == val][feat].dropna()
        ax.hist(subset, bins=20, alpha=0.6, label=label, color=color, density=True)
    ax.set_title(feat, fontsize=11)
    ax.legend(fontsize=8)

for i in range(len(PLOT_FEATS), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("eda_distributions.png", bbox_inches="tight")
plt.show()

# obs: usando o pace como exemplo, note como essa não é uma boa característica por ter muita sobreposição de dados, não sendo um bom indicador pra basear a nossa predição
# o mesmo vale pra TOV%, FT/FGA (esse mais ou menos, porque tem uma separaçãozinha no final), ORB%. São todas medidas que não separam bem times finalistas de não finalistas

"""A análise das distribuições por classe indica que **métricas de desempenho relativo à temporada** (NRtg_zscore, WinPct_zscore) apresentam maior separação entre finalistas e não-finalistas do que métricas absolutas (ORtg, DRtg), sugerindo que o desempenho relativo ao contexto da temporada é mais informativo que o desempenho absoluto para a tarefa de classificação.

Features como **TOV%, ORB% e Pace** exibem distribuições praticamente sobrepostas entre as classes, indicando baixo poder discriminativo individual. Por esse motivo, esses atributos serão eliminados das features de treinamento:
"""

# Remover features com baixo poder discriminativo
to_drop = ["ORB%", "Pace", "TOV%"]

FEATURE_COLS = [f for f in FEATURE_COLS if f not in to_drop]

print(f"Features atualizadas ({len(FEATURE_COLS)}): {FEATURE_COLS}")

"""Análise de correlação:"""

# Verificar a correlação entre todas as features e os dois targets (reached_finals e champion).
# Features muito correlacionadas entre si podem ser removidas (redundância).
# Features correlacionadas com o target são as mais importantes para o modelo.

cols_corr = FEATURE_COLS + ["reached_finals", "champion"]
corr = stats[cols_corr].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(8, 6))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0,
            linewidths=0.5, annot_kws={"size": 8})
plt.title("Correlação entre features e targets", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_correlation.png", bbox_inches="tight")
plt.show()

"""A matriz de correlação revela um **grupo de features altamente correlacionados** (W/L%, SRS, NRtg, NRtg_zscore, WinPct_zscore), com correlações superiores a 0.98 entre si. Para evitar multicolinearidade, apenas **NRtg_zscore** e **WinPct_zscore** foram retidos como representantes do grupo, por capturarem desempenho relativo à temporada. **FT/FGA** foi descartada por correlação próxima de zero com os atributos alvo."""

to_drop_redundantes = [
    "SRS",    # correlação 0.99 com NRtg_zscore
    "NRtg",   # correlação 1.00 com NRtg_zscore
    "W/L%",   # correlação 0.98 com WinPct_zscore
    "FT/FGA", # correlação baixa com alvos
]

stats = stats.drop(columns=[c for c in to_drop_redundantes if c in stats.columns])
FEATURE_COLS = [c for c in FEATURE_COLS if c not in to_drop_redundantes]

print(f"Features finais ({len(FEATURE_COLS)}): {FEATURE_COLS}\n")
# Esperado: ['ORtg', 'DRtg', 'eFG%', 'NRtg_zscore', 'WinPct_zscore']

# Heatmap final com as features selecionadas
cols_corr = FEATURE_COLS + ["reached_finals", "champion"]
corr = stats[cols_corr].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(7, 5))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0,
            linewidths=0.5, annot_kws={"size": 9})
plt.title("Correlação — features finais", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_correlation_final.png", bbox_inches="tight")
plt.show()

# Salvar dataset e lista de features para uso nos blocos seguintes
stats.to_csv("nba_prepared.csv", index=False)
joblib.dump(FEATURE_COLS, "feature_cols.pkl")
print(f"Dataset salvo: {stats.shape} | Features: {FEATURE_COLS}")

"""### **Amostragem: divisão temporal**

Dados de séries temporais NÃO podem ser embaralhados aleatoriamente.
Se treinarmos com dados de 2023 e testarmos em 2018, o modelo "vê o futuro"
— isso é **data leakage** e invalida completamente os resultados. Por isso, a divisão adotada foi:
- **Treino** : 2000–2018  (19 temporadas, 19 campeões)
- **Teste**  : 2019–2025  (7 temporadas, 7 campeões)

O conjunto de teste foi escolhido com 7 campeões para que o AUC final
tenha variância aceitável. Com apenas 3 campeões, um único erro moveria
o AUC em ~0.15 — tornando o número estatisticamente frágil.
"""

data = pd.read_csv("nba_prepared.csv")
FEATURE_COLS = joblib.load("feature_cols.pkl")

train = data[data["Season"] <= 2018]
test  = data[data["Season"] >= 2019]

X_train = train[FEATURE_COLS]
X_test  = test[FEATURE_COLS]

print(f"Treino : {len(train)} registros (2000–2018) | Campeões: {train['champion'].sum()}")
print(f"Teste  : {len(test)} registros (2019–2025) | Campeões: {test['champion'].sum()}")

"""### **Treinamento dos modelos**

Treinamos 5 algoritmos para cada target usando GridSearchCV com
TimeSeriesSplit — que garante que a validação cruzada também respeita
a ordem temporal, nunca usando dados futuros para validar dados passados.

Como vimos, o dataset é desbalanceado (ex: 52 finalistas em 775 exemplos), por isso:
  - Usamos **class_weight="balanced"** nos modelos que suportam
  - Avaliamos com **AUC-ROC** em vez de acurácia (acurácia seria enganosa)
  - Aplicamos **StandardScaler** apenas nos modelos que precisam (LR, SVM, KNN)
"""

def treinar_modelos(target):
    print(f"\n{'─'*55}")
    print(f"  Treinando para target: {target.upper()}")
    print(f"{'─'*55}")

    y_train = train[target]
    y_test  = test[target]

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)   # fit APENAS no treino
    X_te_sc = scaler.transform(X_test)         # aplica o mesmo scaler no teste

    modelos = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            {"C": [0.01, 0.1, 1, 10]}
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {"n_estimators": [100, 200], "max_depth": [3, 5, 7]}
        ),
        "XGBoost (GBM)": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]}
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [7, 11, 15, 21], "weights": ["uniform", "distance"]}
        ),
    }

    # Modelos baseados em distância precisam de escala; tree-based não precisam
    usa_escala = {"Logistic Regression", "SVM", "KNN"}
    tscv       = TimeSeriesSplit(n_splits=5)
    resultados = {}

    for nome, (modelo, grid) in modelos.items():
        Xtr = X_tr_sc if nome in usa_escala else X_train.values
        Xte = X_te_sc if nome in usa_escala else X_test.values

        gs = GridSearchCV(modelo, grid, cv=tscv, scoring="roc_auc", n_jobs=-1)
        gs.fit(Xtr, y_train)

        melhor = gs.best_estimator_
        y_prob = melhor.predict_proba(Xte)[:, 1]
        y_pred = melhor.predict(Xte)
        auc    = roc_auc_score(y_test, y_prob)

        resultados[nome] = {
            "model": melhor, "y_prob": y_prob,
            "y_pred": y_pred, "auc": auc
        }
        print(f"  {nome:<25}  AUC = {auc:.4f}   params = {gs.best_params_}")

    return resultados, y_test, scaler


resultados_finals, y_test_finals, scaler_f = treinar_modelos("reached_finals")
resultados_champ,  y_test_champ,  scaler_c = treinar_modelos("champion")

"""### **Avaliação dos modelos**"""

def plot_roc_side_by_side(resultados1, y_test1, titulo1,
                          resultados2, y_test2, titulo2, arquivo):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    estilos = ["-", "--", "-.", (0,(3,1,1,1)), ":"]

    for ax, resultados, y_test, titulo in zip(
        axes,
        [resultados1, resultados2],
        [y_test1, y_test2],
        [titulo1, titulo2]
    ):
        for (nome, res), ls in zip(resultados.items(), estilos):
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            ax.plot(fpr, tpr, ls=ls, lw=2, label=f"{nome} (AUC={res['auc']:.3f})")
        ax.plot([0,1],[0,1], "k--", alpha=0.4)
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(arquivo, bbox_inches="tight")
    plt.show()

plot_roc_side_by_side(
    resultados_finals, y_test_finals, "Curvas ROC — Previsão de Finalistas",
    resultados_champ,  y_test_champ,  "Curvas ROC — Previsão de Campeão",
    "roc_comparacao.png"
)

# MATRIZES DE CONFUSÃO de todos os modelos, ambos os targets

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrices(resultados, y_test, titulo_geral, arquivo):
    """
    Plota matrizes de confusão para todos os modelos lado a lado.
    Usa o threshold padrão de 0.5 para converter probabilidades em classes.
    """
    nomes = list(resultados.keys())
    n = len(nomes)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(titulo_geral, fontsize=13, fontweight="bold", y=1.02)

    for ax, nome in zip(axes, nomes):
        y_pred = resultados[nome]["y_pred"]  # já calculado no treinamento
        cm = confusion_matrix(y_test, y_pred)

        group_names = ["TN", "FP", "FN", "TP"]
        group_counts = [f"{v}" for v in cm.flatten()]
        labels = [f"{name}\n{count}" for name, count in zip(group_names, group_counts)]
        labels = np.array(labels).reshape(2, 2)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Não (0)", "Sim (1)"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")

        # Substituir anotações padrão pelas com rótulos TN/FP/FN/TP
        for i in range(2):
            for j in range(2):
                ax.text(j, i, labels[i, j],
                        ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        # Limpar anotações padrão do ConfusionMatrixDisplay
        for text in disp.text_.ravel():
            text.set_visible(False)

        auc = resultados[nome]["auc"]
        ax.set_title(f"{nome}\nAUC={auc:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Previsto", fontsize=9)
        ax.set_ylabel("Real", fontsize=9)

    plt.tight_layout()
    plt.savefig(arquivo, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Figura salva: {arquivo}")


# Plot para os dois targets
plot_confusion_matrices(
    resultados_finals, y_test_finals,
    "Matrizes de Confusão — Previsão de Finalistas",
    "confusion_finals.png"
)

plot_confusion_matrices(
    resultados_champ, y_test_champ,
    "Matrizes de Confusão — Previsão de Campeão",
    "confusion_champion.png"
)

# RELATÓRIO TEXTUAL — Precision, Recall, F1 para cada modelo


def print_classification_reports(resultados, y_test, titulo):
    print(f"\n{'═'*60}")
    print(f"  {titulo}")
    print(f"{'═'*60}")
    for nome, res in resultados.items():
        print(f"\n── {nome} (AUC={res['auc']:.4f}) ──")
        print(classification_report(
            y_test, res["y_pred"],
            target_names=["Não (0)", "Sim (1)"],
            zero_division=0
        ))

print_classification_reports(resultados_finals, y_test_finals,
                              "Finalistas — Classification Report")
print_classification_reports(resultados_champ, y_test_champ,
                              "Campeão — Classification Report")

# TABELA RESUMO — TP, FP, FN, TN de todos os modelos


import pandas as pd

def tabela_resumo_cm(resultados, y_test, nome_target):
    rows = []
    for nome, res in resultados.items():
        cm = confusion_matrix(y_test, res["y_pred"])
        tn, fp, fn, tp = cm.ravel()
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall   = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1       = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0
        rows.append({
            "Modelo": nome,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": round(precisao, 3),
            "Recall":    round(recall, 3),
            "F1":        round(f1, 3),
            "AUC":       round(res["auc"], 4),
        })
    df_resumo = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    print(f"\n{'─'*60}")
    print(f"  Resumo — {nome_target}")
    print(f"{'─'*60}")
    print(df_resumo.to_string(index=False))
    return df_resumo

resumo_finals = tabela_resumo_cm(resultados_finals, y_test_finals, "Finalistas")
resumo_champ  = tabela_resumo_cm(resultados_champ,  y_test_champ,  "Campeão")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import precision_recall_curve, f1_score


def avaliar_ranking(resultados, X_test, y_test, df_test, target_col, titulo):
    temporadas = sorted(df_test['Season'].unique())
    print(f"\n{'═'*65}")
    print(f"  AVALIAÇÃO POR RANKING — {titulo}")
    print(f"{'═'*65}")

    summary = []

    for nome, res in resultados.items():
        top1 = top3 = top5 = 0
        reciprocal_ranks = []

        for season in temporadas:
            mask = df_test['Season'] == season
            X_s = X_test[mask]
            y_s = y_test[mask]

            if y_s.sum() == 0:
                continue

            # Probabilidade da classe positiva já guardada no treino
            proba = res['y_prob'][mask]
            ranking = np.argsort(proba)[::-1]

            positivos = np.where(y_s.values == 1)[0]
            melhor_rank = min([np.where(ranking == p)[0][0] for p in positivos]) + 1

            if melhor_rank == 1: top1 += 1
            if melhor_rank <= 3: top3 += 1
            if melhor_rank <= 5: top5 += 1
            reciprocal_ranks.append(1 / melhor_rank)

            print(f"  {season} | {nome:<22} | rank do campeão: #{melhor_rank:>2} "
                  f"| top prob: {proba[ranking[0]]:.3f}")

        n = len(temporadas)
        mrr = np.mean(reciprocal_ranks)
        summary.append({
            'Modelo': nome,
            'Top-1 Acc': f"{top1}/{n} ({top1/n:.0%})",
            'Top-3 Acc': f"{top3}/{n} ({top3/n:.0%})",
            'MRR':       round(mrr, 3),
        })
        print()

    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    return df_summary

ranking_champ  = avaliar_ranking(resultados_champ,  X_test, y_test_champ,
                                  test, 'champion',      'CAMPEÃO')
ranking_finals = avaliar_ranking(resultados_finals, X_test, y_test_finals,
                                  test, 'reached_finals', 'FINALISTAS')

# calibração de Probabilidades (CalibratedClassifierCV)


print(f"\n{'═'*65}")
print("  CALIBRAÇÃO DE PROBABILIDADES — CalibratedClassifierCV")
print(f"{'═'*65}")

modelos_para_calibrar = {
    'SVM':          resultados_champ['SVM']['model'],
    'Random Forest': resultados_champ['Random Forest']['model'],
    'KNN':          resultados_champ['KNN']['model'],
}

resultados_calibrados = {}
usa_escala = {"Logistic Regression", "SVM", "KNN"}
y_train_champ = train["champion"]

fig, axes = plt.subplots(1, len(modelos_para_calibrar), figsize=(14, 4))
fig.suptitle('Calibração de Probabilidades — Previsão de Campeão', fontsize=12, fontweight='bold')

for ax, (nome, modelo) in zip(axes, modelos_para_calibrar.items()):

    # Aplica escala caso o modelo exija, usando o scaler salvo do treino original
    X_tr_cal = scaler_c.transform(X_train) if nome in usa_escala else X_train.values
    X_te_cal = scaler_c.transform(X_test) if nome in usa_escala else X_test.values

    cal = CalibratedClassifierCV(modelo, method='isotonic', cv=5)
    cal.fit(X_tr_cal, y_train_champ)

    proba_cal = cal.predict_proba(X_te_cal)[:, 1]
    proba_orig = resultados_champ[nome]['y_prob']

    try:
        frac_pos_orig, mean_pred_orig = calibration_curve(y_test_champ, proba_orig,  n_bins=5)
        frac_pos_cal,  mean_pred_cal  = calibration_curve(y_test_champ, proba_cal,   n_bins=5)

        ax.plot(mean_pred_orig, frac_pos_orig, 'o--', label='Original', color='steelblue')
        ax.plot(mean_pred_cal,  frac_pos_cal,  's-',  label='Calibrado', color='darkorange')
    except Exception:
        pass

    ax.plot([0, 1], [0, 1], 'k:', label='Perfeito')
    ax.set_title(f'{nome}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Probabilidade predita')
    ax.set_ylabel('Fração real de positivos')
    ax.legend(fontsize=8)

    resultados_calibrados[nome] = {'model': cal, 'proba': proba_cal}
    print(f"\n{nome} calibrado. Probabilidade máxima no teste: {proba_cal.max():.4f}")

plt.tight_layout()
plt.savefig('calibracao_probabilidades.png', dpi=150, bbox_inches='tight')
plt.show()



print(f"\n{'═'*65}")
print("  THRESHOLD ÓTIMO — Curva Precision-Recall (F1 máximo)")
print(f"{'═'*65}")

fig, axes = plt.subplots(1, len(resultados_champ), figsize=(18, 4))
fig.suptitle('Curvas Precision-Recall e Threshold Ótimo — Campeão', fontsize=12, fontweight='bold')

thresholds_otimos = {}

for ax, (nome, res) in zip(axes, resultados_champ.items()):
    proba = res['y_prob']
    prec, rec, threshs = precision_recall_curve(y_test_champ, proba)

    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    idx_best  = np.argmax(f1_scores)
    thresh_best = threshs[idx_best]
    f1_best     = f1_scores[idx_best]

    thresholds_otimos[nome] = thresh_best

    ax.plot(rec, prec, color='steelblue', lw=2)
    ax.axvline(rec[idx_best],  color='darkorange', linestyle='--', alpha=0.7)
    ax.scatter([rec[idx_best]], [prec[idx_best]], color='darkorange', zorder=5,
               label=f'Thr={thresh_best:.2f}\nF1={f1_best:.3f}')
    ax.set_title(f'{nome}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(fontsize=8)

    y_pred_opt = (proba >= thresh_best).astype(int)
    f1_opt = f1_score(y_test_champ, y_pred_opt, zero_division=0)
    print(f"{nome:<22} | threshold ótimo: {thresh_best:.3f} | F1 com thr ótimo: {f1_opt:.3f}")

plt.tight_layout()
plt.savefig('threshold_otimo.png', dpi=150, bbox_inches='tight')
plt.show()

# COMPARATIVO FINAL


print(f"\n{'═'*65}")
print("  COMPARATIVO FINAL — threshold 0,5 vs threshold ótimo")
print(f"{'═'*65}")

rows = []
for nome, res in resultados_champ.items():
    proba    = res['y_prob']
    y_05     = (proba >= 0.5).astype(int)
    y_opt    = (proba >= thresholds_otimos[nome]).astype(int)
    f1_05    = f1_score(y_test_champ, y_05,  zero_division=0)
    f1_opt   = f1_score(y_test_champ, y_opt, zero_division=0)
    rows.append({
        'Modelo':         nome,
        'F1 (thr=0,50)':  round(f1_05,  3),
        'F1 (thr ótimo)': round(f1_opt,  3),
        'Threshold ótimo': round(thresholds_otimos[nome], 3),
        'Ganho F1':        round(f1_opt - f1_05, 3),
    })

print(pd.DataFrame(rows).to_string(index=False))

# ================================================================
# MATRIZES DE CONFUSÃO OTIMIZADAS (Com Threshold Ajustado e AUC)
# ================================================================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrices_otimizadas(resultados, y_test, thresholds, titulo_geral, arquivo):
    """
    Plota matrizes de confusão usando os thresholds ótimos encontrados.
    """
    nomes = list(resultados.keys())
    n = len(nomes)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(titulo_geral, fontsize=13, fontweight="bold", y=1.02)

    for ax, nome in zip(axes, nomes):
        # Pega as probabilidades e o AUC do modelo
        proba = resultados[nome]["y_prob"]
        auc = resultados[nome]["auc"]

        # Pega o threshold ótimo específico deste modelo
        thr = thresholds.get(nome, 0.5)

        # Gera as predições com o novo limiar
        y_pred_opt = (proba >= thr).astype(int)

        # Calcula a nova matriz
        cm = confusion_matrix(y_test, y_pred_opt)

        group_names = ["TN", "FP", "FN", "TP"]
        group_counts = [f"{v}" for v in cm.flatten()]
        labels = [f"{name}\n{count}" for name, count in zip(group_names, group_counts)]
        labels = np.array(labels).reshape(2, 2)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Não (0)", "Sim (1)"])
        disp.plot(ax=ax, colorbar=False, cmap="Greens") # Usando verde para diferenciar das antigas

        for i in range(2):
            for j in range(2):
                ax.text(j, i, labels[i, j],
                        ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        for text in disp.text_.ravel():
            text.set_visible(False)

        ax.set_title(f"{nome}\nAUC={auc:.3f} | Thr={thr:.2f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Previsto", fontsize=9)
        ax.set_ylabel("Real", fontsize=9)

    plt.tight_layout()
    plt.savefig(arquivo, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Figura salva: {arquivo}")

print(f"\n{'═'*65}")
print("  MATRIZES DE CONFUSÃO OTIMIZADAS (Limiar Dinâmico)")
print(f"{'═'*65}")

plot_confusion_matrices_otimizadas(
    resultados_champ,
    y_test_champ,
    thresholds_otimos,
    "Matrizes de Confusão Otimizadas — Previsão de Campeão",
    "confusion_champion_opt.png"
)

"""O modelo SVM apresentou desempenho muito baixo na previsão de finalistas, produzindo resultados próximos ao aleatório e demonstrando baixa capacidade de generalização para esse problema.

Random Forest e XGBoost calculam a importância de cada feature durante o
treino. Isso mostra QUAIS informações o modelo considerou mais relevantes
para tomar suas decisões — e justifica nossas escolhas de features.
"""

def plot_importance(resultados, feat_cols, titulo, arquivo):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    fig.suptitle(titulo, fontsize=13, fontweight="bold")

    for ax, nome in zip(axes, ["Random Forest", "XGBoost (GBM)"]):
        modelo = resultados[nome]["model"]
        imp    = modelo.feature_importances_
        idx    = np.argsort(imp)[::-1]
        labels = [feat_cols[i] for i in idx]

        ax.barh(labels, imp[idx], color="#4C72B0")
        ax.invert_yaxis()
        ax.set_title(nome, fontsize=11, fontweight="bold")
        ax.set_xlabel("Importância relativa")

    plt.tight_layout()
    plt.savefig(arquivo, bbox_inches="tight")
    plt.show()

plot_importance(resultados_finals, FEATURE_COLS,
                "Feature Importance — Finals", "importance_finals.png")
plot_importance(resultados_champ,  FEATURE_COLS,
                "Feature Importance — Champion",    "importance_champion.png")

"""A análise de importância de features revelou que **métricas de desempenho relativo à temporada (NRtg_zscore, WinPct_zscore) foram consistentemente as mais relevantes em ambas as tarefas de classificação**, superando métricas absolutas como ORtg e DRtg.

Para a previsão de finalistas, a dominância relativa (NRtg_zscore) foi o fator principal, enquanto para a previsão de campeão, a consistência ao longo da temporada (WinPct_zscore) mostrou maior poder discriminativo.

Tabela comparativa de AUC e seleção dos melhores modelos:
"""

print("AUC-ROC no conjunto de teste (2019–2025)")
print(f"\n{'Modelo':<25} {'Finalistas':>12} {'Campeão':>12}")
print("─" * 52)
for nome in resultados_finals:
    af = resultados_finals[nome]["auc"]
    ac = resultados_champ[nome]["auc"]
    print(f"{nome:<25} {af:>12.4f} {ac:>12.4f}")

melhor_f = max(resultados_finals, key=lambda k: resultados_finals[k]["auc"])
melhor_c = max(resultados_champ,  key=lambda k: resultados_champ[k]["auc"])

print(f"\nMelhor para finalistas : {melhor_f} (AUC={resultados_finals[melhor_f]['auc']:.4f})")
print(f"Melhor para campeão    : {melhor_c} (AUC={resultados_champ[melhor_c]['auc']:.4f})")

# Salvar os melhores modelos e scalers para uso na aplicação
joblib.dump(resultados_finals[melhor_f]["model"], "model_finals.pkl")
joblib.dump(resultados_champ[melhor_c]["model"],  "model_champion.pkl")
joblib.dump(scaler_f, "scaler_finals.pkl")
joblib.dump(scaler_c, "scaler_champion.pkl")
joblib.dump(FEATURE_COLS, "feature_cols.pkl")

"""### **Retreinamento e aplicação dos modelos**

Os modelos foram inicialmente **retreinados com dados históricos até 2024**, permitindo validar as previsões na temporada de 2025, em que o Oklahoma City Thunder conquistou o título. Após essa etapa de validação, os modelos foram **novamente treinados utilizando todo o histórico disponível** (2000–2025) para gerar as previsões da temporada de 2026.

O retreinamento foi necessário porque os modelos de avaliação apresentados anteriormente haviam sido treinados apenas com dados até 2018. Ao incorporar temporadas mais recentes, aumenta-se a quantidade de dados disponíveis e melhora-se a capacidade de generalização e calibração dos modelos.
"""

data = pd.read_csv("nba_prepared.csv")
FEATURE_COLS = joblib.load("feature_cols.pkl")

# treinar até 2024, validar em 2025 —
train_ate_2024 = data[data["Season"] <= 2024].copy()
X_tr = train_ate_2024[FEATURE_COLS]

scaler_val = StandardScaler()
X_tr_sc = scaler_val.fit_transform(X_tr)

model_val = LogisticRegression(C=0.1, max_iter=1000,
                                class_weight="balanced", random_state=42)
model_val.fit(X_tr_sc, train_ate_2024["champion"])

season_2025 = data[data["Season"] == 2025].copy()
X_25_sc = scaler_val.transform(season_2025[FEATURE_COLS])
probs_25 = model_val.predict_proba(X_25_sc)[:, 1]

season_2025["prob_%"]    = (probs_25 / probs_25.sum() * 100).round(2)
season_2025["resultado"] = season_2025["champion"].map({1: "🏆     ", 0: ""})

ranking_2025 = (season_2025[["Team", "W", "prob_%", "resultado"]]
                .sort_values("prob_%", ascending=False)
                .reset_index(drop=True))
ranking_2025.index += 1

pos_okc  = ranking_2025[ranking_2025["resultado"] == "🏆     "].index[0]
prob_okc = ranking_2025[ranking_2025["resultado"] == "🏆     "]["prob_%"].values[0]

print("Validação — Temporada 2025")
print(f"Campeão real: OKC Thunder")
print(f"OKC ficou na posição #{pos_okc} com {prob_okc}%\n")
print(ranking_2025[["Team", "W", "prob_%", "resultado"]].head(10).to_string())

"""Retreinar com tudo (2000–2025) e prever 2026:"""

train_full   = data.copy()
X_train_full = train_full[FEATURE_COLS]

scaler_f26 = StandardScaler()
model_f26  = LogisticRegression(C=0.01, max_iter=1000,
                                 class_weight="balanced", random_state=42)
model_f26.fit(scaler_f26.fit_transform(X_train_full), train_full["reached_finals"])

scaler_c26 = StandardScaler()
model_c26  = LogisticRegression(C=0.1, max_iter=1000,
                                 class_weight="balanced", random_state=42)
model_c26.fit(scaler_c26.fit_transform(X_train_full), train_full["champion"])

data_2026 = pd.DataFrame([
    # Time                      W    L    W/L%   NRtg   ORtg    DRtg   eFG%
    ["OKC Thunder",             64,  18,  0.780, 11.2,  119.0,  107.8, 0.572],
    ["San Antonio Spurs",       62,  20,  0.756,  8.3,  119.7,  111.4, 0.568],
    ["Detroit Pistons",         60,  22,  0.732,  8.1,  117.9,  109.8, 0.561],
    ["Boston Celtics",          56,  26,  0.683,  8.1,  120.8,  112.7, 0.588],
    ["Denver Nuggets",          54,  28,  0.659,  5.2,  122.6,  117.4, 0.571],
    ["NY Knicks",               53,  29,  0.646,  6.5,  119.9,  113.4, 0.558],
    ["LA Lakers",               53,  29,  0.646,  1.8,  118.3,  116.5, 0.562],
    ["Cleveland Cavs",          52,  30,  0.634,  4.1,  119.2,  115.1, 0.554],
    ["Houston Rockets",         52,  30,  0.634,  5.3,  118.6,  113.3, 0.551],
    ["Minnesota TWolves",       49,  33,  0.598,  3.3,  116.8,  113.5, 0.547],
    ["Atlanta Hawks",           46,  36,  0.561,  2.3,  116.1,  113.8, 0.543],
    ["Toronto Raptors",         46,  36,  0.561,  2.8,  115.9,  113.1, 0.539],
    ["Orlando Magic",           45,  37,  0.549,  0.7,  115.0,  114.3, 0.536],
    ["Philadelphia 76ers",      45,  37,  0.549, -0.2,  115.4,  115.6, 0.530],
    ["Phoenix Suns",            45,  37,  0.549,  1.5,  115.5,  114.0, 0.527],
    ["LA Clippers",             42,  40,  0.512,  1.1,  117.3,  116.2, 0.524],
    ["Portland Trail Blazers",  42,  40,  0.512, -0.3,  114.5,  114.8, 0.521],
    ["Miami Heat",              43,  39,  0.524,  2.2,  116.7,  114.5, 0.517],
    ["Charlotte Hornets",       44,  38,  0.537,  5.0,  119.4,  114.4, 0.514],
    ["Golden State Warriors",   37,  45,  0.451, -0.5,  115.1,  115.6, 0.511],
    ["Milwaukee Bucks",         32,  50,  0.390, -6.3,  113.0,  119.3, 0.507],
    ["Chicago Bulls",           31,  51,  0.378, -5.1,  113.0,  118.1, 0.502],
    ["New Orleans Pelicans",    26,  56,  0.317, -4.4,  114.5,  118.9, 0.498],
    ["Dallas Mavericks",        26,  56,  0.317, -5.4,  111.2,  116.6, 0.494],
    ["Memphis Grizzlies",       25,  57,  0.305, -5.9,  112.9,  118.8, 0.489],
    ["Indiana Pacers",          19,  63,  0.232, -7.9,  110.9,  118.8, 0.485],
    ["Sacramento Kings",        22,  60,  0.268,-10.0,  111.5,  121.5, 0.481],
    ["Utah Jazz",               22,  60,  0.268, -8.2,  114.2,  122.4, 0.477],
    ["Brooklyn Nets",           20,  62,  0.244,-10.3,  108.7,  119.0, 0.473],
    ["Washington Wizards",      17,  65,  0.207,-11.8,  110.9,  122.7, 0.469],
], columns=["Team", "W", "L", "W/L%", "NRtg", "ORtg", "DRtg", "eFG%"])

# Z-scores calculados sobre os 30 times, igual ao processo do treino
data_2026["NRtg_zscore"]   = (data_2026["NRtg"]  - data_2026["NRtg"].mean())  / data_2026["NRtg"].std()
data_2026["WinPct_zscore"] = (data_2026["W/L%"]  - data_2026["W/L%"].mean())  / data_2026["W/L%"].std()

feats_2026 = [c for c in FEATURE_COLS if c in data_2026.columns]
X_26 = data_2026[feats_2026]

prob_f = model_f26.predict_proba(scaler_f26.transform(X_26))[:, 1]
prob_c = model_c26.predict_proba(scaler_c26.transform(X_26))[:, 1]

# Normalizar para somar 100%
data_2026["P(Finais) %"]  = (prob_f / prob_f.sum() * 100).round(2)
data_2026["P(Campeão) %"] = (prob_c / prob_c.sum() * 100).round(2)

ranking_2026 = (data_2026[["Team","W","L","NRtg","P(Finais) %","P(Campeão) %"]]
                .sort_values("P(Campeão) %", ascending=False)
                .reset_index(drop=True))
ranking_2026.index += 1

print("PREVISÃO FINAL — NBA 2026\n")
print(ranking_2026.to_string())
ranking_2026.to_csv("nba_2026_ranking.csv", index=False)

"""Gráfico de probabilidades para 2026:"""

top8  = ranking_2026.head(8)
cores = ["#D4AF37", "#C0C0C0", "#CD7F32"] + ["#4C72B0"] * 5

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("NBA 2026 Championship probability forecast", fontsize=14, fontweight="bold")

for ax, col, label in zip(
    axes,
    ["P(Finais) %", "P(Campeão) %"],
    ["Finals Probability (%)", "Champions Probability (%)"]
):
    bars = ax.barh(top8["Team"][::-1], top8[col][::-1], color=cores[::-1])
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xlabel(label, fontsize=10)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, top8[col].max() * 1.3)

plt.tight_layout()
plt.savefig("nba_2026_predictions.png", bbox_inches="tight", dpi=150)
plt.show()