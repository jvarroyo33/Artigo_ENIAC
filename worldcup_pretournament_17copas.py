# -*- coding: utf-8 -*-
"""WorldCup_PreTournament_17Copas.ipynb


# FIFA World Cup — Pre-Tournament Champion Forecast
## 17 Copas (1958–2022) | Qualifying + ELO + Historical Titles + Robustez Estatística

**Framework paper:** *A Supervised Machine Learning Framework for Sports Championship Prediction*



### Células de análise:
- **Cell 1–2** — imports e download (igual ao original)
- **Cell 3** — ELO com snapshots para todas as 17 copas
- **Cell 4** — configuração completa: CONF, TITLES_BEFORE, CUP_WINDOWS, CUP_TEAMS, CUPS_TARGETS para 17 copas
- **Cell 5** — feature engineering (idêntico ao original — pipeline generaliza automaticamente)
- **Cell 6** — integrity check (todos os ✓ passam nas 17 copas)
- **Cell 7–8** — EDA com dataset ampliado
- **Cell 9** — split temporal: treino 1958–2006 / teste 2010–2022 (4 campeões)
- **Cell 10** — 5 modelos (GridSearchCV + TimeSeriesSplit)
- **Cell 10b** — Bootstrap 95% CI para AUC
- **Cell 10c** — Leave-One-Cup-Out CV (17 folds)
- **Cell 10d** — Threshold F1-ótimo + PR Curves
- **Cell 10e** — McNemar Test (agora com poder estatístico real)
- **Cell 11–14** — tabelas, ROC, confusion matrices, feature importance
- **Cell 15** — tabela de robustez consolidada
- **Cell 16–17** — validação 2022 + forecast 2026

### Notas sobre nomes históricos de times
O dataset `martj42/international_results` usa nomes unificados:
- `"Germany"` = West Germany (copas até 1990) + Alemanha reunificada (1994+)
- `"Russia"` = USSR/Soviet Union (copas até 1990) + Rússia (1994+)
- `"Czechoslovakia"` = nome real no dataset (aparece até 1994)
- `"Republic of Ireland"` = Irlanda (nome real no dataset)
- `"United States"` = USA (nome real no dataset pós-1990)
- Times sem dados de qualificatória (hosts, dados ausentes) → imputação pela mediana da confederação

## 1: Imports
"""

import os, warnings, urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random, os

# Adicionar isso logo após os imports:
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             average_precision_score, precision_recall_curve)
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")
print("✓ Imports OK")

"""## 2: Download"""

import os
import urllib.request
import pandas as pd

HASH_DO_COMMIT = "9ccacf7cd174236ff196d1190c6f8d6f3c909f38"

# 2. A URL agora aponta para o estado exato daquele commit
DATA_URL = f"https://raw.githubusercontent.com/martj42/international_results/{HASH_DO_COMMIT}/results.csv"
DATA_PATH = f"results_{HASH_DO_COMMIT[:7]}.csv" # Salva com o ID do commit para não misturar

if not os.path.exists(DATA_PATH):
    print(f"Downloading dataset from commit {HASH_DO_COMMIT[:7]} (~5 MB)...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("✓ Download complete")
else:
    print("✓ Dataset already exists locally")

df_raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
df_raw = df_raw.dropna(subset=["home_team","away_team","home_score","away_score"])
df_raw = df_raw.sort_values("date").reset_index(drop=True)
quals  = df_raw[df_raw["tournament"] == "FIFA World Cup qualification"].copy()

print(f"Total matches : {len(df_raw):,}")
print(f"Date range    : {df_raw['date'].min().date()} → {df_raw['date'].max().date()}")
print(f"Qualifying    : {len(quals):,} matches")

"""## 3: ELO"""

K_MAP = {
    "FIFA World Cup":               60,
    "UEFA Euro":                    45,
    "Copa América":                 45,
    "African Cup of Nations":       40,
    "AFC Asian Cup":                40,
    "Gold Cup":                     35,
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification":      30,
    "African Cup of Nations qualification": 30,
    "AFC Asian Cup qualification":  28,
    "CONCACAF Nations League":      28,
    "UEFA Nations League":          28,
    "Friendly":                     20,
}
HOME_BONUS = 75

def get_k(tournament):
    for key, k in K_MAP.items():
        if key in str(tournament): return k
    return 25

def goal_mult(margin):
    if margin <= 1: return 1.0
    if margin == 2: return 1.5
    if margin == 3: return 1.75
    return 1.75 + (margin - 3) / 8

def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

# Snapshot ~1 mês antes de cada Copa
SNAPSHOT_DATES = {
    1958: "1958-06-01", 1962: "1962-05-30", 1966: "1966-07-01",
    1970: "1970-05-30", 1974: "1974-06-01", 1978: "1978-06-01",
    1982: "1982-06-01", 1986: "1986-05-30", 1990: "1990-06-01",
    1994: "1994-06-01", 1998: "1998-06-01", 2002: "2002-05-15",
    2006: "2006-06-01", 2010: "2010-06-01", 2014: "2014-06-01",
    2018: "2018-06-01", 2022: "2022-11-01", 2026: "2026-06-10",
}

elo       = {}
snapshots = {}
snap_done = set()

print("Computing ELO since 1872 — ~30s ...")
for _, row in df_raw.iterrows():
    for cup_year, snap_date in SNAPSHOT_DATES.items():
        if row["date"] >= pd.Timestamp(snap_date) and cup_year not in snap_done:
            snapshots[cup_year] = dict(elo)
            snap_done.add(cup_year)
    ht, at  = row["home_team"], row["away_team"]
    k       = get_k(row["tournament"])
    ra      = elo.get(ht, 1500)
    rb      = elo.get(at, 1500)
    neutral = str(row.get("neutral", "TRUE")).upper() == "TRUE"
    ea      = expected_score(ra + (HOME_BONUS if not neutral else 0), rb)
    hs, aws = int(row["home_score"]), int(row["away_score"])
    sa      = 1.0 if hs > aws else (0.5 if hs == aws else 0.0)
    gm      = goal_mult(abs(hs - aws))
    elo[ht] = ra + k * gm * (sa - ea)
    elo[at] = rb + k * gm * ((1 - sa) - (1 - ea))

if 2026 not in snap_done:
    snapshots[2026] = dict(elo)

print(f"✓ ELO OK | Snapshots: {sorted(snapshots.keys())}")

"""## 4: Configuração — 17 Copas"""

# ── 4a. Confederações ──────────────────────────────────────
# Inclui todos os times históricos que aparecem no dataset 1958-2022
CONF = {
    # CONMEBOL
    "Brazil":"CONMEBOL","Argentina":"CONMEBOL","Uruguay":"CONMEBOL",
    "Colombia":"CONMEBOL","Ecuador":"CONMEBOL","Chile":"CONMEBOL",
    "Paraguay":"CONMEBOL","Peru":"CONMEBOL","Bolivia":"CONMEBOL",
    "Venezuela":"CONMEBOL","Suriname":"CONMEBOL","Guyana":"CONMEBOL",

    # UEFA — times modernos
    "France":"UEFA","Germany":"UEFA","Italy":"UEFA","Spain":"UEFA",
    "Netherlands":"UEFA","England":"UEFA","Portugal":"UEFA","Croatia":"UEFA",
    "Belgium":"UEFA","Denmark":"UEFA","Sweden":"UEFA","Switzerland":"UEFA",
    "Czech Republic":"UEFA","Poland":"UEFA","Serbia":"UEFA","Greece":"UEFA",
    "Romania":"UEFA","Bulgaria":"UEFA","Slovakia":"UEFA","Slovenia":"UEFA",
    "Ireland":"UEFA","Norway":"UEFA","Ukraine":"UEFA","Austria":"UEFA",
    "Scotland":"UEFA","Turkey":"UEFA","Bosnia":"UEFA","Iceland":"UEFA",
    "Russia":"UEFA","Wales":"UEFA","Hungary":"UEFA","Albania":"UEFA",
    "Cyprus":"UEFA","Estonia":"UEFA","Finland":"UEFA","Faroe Islands":"UEFA",
    "Latvia":"UEFA","Lithuania":"UEFA","Luxembourg":"UEFA","Malta":"UEFA",
    "Northern Ireland":"UEFA","San Marino":"UEFA","Israel":"UEFA",
    "Republic of Ireland":"UEFA",
    # UEFA — times históricos (dataset usa estes nomes)
    "Yugoslavia":"UEFA","Czechoslovakia":"UEFA","German DR":"UEFA",
    "Serbia and Montenegro":"UEFA","Saarland":"UEFA",

    # CAF
    "Cameroon":"CAF","Nigeria":"CAF","Senegal":"CAF","Ghana":"CAF",
    "Morocco":"CAF","Tunisia":"CAF","Egypt":"CAF","South Africa":"CAF",
    "Algeria":"CAF","Ivory Coast":"CAF","Togo":"CAF","DR Congo":"CAF",
    "RD Congo":"CAF","Cabo Verde":"CAF","Angola":"CAF","Zambia":"CAF",
    "Zimbabwe":"CAF","Kenya":"CAF","Ethiopia":"CAF","Sudan":"CAF",
    "Uganda":"CAF","Tanzania":"CAF","Mozambique":"CAF","Malawi":"CAF",
    "Namibia":"CAF","Botswana":"CAF","Liberia":"CAF","Guinea":"CAF",
    "Burkina Faso":"CAF","Benin":"CAF","Niger":"CAF","Mali":"CAF",
    "Sierra Leone":"CAF","Libya":"CAF","Gabon":"CAF","Congo":"CAF",
    "Madagascar":"CAF","Mauritius":"CAF","Mauritania":"CAF",
    "Eswatini":"CAF","Lesotho":"CAF","Somalia":"CAF","Gambia":"CAF",

    # AFC
    "Japan":"AFC","South Korea":"AFC","Iran":"AFC","Saudi Arabia":"AFC",
    "Australia":"AFC","Iraq":"AFC","Qatar":"AFC","China":"AFC",
    "North Korea":"AFC","United Arab Emirates":"AFC","Jordan":"AFC",
    "Uzbekistan":"AFC","Kuwait":"AFC","Bahrain":"AFC","Syria":"AFC",
    "Lebanon":"AFC","Oman":"AFC","Vietnam":"AFC","Thailand":"AFC",
    "Indonesia":"AFC","Malaysia":"AFC","Singapore":"AFC","India":"AFC",
    "Bangladesh":"AFC","Sri Lanka":"AFC","Nepal":"AFC","Pakistan":"AFC",
    "Taiwan":"AFC","Hong Kong":"AFC","Macau":"AFC","Brunei":"AFC",
    "Vietnam Republic":"AFC","Yemen":"AFC","Yemen DPR":"AFC",

    # CONCACAF
    "Mexico":"CONCACAF","USA":"CONCACAF","United States":"CONCACAF",
    "Costa Rica":"CONCACAF","Honduras":"CONCACAF","Panama":"CONCACAF",
    "Trinidad":"CONCACAF","Trinidad and Tobago":"CONCACAF",
    "Jamaica":"CONCACAF","Canada":"CONCACAF","Haiti":"CONCACAF",
    "Curacao":"CONCACAF","Curaçao":"CONCACAF","Cuba":"CONCACAF",
    "El Salvador":"CONCACAF","Guatemala":"CONCACAF","Nicaragua":"CONCACAF",
    "Bermuda":"CONCACAF","Barbados":"CONCACAF","Puerto Rico":"CONCACAF",
    "Dominican Republic":"CONCACAF","Guyana":"CONCACAF",
    "Antigua and Barbuda":"CONCACAF",
    "Saint Lucia":"CONCACAF","Saint Vincent and the Grenadines":"CONCACAF",

    # OFC
    "New Zealand":"OFC","Australia":"OFC","Fiji":"OFC",
    "Solomon Islands":"OFC","Tahiti":"OFC","Vanuatu":"OFC",
}
# Nota: Australia migrou AFC→OFC→AFC; para simplificar: AFC pós-2006, OFC antes
# O z-score por confederação corrige automaticamente diferenças estruturais

# ── 4b. Títulos antes de cada edição ───────────────────────
TITLES_BEFORE = {
    1958: {"Brazil":1,"Italy":2,"Uruguay":2,"Germany":1,"England":0},
    1962: {"Brazil":2,"Italy":2,"Uruguay":2,"Germany":1},
    1966: {"Brazil":2,"Italy":2,"Uruguay":2,"Germany":1,"England":0},
    1970: {"Brazil":2,"Italy":2,"Uruguay":2,"Germany":1,"England":1},
    1974: {"Brazil":3,"Italy":2,"Uruguay":2,"Germany":1,"England":1},
    1978: {"Brazil":3,"Italy":2,"Uruguay":2,"Germany":2,"England":1,"Argentina":0},
    1982: {"Brazil":3,"Italy":2,"Uruguay":2,"Germany":2,"England":1,"Argentina":1},
    1986: {"Brazil":3,"Italy":3,"Uruguay":2,"Germany":2,"England":1,"Argentina":1},
    1990: {"Brazil":3,"Italy":3,"Uruguay":2,"Germany":2,"England":1,"Argentina":2},
    1994: {"Brazil":3,"Italy":3,"Uruguay":2,"Germany":3,"England":1,"Argentina":2},
    1998: {"Brazil":4,"Germany":3,"Italy":3,"Argentina":2,"Uruguay":2,"England":1},
    2002: {"Brazil":4,"Germany":3,"Italy":3,"Argentina":2,"Uruguay":2,"France":1,"England":1},
    2006: {"Brazil":5,"Germany":3,"Italy":3,"Argentina":2,"Uruguay":2,"France":1,"England":1},
    2010: {"Brazil":5,"Germany":3,"Italy":4,"Argentina":2,"Uruguay":2,"France":1,"England":1},
    2014: {"Brazil":5,"Germany":3,"Italy":4,"Argentina":2,"Uruguay":2,"France":1,"Spain":1,"England":1},
    2018: {"Brazil":5,"Germany":4,"Italy":4,"Argentina":2,"Uruguay":2,"France":1,"Spain":1,"England":1},
    2022: {"Brazil":5,"Germany":4,"Italy":4,"Argentina":2,"Uruguay":2,"France":2,"Spain":1,"England":1},
}
TITLES_2026 = {
    "Brazil":5,"Germany":4,"Italy":4,"Argentina":3,"Uruguay":2,
    "France":2,"Spain":1,"England":1,
}

# ── 4c. Janelas de qualificatória (baseadas nos dados reais do dataset) ─
CUP_WINDOWS = {
    1958: ("1953-01-01","1957-12-31"),
    1962: ("1957-01-01","1961-12-31"),
    1966: ("1961-01-01","1965-12-31"),
    1970: ("1965-01-01","1969-12-31"),
    1974: ("1969-01-01","1973-12-31"),
    1978: ("1973-01-01","1977-12-31"),
    1982: ("1977-01-01","1981-12-31"),
    1986: ("1981-01-01","1985-12-31"),
    1990: ("1985-01-01","1989-12-31"),
    1994: ("1989-01-01","1993-12-31"),
    1998: ("1993-01-01","1997-12-31"),
    2002: ("1997-12-01","2001-11-30"),
    2006: ("2001-12-01","2005-11-30"),
    2010: ("2005-12-01","2009-11-30"),
    2014: ("2009-12-01","2013-11-30"),
    2018: ("2013-12-01","2017-11-30"),
    2022: ("2017-12-01","2022-03-31"),
    2026: ("2022-03-01","2026-03-31"),
}

# ── 4d. Elencos classificados — 17 Copas ───────────────────
#
# CONVENÇÃO DE NOMES (alinhado ao dataset):
#   "Germany"          = West Germany nas copas até 1990, Germany a partir de 1994
#   "Russia"           = USSR/Soviet Union nas copas até 1990
#   "Czechoslovakia"   = nome real no dataset (existe até 1994)
#   "Yugoslavia"       = nome real no dataset
#   "German DR"        = Alemanha Oriental (dataset usa este nome)
#   "United States"    = USA pós-1990 (dataset usa "United States")
#   "Republic of Ireland" = Irlanda (dataset usa este nome)
#   "Trinidad and Tobago"  = dataset usa este nome (não "Trinidad")
#   "DR Congo"         = Zaire/Congo (dataset usa "DR Congo")
#
CUP_TEAMS = {
    1958: {
        "Brazil","France","Germany","Sweden","England","Russia",
        "Austria","Argentina","Northern Ireland","Czechoslovakia",
        "Yugoslavia","Paraguay","Scotland","Hungary","Wales","Mexico",
    },
    1962: {
        "Brazil","Czechoslovakia","Yugoslavia","Russia","Uruguay",
        "Colombia","Chile","Germany","Italy","Spain","Hungary",
        "England","Argentina","Bulgaria","Mexico","Switzerland",
    },
    1966: {
        "England","Germany","Portugal","Russia","Argentina","Uruguay",
        "Hungary","Spain","Brazil","Italy","Bulgaria","Mexico",
        "France","Chile","North Korea","Switzerland",
    },
    1970: {
        "Brazil","Italy","Germany","Uruguay","Russia","Mexico",
        "England","Peru","Belgium","Czechoslovakia","Romania","Sweden",
        "Bolivia","El Salvador","Israel","Morocco","Bulgaria","German DR",
    },
    1974: {
        "Germany","Netherlands","Poland","Brazil","Sweden","Yugoslavia",
        "Argentina","German DR","Uruguay","Chile","Scotland","Bulgaria",
        "DR Congo","Italy","Haiti","Australia",
    },
    1978: {
        "Argentina","Netherlands","Brazil","Italy","Germany","Poland",
        "Peru","France","Austria","Scotland","Hungary","Tunisia",
        "Sweden","Spain","Mexico","Iran",
    },
    1982: {
        "Italy","Germany","Poland","France","Brazil","England",
        "Austria","Russia","Belgium","Yugoslavia","Argentina",
        "Northern Ireland","Spain","Scotland","Algeria","Hungary",
        "Kuwait","Cameroon","Chile","Czechoslovakia","Honduras",
        "New Zealand","Peru",
    },
    1986: {
        "Argentina","Germany","France","Belgium","Brazil","England",
        "Spain","Mexico","Russia","Morocco","Denmark","Uruguay",
        "Italy","Poland","South Korea","Bulgaria","Hungary","Algeria",
        "Canada","Iraq","Northern Ireland","Paraguay","Portugal","Scotland",
    },
    1990: {
        "Germany","Argentina","Italy","Yugoslavia","England","Brazil",
        "Czechoslovakia","Costa Rica","Spain","Cameroon",
        "Republic of Ireland","Romania","Netherlands","Belgium",
        "Colombia","Uruguay","Sweden","Scotland","Egypt",
        "South Korea","United Arab Emirates","Austria",
        "Bolivia","United States",
    },
    1994: {
        "Brazil","Italy","Sweden","Bulgaria","Germany","Netherlands",
        "Romania","Argentina","Nigeria","Spain","Belgium",
        "United States","Mexico","South Korea","Switzerland",
        "Saudi Arabia","Russia","Cameroon","Bolivia","Greece",
        "Norway","Morocco","Republic of Ireland","Colombia",
    },
    1998: {
        "France","Brazil","Scotland","Norway","Italy","Chile",
        "Cameroon","Austria","Netherlands","Belgium","South Korea",
        "Mexico","Germany","Yugoslavia","Iran","United States",
        "Argentina","Jamaica","Japan","Croatia","England","Colombia",
        "Romania","Tunisia","Nigeria","Paraguay","Spain","Bulgaria",
    },
    2002: {
        "France","Senegal","Uruguay","Denmark","Spain","Paraguay",
        "South Africa","Slovenia","Brazil","Turkey","Costa Rica",
        "China","Germany","Republic of Ireland","Cameroon",
        "Saudi Arabia","Japan","Belgium","Russia","Tunisia",
        "Argentina","England","Nigeria","Sweden","Mexico","Croatia",
        "Ecuador","Italy","South Korea","United States","Portugal","Poland",
    },
    2006: {
        "Germany","Ecuador","Poland","Costa Rica","England","Sweden",
        "Paraguay","Trinidad and Tobago","Argentina","Netherlands",
        "Ivory Coast","Serbia and Montenegro","Portugal","Mexico",
        "Angola","Iran","Italy","Ghana","United States","Czech Republic",
        "Brazil","Australia","Japan","Croatia","France","Switzerland",
        "South Korea","Togo","Spain","Ukraine","Tunisia","Saudi Arabia",
    },
    2010: {
        "Uruguay","Mexico","South Africa","France","Argentina",
        "South Korea","Greece","Nigeria","England","United States",
        "Algeria","Slovenia","Germany","Australia","Ghana","Serbia",
        "Netherlands","Japan","Cameroon","Denmark","Paraguay",
        "Slovakia","New Zealand","Italy","Brazil","Portugal",
        "North Korea","Ivory Coast","Spain","Switzerland","Honduras","Chile",
    },
    2014: {
        "Brazil","Mexico","Croatia","Cameroon","Chile","Netherlands",
        "Spain","Australia","Colombia","Greece","Ivory Coast","Japan",
        "Uruguay","Costa Rica","England","Italy","France","Ecuador",
        "Switzerland","Honduras","Argentina","Nigeria","Bosnia","Iran",
        "Germany","Ghana","Portugal","United States","Belgium","Algeria",
        "Russia","South Korea",
    },
    2018: {
        "Uruguay","Russia","Saudi Arabia","Egypt","Portugal","Spain",
        "Iran","Morocco","France","Denmark","Peru","Australia",
        "Argentina","Croatia","Nigeria","Iceland","Brazil","Switzerland",
        "Costa Rica","Serbia","Sweden","Mexico","South Korea","Germany",
        "Belgium","England","Tunisia","Panama","Colombia","Japan",
        "Poland","Senegal",
    },
    2022: {
        "Netherlands","Senegal","Ecuador","Qatar","England",
        "United States","Iran","Wales","Argentina","Poland","Mexico",
        "Saudi Arabia","France","Australia","Tunisia","Denmark","Japan",
        "Spain","Germany","Costa Rica","Morocco","Croatia","Belgium",
        "Canada","Brazil","Switzerland","Cameroon","Serbia","Portugal",
        "Ghana","Uruguay","South Korea",
    },
    2026: {
        "Mexico","South Africa","South Korea","Czech Republic","Canada",
        "Qatar","Switzerland","Bosnia","Brazil","Morocco","Scotland",
        "Haiti","United States","Australia","Paraguay","Turkey","Germany",
        "Ecuador","Ivory Coast","Curaçao","Netherlands","Japan","Sweden",
        "Tunisia","Belgium","Iran","Egypt","New Zealand","Spain",
        "Cabo Verde","Saudi Arabia","Uruguay","France","Senegal","Norway",
        "Iraq","Argentina","Algeria","Austria","Jordan","Portugal",
        "Colombia","Uzbekistan","RD Congo","England","Croatia","Ghana","Panama",
    },
}

# ── 4e. Resultados das finais ───────────────────────────────
CUPS_TARGETS = {
    1958: ("Brazil",     "Sweden"),
    1962: ("Brazil",     "Czechoslovakia"),
    1966: ("England",    "Germany"),
    1970: ("Brazil",     "Italy"),
    1974: ("Germany",    "Netherlands"),
    1978: ("Argentina",  "Netherlands"),
    1982: ("Italy",      "Germany"),
    1986: ("Argentina",  "Germany"),
    1990: ("Germany",    "Argentina"),
    1994: ("Brazil",     "Italy"),
    1998: ("France",     "Brazil"),
    2002: ("Brazil",     "Germany"),
    2006: ("Italy",      "France"),
    2010: ("Spain",      "Netherlands"),
    2014: ("Germany",    "Argentina"),
    2018: ("France",     "Croatia"),
    2022: ("Argentina",  "France"),
}

print(f"✓ Config OK | {len(CUPS_TARGETS)} copas configuradas: {sorted(CUPS_TARGETS.keys())}")
print(f"  Confederações mapeadas: {len(set(CONF.values()))} grupos distintos")

"""## 5: Feature Engineering"""

def calc_qual_features(cup_year, start, end, team_set):
    """Computa estatísticas de qualificatória por time na janela dada."""
    w = quals[(quals["date"] >= start) & (quals["date"] <= end)]
    rows = []
    for team in team_set:
        h   = w[w["home_team"] == team]
        a   = w[w["away_team"] == team]
        gp  = len(h) + len(a)
        conf = CONF.get(team, "OTHER")
        if gp == 0:
            # Host ou dado ausente → imputação pela mediana da confederação (feita abaixo)
            rows.append({"team": team, "conf": conf, "q_gp": 0,
                         "q_win_pct": np.nan, "q_gd_pg": np.nan, "q_pts_pg": np.nan})
            continue
        gf = float(h["home_score"].sum() + a["away_score"].sum())
        ga = float(h["away_score"].sum() + a["home_score"].sum())
        ww = int((h["home_score"] > h["away_score"]).sum() +
                 (a["away_score"] > a["home_score"]).sum())
        dd = int((h["home_score"] == h["away_score"]).sum() +
                 (a["away_score"] == a["home_score"]).sum())
        rows.append({
            "team": team, "conf": conf, "q_gp": gp,
            "q_win_pct": ww / gp,
            "q_gd_pg":   (gf - ga) / gp,
            "q_pts_pg":  (ww * 3 + dd) / gp,
        })
    return pd.DataFrame(rows)


def build_cup_dataset(cup_year, titles_dict, include_target=True):
    """Monta a matriz de features completa para uma edição da Copa."""
    start, end = CUP_WINDOWS[cup_year]
    df_q = calc_qual_features(cup_year, start, end, CUP_TEAMS[cup_year])
    df_q["cup"] = cup_year

    # ── ELO snapshot ────────────────────────────────────────
    snap     = snapshots.get(cup_year, {})
    df_q["elo"] = df_q["team"].map(snap)
    df_q["elo"] = df_q["elo"].fillna(df_q["elo"].median())

    # ── Títulos históricos ───────────────────────────────────
    df_q["titles"] = df_q["team"].map(titles_dict).fillna(0).astype(int)

    # ── Imputação: hosts e dados ausentes → mediana da confederação ─
    for col in ["q_win_pct", "q_gd_pg", "q_pts_pg"]:
        df_q[col] = df_q.groupby("conf")[col].transform(
            lambda x: x.fillna(x.median()))
        df_q[col] = df_q[col].fillna(df_q[col].median()).fillna(0.0)

    # ── Z-scores globais (ELO e títulos) ────────────────────
    for col in ["elo", "titles"]:
        mu, sd = df_q[col].mean(), df_q[col].std()
        df_q[f"{col}_z"] = ((df_q[col] - mu) / (sd if sd > 0 else 1)).fillna(0.0)

    # ── Z-scores por confederação (stats de qualificatória) ─
    for col in ["q_win_pct", "q_gd_pg", "q_pts_pg"]:
        mu = df_q.groupby("conf")[col].transform("mean")
        sd = df_q.groupby("conf")[col].transform("std").replace(0, np.nan)
        df_q[f"{col}_conf_z"] = ((df_q[col] - mu) / sd).fillna(0.0)

    # ── Targets ─────────────────────────────────────────────
    if include_target and cup_year in CUPS_TARGETS:
        champ, runner = CUPS_TARGETS[cup_year]
        df_q["champion"]      = (df_q["team"] == champ).astype(int)
        df_q["reached_final"] = df_q["team"].isin([champ, runner]).astype(int)
    else:
        df_q["champion"] = np.nan
        df_q["reached_final"] = np.nan

    return df_q


# Construir dataset histórico completo (17 copas)
dfs = [build_cup_dataset(y, TITLES_BEFORE.get(y, {})) for y in sorted(CUPS_TARGETS.keys())]
df  = pd.concat(dfs, ignore_index=True)

FEATURE_COLS = ["elo_z", "q_win_pct_conf_z", "q_gd_pg_conf_z", "titles_z"]
SCALE_MODELS = {"Logistic Regression", "SVM", "KNN"}

print(f"✓ Dataset: {len(df)} instâncias | {df['cup'].nunique()} Copas "
      f"({df['cup'].min()}–{df['cup'].max()})")
print(f"  Champion rate   : {df['champion'].mean()*100:.1f}%")
print(f"  Finalist rate   : {df['reached_final'].mean()*100:.1f}%")
print(f"  Positivos totais: {int(df['champion'].sum())} campeões | "
      f"{int(df['reached_final'].sum())} finalistas")

"""## 6: Integrity Check"""

print("\n── Integrity check ─────────────────────────────────────")
all_ok = True
for cup in sorted(df["cup"].unique()):
    s      = df[df["cup"] == cup]
    champ  = s[s["champion"] == 1]["team"].tolist()
    finals = s[s["reached_final"] == 1]["team"].tolist()
    nans   = s[FEATURE_COLS].isna().sum().sum()
    ok_t   = "✓" if len(champ) == 1 and len(finals) == 2 else "⚠ ERRO"
    ok_f   = "✓" if nans == 0 else f"⚠ {nans} NaN"
    if "ERRO" in ok_t or "⚠" in ok_f:
        all_ok = False
    print(f"  {cup}: campeão={champ[0] if champ else '?':<20} "
          f"targets={ok_t}  features={ok_f}  n={len(s)}")
print(f"\n  Todos os checks OK: {all_ok}")

"""## 7: EDA — Distribuições e Correlação"""

FEAT_LABELS = {
    "elo_z":            "ELO z-score",
    "q_win_pct_conf_z": "Win% conf. z-score",
    "q_gd_pg_conf_z":   "GD/game conf. z-score",
    "titles_z":         "WC Titles z-score",
}

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Feature Distributions: Finalists vs Non-Finalists\n"
             f"World Cup Pre-Tournament Pipeline ({df['cup'].min()}–{df['cup'].max()})",
             fontsize=12, fontweight="bold")
COLORS = {0: "#4C72B0", 1: "#DD8452"}
LABELS = {0: "Non-Finalist", 1: "Finalist"}

for ax, feat in zip(axes, FEATURE_COLS):
    for cls, color in COLORS.items():
        ax.hist(df[df["reached_final"] == cls][feat], bins=20,
                alpha=0.6, color=color, label=LABELS[cls], density=True)
    ax.set_title(FEAT_LABELS[feat], fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.set_ylabel("Density"); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("fig1_wc_distributions_17cups.png", bbox_inches="tight", dpi=150)
plt.show()

# Correlação
targets  = ["reached_final", "champion"]
corr_df  = df[FEATURE_COLS + targets].corr()

plt.figure(figsize=(7, 5))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, linewidths=0.5,
            annot_kws={"size": 10})
plt.title(f"Correlação — Features vs Targets ({df['cup'].min()}–{df['cup'].max()})",
          fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("fig2_wc_correlation_17cups.png", bbox_inches="tight", dpi=150)
plt.show()

print("\nCorrelação com targets:")
print(corr_df[targets].round(3).to_string())

"""## 8: Boxplot por Confederação"""

fig, ax = plt.subplots(figsize=(10, 5))
conf_order = ["CONMEBOL", "UEFA", "CAF", "AFC", "CONCACAF", "OFC"]
valid = df[(df["q_gp"] > 0) & (df["conf"].isin(conf_order))]
data_by_conf = [valid[valid["conf"] == c]["q_win_pct"].values for c in conf_order]

bp = ax.boxplot(data_by_conf, labels=conf_order, patch_artist=True,
                medianprops=dict(color="black", lw=2))
colors_conf = ["#D62728","#1F77B4","#2CA02C","#FF7F0E","#9467BD","#8C564B"]
for patch, color in zip(bp["boxes"], colors_conf):
    patch.set_facecolor(color); patch.set_alpha(0.7)

ax.set_ylabel("Qualifying Win%", fontsize=11)
ax.set_title(f"Qualifying Win% by Confederation ({df['cup'].min()}–{df['cup'].max()})\n"
             "(CONMEBOL structurally harder — lower win rates)",
             fontsize=11, fontweight="bold")
ax.axhline(valid["q_win_pct"].mean(), color="grey", linestyle="--",
           alpha=0.6, label=f"Global mean ({valid['q_win_pct'].mean():.2f})")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig3_wc_conf_winpct_17cups.png", bbox_inches="tight", dpi=150)
plt.show()

"""## 9: Temporal Split"""

#
 # Com 17 copas (1958-2022), temos várias opções de split.
 # Escolha padrão: treino 1958-2006 (12 copas), teste 2010-2022 (5 copas, 5 campeões).
 # Isso resolve o principal problema estatístico: de 2 para 5 positivos no test set,
 # reduzindo a variância do AUC em ~60%.
 #
 # Alternativas comentadas abaixo.

#TRAIN_UNTIL = 2006
#TEST_FROM   = 2010

# Alternativa mais conservadora (mais treino):
TRAIN_UNTIL = 2010
TEST_FROM = 2014

# Alternativa replicando o paper original (só copas modernas):
# TRAIN_UNTIL = 2014; TEST_FROM = 2018  → test com 2 campeões (caso base)

train = df[df["cup"] <= TRAIN_UNTIL].copy()
test  = df[df["cup"] >= TEST_FROM].copy()

X_train = train[FEATURE_COLS]
X_test  = test[FEATURE_COLS]
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit ONLY em treino
X_test_sc  = scaler.transform(X_test)

tscv = TimeSeriesSplit(n_splits=5)

print(f"Train: {sorted(train['cup'].unique())} — {len(train)} instâncias, "
      f"{int(train['champion'].sum())} campeões")
print(f"Test : {sorted(test['cup'].unique())}  — {len(test)} instâncias, "
      f"{int(test['champion'].sum())} campeões")
print(f"\n→ Test set tem {int(test['champion'].sum())} campeões "
      f"(vs 2 no paper original). ΔAUCₘₐₓ por erro ≈ ±{1/int(test['champion'].sum()):.2f}")

"""## 10: Treinar 5 Modelos"""

MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {"n_estimators": [100, 200, 300, 400, 500], "max_depth": [3, 5]},
    ),
    "XGBoost (GBM)": (
        XGBClassifier(eval_metric="logloss", random_state=42, nthread=1),
        {"n_estimators": [100, 200], "max_depth": [2, 3, 5], "learning_rate": [0.05, 0.1]},
    ),
    "SVM": (
        SVC(probability=True, class_weight="balanced", random_state=42),
        {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [5, 7, 11], "weights": ["uniform", "distance"]},
    ),
}

def train_models(target):
    print(f"\n{'─'*55}  TARGET: {target.upper()}")
    y_train, y_test = train[target], test[target]
    results = {}
    for name, (model, grid) in MODELS.items():
        Xtr = X_train_sc if name in SCALE_MODELS else X_train.values
        Xte = X_test_sc  if name in SCALE_MODELS else X_test.values
        gs  = GridSearchCV(model, grid, cv=tscv, scoring="roc_auc", n_jobs=-1)
        gs.fit(Xtr, y_train)
        best   = gs.best_estimator_
        y_prob = best.predict_proba(Xte)[:, 1]
        y_pred = best.predict(Xte)
        auc    = roc_auc_score(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results[name] = {
            "model": best, "y_prob": y_prob, "y_pred": y_pred, "auc": auc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "prec":        precision_score(y_test, y_pred, zero_division=0),
            "rec":         recall_score(y_test, y_pred, zero_division=0),
            "f1":          f1_score(y_test, y_pred, zero_division=0),
            "best_params": gs.best_params_,
        }
        print(f"  {name:<25}  AUC={auc:.4f}  Recall={results[name]['rec']:.3f}  "
              f"F1={results[name]['f1']:.3f}  {gs.best_params_}")
    return results

results_final_wc = train_models("reached_final")
results_champ_wc = train_models("champion")

"""## 10b: Bootstrap 95% CI para AUC"""

def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap percentile CI para AUC-ROC."""
    rng = np.random.RandomState(seed)
    aucs = []
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs  = np.array(aucs)
    alpha = (1 - ci) / 2
    return np.mean(aucs), np.percentile(aucs, 100*alpha), np.percentile(aucs, 100*(1-alpha)), len(aucs)

y_true_ch = test["champion"].values
print("\nBootstrap 95% CI (n_boot=2000) — Champion")
print(f"{'Modelo':<25}  AUC     95% CI                n_boots")
print("─" * 62)
ci_results = {}
for name, r in results_champ_wc.items():
    m, lo, hi, nv = bootstrap_auc_ci(y_true_ch, r["y_prob"])
    ci_results[name] = (r["auc"], lo, hi)
    print(f"  {name:<23}  {r['auc']:.3f}   [{lo:.3f} – {hi:.3f}]   ({nv})")

# Plot CI
fig, ax = plt.subplots(figsize=(9, 4))
colors_m = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
for i, (name, (auc, lo, hi)) in enumerate(ci_results.items()):
    ax.barh(i, auc, 0.5, color=colors_m[i], alpha=0.8)
    ax.errorbar(auc, i, xerr=[[auc-lo],[hi-auc]],
                fmt='none', color='black', capsize=5, lw=2)
    ax.text(hi+0.005, i, f"{auc:.3f}  [{lo:.3f}–{hi:.3f}]", va='center', fontsize=8.5)
ax.set_yticks(range(len(ci_results)))
ax.set_yticklabels(list(ci_results.keys()), fontsize=10)
ax.set_xlabel("AUC-ROC"); ax.set_xlim(0.3, 1.2)
ax.axvline(0.5, color='gray', ls='--', alpha=0.5, lw=1)
ax.set_title(f"AUC-ROC com Bootstrap 95% CI (n=2000)\n"
             f"WC Champion — test {TEST_FROM}–2022 ({int(test['champion'].sum())} campeões)",
             fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("fig_bootstrap_ci_17cups.png", bbox_inches='tight', dpi=150)
plt.show()

"""## 10c: Leave-One-Cup-Out CV (LOCO)"""

cups_all  = sorted(df["cup"].unique())
loco_full = {name: [] for name in MODELS}
cups_used = []

for test_cup in cups_all:
    tr_l = df[df["cup"] != test_cup].copy()
    te_l = df[df["cup"] == test_cup].copy()
    if tr_l["champion"].sum() == 0 or te_l["champion"].sum() == 0:
        continue
    Xtr_l  = tr_l[FEATURE_COLS].values
    Xte_l  = te_l[FEATURE_COLS].values
    sc_l   = StandardScaler()
    Xtr_sc = sc_l.fit_transform(Xtr_l)
    Xte_sc = sc_l.transform(Xte_l)
    for name, (model, grid) in MODELS.items():
        Xtr_ = Xtr_sc if name in SCALE_MODELS else Xtr_l
        Xte_ = Xte_sc if name in SCALE_MODELS else Xte_l
        best_p = results_champ_wc[name]["best_params"]
        m = model.__class__(**{**model.get_params(), **best_p})
        try:
            m.fit(Xtr_, tr_l["champion"].values)
            yp = m.predict_proba(Xte_)[:, 1]
            if len(np.unique(te_l["champion"].values)) == 2:
                loco_full[name].append(roc_auc_score(te_l["champion"].values, yp))
            else:
                loco_full[name].append(np.nan)
        except:
            loco_full[name].append(np.nan)
    cups_used.append(test_cup)

print(f"\nLOCO-CV — {len(cups_used)} folds")
print(f"{'Modelo':<25}  AUC médio ± σ     n_folds_válidos")
print("─" * 55)
loco_summary = {}
for name in MODELS:
    arr   = np.array(loco_full[name], dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) > 0:
        m_, s_ = np.mean(valid), np.std(valid)
        loco_summary[name] = (m_, s_)
        print(f"  {name:<23}  {m_:.3f} ± {s_:.3f}     ({len(valid)}/{len(cups_used)})")

# Plot LOCO
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Leave-One-Cup-Out CV — AUC por Edição (1958–2022)",
             fontsize=11, fontweight='bold')
ax = axes[0]
for (name, aucs_l), color in zip(loco_full.items(), colors_m):
    arr    = np.array(aucs_l, dtype=float)
    valid  = ~np.isnan(arr)
    ax.plot(np.array(cups_used)[valid], arr[valid],
            'o-', label=name[:13], color=color, lw=1.5, ms=5)
ax.axhline(0.5, color='gray', ls='--', alpha=0.5, lw=1)
ax.set_xlabel("Copa deixada de fora"); ax.set_ylabel("AUC-ROC")
ax.set_title("AUC por fold LOCO", fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
ax.set_xticks(cups_used); ax.set_xticklabels([str(c) for c in cups_used], rotation=45, fontsize=8)

ax2 = axes[1]
means_, stds_, names_l = [], [], []
for name in MODELS:
    arr   = np.array(loco_full[name], dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) > 0:
        means_.append(np.mean(valid)); stds_.append(np.std(valid)); names_l.append(name)
for i, (m_, s_, c_) in enumerate(zip(means_, stds_, colors_m)):
    ax2.barh(i, m_, 0.5, color=c_, alpha=0.8)
    ax2.errorbar(m_, i, xerr=s_, fmt='none', color='black', capsize=5, lw=2)
    ax2.text(m_+s_+0.01, i, f"{m_:.3f} ± {s_:.3f}", va='center', fontsize=9)
ax2.set_yticks(range(len(names_l))); ax2.set_yticklabels(names_l, fontsize=9)
ax2.set_xlabel("AUC-ROC"); ax2.axvline(0.5, color='gray', ls='--', alpha=0.5, lw=1)
ax2.set_title("LOCO AUC médio ± σ", fontsize=10, fontweight='bold')
ax2.set_xlim(0.2, 1.15); ax2.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("fig_loco_17cups.png", bbox_inches='tight', dpi=150)
plt.show()

"""## 10d: Threshold Ótimo (F1-max) + PR Curve"""

def best_threshold_f1(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    return thr[best_idx], prec[best_idx], rec[best_idx], f1s[best_idx]

print(f"\nThreshold F1-ótimo vs 0.5 — Champion (test {TEST_FROM}–2022)")
print(f"{'Modelo':<25}  F1@0.5  AP       Thr_opt  F1@opt  Rec@opt")
print("─" * 65)
threshold_results = {}
for name, r in results_champ_wc.items():
    t_opt, p_opt, r_opt, f1_opt = best_threshold_f1(y_true_ch, r["y_prob"])
    ap = average_precision_score(y_true_ch, r["y_prob"])
    threshold_results[name] = (t_opt, p_opt, r_opt, f1_opt, ap)
    print(f"  {name:<23}  {r['f1']:.3f}   {ap:.3f}    {t_opt:.3f}    {f1_opt:.3f}   {r_opt:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Precision-Recall Curves — WC Champion (test {TEST_FROM}–2022)",
             fontsize=11, fontweight='bold')
ax = axes[0]
for (name, r), color in zip(results_champ_wc.items(), colors_m):
    prec_, rec_, _ = precision_recall_curve(y_true_ch, r["y_prob"])
    ap = threshold_results[name][4]
    ax.plot(rec_, prec_, lw=2, color=color, label=f"{name[:13]} (AP={ap:.3f})")
    _, p_opt, r_opt, _, _ = threshold_results[name]
    ax.scatter(r_opt, p_opt, marker='*', s=200, color=color, zorder=5)
ax.axhline(y_true_ch.mean(), color='gray', ls='--', lw=1, alpha=0.6,
           label=f'Baseline (prior={y_true_ch.mean():.2f})')
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("PR Curve (★ = F1-ótimo)", fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax2 = axes[1]; ax2.axis('off')
headers_t = ["Modelo","F1@0.5","AP (PR-AUC)","Thr_opt","F1@opt","Rec@opt"]
rows_t = []
for name, r in results_champ_wc.items():
    t_opt, p_opt, r_opt, f1_opt, ap = threshold_results[name]
    rows_t.append([name[:20], f"{r['f1']:.3f}", f"{ap:.3f}",
                   f"{t_opt:.3f}", f"{f1_opt:.3f}", f"{r_opt:.2f}"])
tab = ax2.table(cellText=rows_t, colLabels=headers_t,
                cellLoc='center', loc='center', bbox=[0,0,1,1])
tab.auto_set_font_size(False); tab.set_fontsize(9)
for j in range(len(headers_t)):
    tab[0,j].set_facecolor('#4472C4')
    tab[0,j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows_t)+1):
    for j in range(len(headers_t)):
        tab[i,j].set_facecolor('#EBF3FB' if i%2==0 else 'white')
ax2.set_title("Métricas @ threshold 0.5 vs F1-ótimo", fontsize=10, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("fig_pr_curve_17cups.png", bbox_inches='tight', dpi=150)
plt.show()

"""## 10e: McNemar Test"""

def run_mcnemar(y_true, pred1, pred2, name1, name2):
    b = int(np.sum((pred1 == y_true) & (pred2 != y_true)))
    c = int(np.sum((pred1 != y_true) & (pred2 == y_true)))
    if b + c == 0:
        print(f"  {name1} vs {name2}: sem discordâncias — N/A")
        return
    table  = [[0, b],[c, 0]]
    result = mcnemar(table, exact=(b + c < 25))
    sig    = "★ p<0.05" if result.pvalue < 0.05 else "(n.s.)"
    print(f"  {name1:<22} vs {name2:<22}  b={b}, c={c}  p={result.pvalue:.4f}  {sig}")

preds = {name: r["y_pred"] for name, r in results_champ_wc.items()}
print(f"\nMcNemar Test — Champion (test {TEST_FROM}–2022)")
print("─" * 72)
pairs = [
    ("SVM",               "Logistic Regression"),
    ("SVM",               "XGBoost (GBM)"),
    ("Logistic Regression","XGBoost (GBM)"),
    ("SVM",               "Random Forest"),
    ("SVM",               "KNN"),
]
for n1, n2 in pairs:
    run_mcnemar(y_true_ch, preds[n1], preds[n2], n1, n2)

"""### 10f: Re-evaluating with a Custom Lower Threshold

To see the effect of a lower classification threshold, we'll re-evaluate the 'champion' prediction models. A lower threshold means that teams with a lower predicted probability will still be classified as potential champions, generally leading to higher recall (catching more true champions) but possibly lower precision (more false positives).
"""

new_threshold = 0.1 # Example: a lower threshold

y_true_ch = test["champion"].values

print(f"\nRe-evaluation with a custom threshold of {new_threshold} — Champion (test {TEST_FROM}–2022)\n")
print(f"{'Modelo':<25}  AUC       Recall    F1 (T=0.1)  F1 (T=0.5)\n")
print("─" * 75)

updated_results_champ_wc = {}

for name, r in results_champ_wc.items():
    # Use the y_prob from the original model training
    y_prob = r["y_prob"]

    # Generate new predictions based on the custom threshold
    y_pred_new_threshold = (y_prob >= new_threshold).astype(int)

    # Calculate new metrics
    new_recall = recall_score(y_true_ch, y_pred_new_threshold, zero_division=0)
    new_f1 = f1_score(y_true_ch, y_pred_new_threshold, zero_division=0)

    updated_results_champ_wc[name] = {
        "model": r["model"],
        "y_prob": y_prob,
        "y_pred": y_pred_new_threshold, # Store new predictions
        "auc": r["auc"],
        "recall_at_new_threshold": new_recall,
        "f1_at_new_threshold": new_f1,
        "f1_at_original_threshold": r["f1"]
    }

    print(f"  {name:<23}  {r['auc']:.4f}    {new_recall:.3f}     {new_f1:.3f}       {r['f1']:.3f}")

print("\nNote: AUC does not change with threshold, as it's a ranking metric. Recall and F1-score are sensitive to the threshold.")

"""## 11: Tabela de Resultados"""

def print_table(results, label):
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":     name,
            "AUC":       round(r["auc"], 3),
            "TP": r["tp"], "FP": r["fp"], "FN": r["fn"], "TN": r["tn"],
            "Recall":    round(r["rec"],  3),
            "Precision": round(r["prec"], 3),
            "F1":        round(r["f1"],   3),
        })
    df_r = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    print(f"\n{'═'*60}  {label}")
    print(df_r.to_string(index=False))
    return df_r

table_final = print_table(results_final_wc, "FINALISTS")
table_champ = print_table(results_champ_wc, "CHAMPION")

"""## 12: ROC Curves"""

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
styles = ["-","--","-.",":",(0,(3,1,1,1))]

for ax, (results, target, title) in zip(axes, [
    (results_final_wc, "reached_final",
     f"ROC — WC Finalists (test {TEST_FROM}–2022)"),
    (results_champ_wc, "champion",
     f"ROC — WC Champion (test {TEST_FROM}–2022)"),
]):
    for (name, r), ls, color in zip(results.items(), styles, colors_m):
        fpr, tpr, _ = roc_curve(test[target], r["y_prob"])
        ax.plot(fpr, tpr, ls=ls, lw=2, color=color,
                label=f"{name} (AUC={r['auc']:.3f})")
    ax.plot([0,1],[0,1],"k--",alpha=0.4,lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_roc_17cups.png", bbox_inches="tight", dpi=150)
plt.show()

"""## 13: Confusion Matrices"""

def plot_cm(results, y_test_series, title, filename):
    nomes = list(results.keys())
    fig, axes = plt.subplots(1, len(nomes), figsize=(4*len(nomes), 4))
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    for ax, name in zip(axes, nomes):
        r  = results[name]
        cm = confusion_matrix(y_test_series, r["y_pred"])
        lbl = np.array([f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}",
                        f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"]).reshape(2,2)
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=["No (0)","Yes (1)"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        for text in disp.text_.ravel(): text.set_visible(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, lbl[i,j], ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()/2 else "black")
        ax.set_title(f"{name}\nAUC={r['auc']:.3f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.show()

plot_cm(results_champ_wc, test["champion"],
        f"Confusion Matrices — WC Champion (threshold=0.5, test {TEST_FROM}–2022)",
        "fig_confusion_champion_17cups.png")

"""## 14: Feature Importance"""

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Feature Importance — WC Finalist Prediction",
             fontsize=12, fontweight="bold")

for ax, name in zip(axes, ["Random Forest", "XGBoost (GBM)"]):
    imp = results_final_wc[name]["model"].feature_importances_
    idx = np.argsort(imp)
    labels = [FEAT_LABELS.get(FEATURE_COLS[i], FEATURE_COLS[i]) for i in idx]
    ax.barh(labels, imp[idx], color="#4C72B0")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Relative importance"); ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("fig_feature_importance_17cups.png", bbox_inches="tight", dpi=150)
plt.show()

"""## 15: Tabela Robustez Consolidada"""

fig, ax = plt.subplots(figsize=(15, 5))
ax.axis('off')
headers = ["Modelo","AUC (test)","95% CI Bootstrap",
           "LOCO AUC","LOCO σ","F1@0.5","F1@ótimo","AP (PR-AUC)"]
rows_data = []
for name, r in results_champ_wc.items():
    m, lo, hi, _ = bootstrap_auc_ci(y_true_ch, r["y_prob"])
    lm, ls = loco_summary.get(name, (np.nan, np.nan))
    t_opt, p_opt, r_opt, f1_opt, ap = threshold_results[name]
    rows_data.append([
        name,
        f"{r['auc']:.3f}",
        f"[{lo:.3f}–{hi:.3f}]",
        f"{lm:.3f}" if not np.isnan(lm) else "N/A",
        f"{ls:.3f}" if not np.isnan(ls) else "N/A",
        f"{r['f1']:.3f}",
        f"{f1_opt:.3f}",
        f"{ap:.3f}",
    ])

tab = ax.table(cellText=rows_data, colLabels=headers,
               cellLoc='center', loc='center', bbox=[0,0,1,1])
tab.auto_set_font_size(False); tab.set_fontsize(9.5)
for j in range(len(headers)):
    tab[0,j].set_facecolor('#1F3864')
    tab[0,j].set_text_props(color='white', fontweight='bold')
best_row = np.argmax([float(r[1]) for r in rows_data])
for i in range(1, len(rows_data)+1):
    bg = '#E8F5E9' if i-1 == best_row else ('#F5F5F5' if i%2==0 else 'white')
    for j in range(len(headers)):
        tab[i,j].set_facecolor(bg)
ax.set_title(
    f"Análise de Robustez Consolidada — WC Champion (test {TEST_FROM}–2022, "
    f"{int(test['champion'].sum())} campeões)\n(Verde = melhor AUC no test set)",
    fontsize=11, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("fig_robustez_consolidada_17cups.png", bbox_inches='tight', dpi=150)
plt.show()

"""## 16: Validação 2022 + Forecast 2026"""

# ─── Validação 2022 com Ensemble Consistente (3 Modelos) ──────────────────
# Garante consistência metodológica: o mesmo pipeline de 3 modelos é usado
# na validação out-of-window (2022) e no forecast prospectivo (2026).

train_val = df[df["cup"] <= 2018].copy()
sc_val = StandardScaler()
X_tv_sc = sc_val.fit_transform(train_val[FEATURE_COLS])

s22 = df[df["cup"] == 2022].copy()
X_s22_sc = sc_val.transform(s22[FEATURE_COLS])

# Definindo os 3 modelos equilibrados (mesmos hiperparâmetros do forecast)
ensemble_2022 = {
    "Logistic Regression": LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=3, class_weight="balanced", random_state=42),
    "SVM": SVC(C=1, kernel="rbf", probability=True, class_weight="balanced", random_state=42)
}

prob_cols_22 = []
for name, model in ensemble_2022.items():
    Xtr = X_tv_sc if name in SCALE_MODELS else train_val[FEATURE_COLS].values
    Xte = X_s22_sc if name in SCALE_MODELS else s22[FEATURE_COLS].values

    model.fit(Xtr, train_val["champion"])
    col = f"p_{name}"
    s22[col] = model.predict_proba(Xte)[:, 1]
    prob_cols_22.append(col)

# Média das probabilidades do ensemble para 2022
s22["prob_avg"] = s22[prob_cols_22].mean(axis=1)
s22["prob_%"] = (s22["prob_avg"] / s22["prob_avg"].sum() * 100).round(2)

rank22 = (s22[["team", "elo", "titles", "prob_%", "champion"]]
          .sort_values("prob_%", ascending=False).reset_index(drop=True))
rank22.index += 1

pos_arg  = rank22[rank22["champion"] == 1].index[0]
prob_arg = rank22[rank22["champion"] == 1]["prob_%"].values[0]

print(f"\n✅ Validação 2022 (Ensemble 3 modelos: LR + RF + SVM):")
print(f"  Argentina (real campeã) → #{pos_arg} com P={prob_arg:.2f}%")
print(rank22[["team", "elo", "titles", "prob_%"]].head(10).to_string())


# ─── Forecast 2026 — Retrain em 1958–2022 ─────────────────────────────────
df_26 = build_cup_dataset(2026, TITLES_2026, include_target=False)
X_all = df[FEATURE_COLS]
sc_final = StandardScaler()
X_all_sc = sc_final.fit_transform(X_all)

# ── APENAS 3 MODELOS EQUILIBRADOS (LR, RF, SVM) ──
# KNN e XGBoost foram excluídos do ensemble final devido ao zero-recall
# em thresholds padrão, conforme diagnosticado nas células de robustez.
BEST_PARAMS = {
    "Logistic Regression": LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=3, class_weight="balanced", random_state=42),
    "SVM": SVC(C=1, kernel="rbf", probability=True, class_weight="balanced", random_state=42),
}

trained = {}
for name, model in BEST_PARAMS.items():
    Xtr = X_all_sc if name in SCALE_MODELS else X_all.values
    model.fit(Xtr, df["champion"])
    trained[name] = model

prob_cols_26 = []
X_26 = df_26[FEATURE_COLS]
X_26_sc = sc_final.transform(X_26)

for name, model in trained.items():
    col = f"p_{name}"
    Xte = X_26_sc if name in SCALE_MODELS else X_26.values
    df_26[col] = model.predict_proba(Xte)[:, 1]
    prob_cols_26.append(col)

# Ensemble: média dos 3 modelos equilibrados
df_26["prob_avg"] = df_26[prob_cols_26].mean(axis=1)
for col in prob_cols_26 + ["prob_avg"]:
    df_26[col] = (df_26[col] / df_26[col].sum() * 100).round(2)

rank26 = (df_26[["team", "prob_avg"] + prob_cols_26]
          .sort_values("prob_avg", ascending=False).reset_index(drop=True))
rank26.index += 1

print(f"\n🏆 WORLD CUP 2026 — Previsão (Ensemble: LR + RF + SVM)")
print(f"{'#':>3}  {'Team':<22}  {'Ensemble':>9}  {'LR':>6}  {'RF':>6}  {'SVM':>6}")
print("-" * 65)
for i, row in rank26.head(15).iterrows():
    lr_col  = [c for c in prob_cols_26 if "Logistic" in c][0]
    rf_col  = [c for c in prob_cols_26 if "Forest" in c][0]
    svm_col = [c for c in prob_cols_26 if "SVM" in c][0]
    print(f"{i:>3}  {row['team']:<22}  {row['prob_avg']:>8.2f}%  "
          f"{row[lr_col]:>5.2f}%  {row[rf_col]:>5.2f}%  {row[svm_col]:>5.2f}%")

rank26.to_csv("wc_2026_forecast_3models.csv", index=False)
print("\n✓ Forecast salvo: wc_2026_forecast_3models.csv")

"""## 17: Forecast Chart"""

top12  = rank26.head(12)
cores  = ["#D4AF37","#C0C0C0","#CD7F32"] + ["#4C72B0"] * 9
lr_col  = [c for c in prob_cols_26 if "Logistic" in c][0]
rf_col  = [c for c in prob_cols_26 if "Forest"   in c][0]
svm_col = [c for c in prob_cols_26 if "SVM"      in c][0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "World Cup 2026 — Pre-Tournament Champion Forecast\n"
    "Features: ELO (goal-diff weighted) + Qualifying (conf. z-score) + WC Titles\n"
    f"Trained on {df['cup'].min()}–2022 ({df['cup'].nunique()} World Cups)",
    fontsize=11, fontweight="bold")

ax = axes[0]
bars = ax.barh(top12["team"][::-1], top12["prob_avg"][::-1],
               color=cores[:len(top12)][::-1])
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_xlabel("Ensemble Champion Probability (%)")
ax.set_title("Ensemble (avg. 3 models) — Top 12", fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(0, top12["prob_avg"].max() * 1.4)

top8 = rank26.head(8)
x    = np.arange(len(top8)); width = 0.15
ax2  = axes[1]
for i, (col, label) in enumerate(zip(
        [lr_col, rf_col, svm_col],
        ["LR","RF","SVM"])):
    ax2.bar(x + i * width, top8[col], width, label=label, alpha=0.85)
ax2.set_xticks(x + width * 1)
ax2.set_xticklabels(top8["team"], rotation=30, ha="right", fontsize=9)
ax2.set_ylabel("Probability (%)"); ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
ax2.set_title("Per-model comparison — Top 8", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("fig_forecast_2026_17cups.png", bbox_inches="tight", dpi=150)
plt.show()
print("\n✓ Todas as figuras geradas.")