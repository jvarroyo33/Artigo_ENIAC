# A Supervised Machine Learning Framework for Sports Championship Prediction
**Supplementary Material for ENIAC 2026 (Double-Blind Submission)**

This repository contains the complete source code, data preprocessing pipelines, and computational notebooks used to generate all figures, tables, and metrics presented in the paper *"A Supervised Machine Learning Framework for Sports Championship Prediction: Case Studies in the NBA and FIFA World Cup"*.

## 📂 Repository Structure
- `nba_eniac_2026.py`: Complete pipeline for Case Study 1 (NBA 2000–2025), including feature engineering, temporal splitting, model training, probability calibration, statistical validation (DeLong, McNemar, bootstrap 95% CI, walk-forward analysis), and the 2026 forecast.
- `worldcup_pretournament_17copas.py`: Complete pipeline for Case Study 2 (FIFA World Cup 1958–2022). Includes ELO computation, confederation-level z-score normalization, and robustness analyses (Bootstrap 95% CI, Leave-One-Cup-Out CV, DeLong test, McNemar's test, ranking evaluation).
- `supplementary_figures/nba/` and `supplementary_figures/world_cup/`: Figures referenced in the paper as supplementary material — exploratory feature distributions/correlations, confusion matrices, calibration diagrams, LOCO-CV plots, and the full 2026 forecast charts. Each is cited from the corresponding section of the paper.
- `requierements.txt`: Python dependencies required to run the scripts.

Both scripts were exported from the original Colab notebooks (`NBA-ENIAC-2026.ipynb`, `WorldCup_PreTournament_17Copas.ipynb`) and are runnable standalone via `python3 <script>.py`.

### Reproducibility note
While preparing the ENIAC 2026 camera-ready version, a non-determinism bug was found
and fixed in the World Cup pipeline: `calc_qual_features()` iterated over a Python
`set` of team names, whose order depends on the process's hash seed and is not fixed
by `PYTHONHASHSEED` set mid-script. This made tree-based/kernel model results
(Random Forest, XGBoost, SVM, KNN) vary slightly between runs, even with fixed random
seeds — Logistic Regression was unaffected, since its convex optimum doesn't depend on
row order. The fix (`sorted(team_set)`, plus `n_jobs=1` and single-threaded BLAS env
vars) makes both pipelines fully deterministic — verified by running each twice and
diffing the full output logs (0 differences). **All numbers in the published paper
reflect the corrected, deterministic pipeline.** See `CHANGELOG_COMPRESSAO.md` in this
repository for the full before/after numbers that changed as a result.

## ⚙️ Requirements & Setup
The code was developed and tested in **Python 3.10+** (Google Colab environment). 
To install the required dependencies locally, run:

```bash
pip install -r requirements.txt
```

## 📊 Data Acquisition
The notebooks are designed to fetch the datasets automatically, but require minimal setup for the NBA data due to Kaggle's API restrictions.

### 1. NBA Dataset (Case Study 1)
The NBA data is hosted on Kaggle. To download it automatically via the notebook, you must provide your own Kaggle API credentials.
1. Go to your Kaggle account settings and create a new API token (this downloads a `kaggle.json` file).
2. In the `NBA-ENIAC-2026.ipynb` notebook, locate the **"Configure Kaggle credentials"** cell.
3. Replace the placeholder with your credentials:
   ```python
   os.environ["KAGGLE_API_TOKEN"] = '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}'
   ```
*(Note: For the double-blind review, the authors' credentials have been redacted from the submitted notebooks).*

### 2. FIFA World Cup Dataset (Case Study 2)
The international football results dataset is hosted on GitHub (`martj42/international_results`). 
- The notebook automatically downloads the exact commit hash used in this study via `urllib`. No manual setup or API keys are required.

## 🚀 How to Reproduce the Experiments
To ensure exact reproducibility, all random seeds are fixed globally at the beginning of each notebook (`SEED = 42`). 

1. **Open the Notebooks:** It is highly recommended to run these notebooks in **Google Colab** to avoid local environment issues.
2. **Run NBA Notebook:** Execute all cells in `NBA-ENIAC-2026.ipynb`. It will generate the NBA 2026 championship probabilities.
3. **Run World Cup Notebook:** Execute all cells in `WorldCup_PreTournament_17Copas.ipynb`. It will compute the ELO ratings, train the models, run the LOCO-CV and Bootstrap robustness checks, and generate the 2026 World Cup forecast.

## 📈 Key Methodological Highlights (Code Implementation)
- **Temporal Anti-Leakage:** Strict chronological splitting (e.g., NBA Train: 2000–2018, Test: 2019–2025). `TimeSeriesSplit` is used for internal cross-validation.
- **Confederation Normalization:** The World Cup pipeline applies z-scoring *per confederation* to correct structural biases (e.g., CONMEBOL vs. UEFA qualifying formats).
- **Threshold Calibration:** Implementation of F1-maximizing thresholds via Precision-Recall curves to address severe class imbalance.
- **Robustness Statistics:** Bootstrap (n=2000) confidence intervals for AUC and Leave-One-Cup-Out (LOCO-CV) cross-validation.

---
*For questions regarding the methodology or code, please refer to the main manuscript submitted to ENIAC 2026.*
```