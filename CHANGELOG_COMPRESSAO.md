# Changelog — ENIAC 2026 camera-ready revision

This note documents a reproducibility bug found and fixed in the World Cup pipeline
while preparing the paper's compressed camera-ready version, and the resulting changes
to numbers reported in the paper. For the full page-by-page compression log (what was
cut from the paper and moved here as supplementary material), see
`RELATORIO_ALTERACOES.md` in the paper's own source folder.

## Bug: non-deterministic model results in the World Cup pipeline

`worldcup_pretournament_17copas.py`'s `calc_qual_features()` iterated over a Python
`set()` of team names. Set iteration order in CPython depends on the process's hash
seed, which is randomized per run unless `PYTHONHASHSEED` is fixed **before** the
interpreter starts — the script's `os.environ["PYTHONHASHSEED"] = ...` line runs too
late to have any effect. This made the row order of the training data vary between
runs, which in turn changed GridSearchCV's selected hyperparameters and final metrics
for every model except Logistic Regression (whose convex optimum doesn't depend on row
order).

**Fix:** `for team in team_set:` → `for team in sorted(team_set):` (`calc_qual_features`,
~line 457), plus `n_jobs=1` on both `GridSearchCV` calls and single-threaded BLAS
env vars (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
MKL_NUM_THREADS=1`) to eliminate floating-point non-determinism from parallel
execution as a secondary source of drift.

**Verification:** ran the corrected script twice with `PYTHONHASHSEED=42` and the
env vars above; `diff` between the two complete stdout logs returned 0 lines
(byte-identical). The NBA pipeline (`nba_eniac_2026.py`) was already deterministic —
two independent runs also diffed to 0 lines, and matched the AUC/CI/DeLong/McNemar
values already published in the paper exactly.

## What changed in the paper as a result

| Metric | Old (non-reproducible) | New (deterministic, verified) |
|---|---|---|
| Best WC champion AUC | Random Forest, 0.943 | Random Forest **and SVM tied**, 0.921 |
| SVM Recall (WC champion, threshold 0.5) | 0.000 | **1.000** |
| DeLong RF vs. LR (WC champion) | p=0.041 (significant) | p=0.384 (not significant) |
| LOCO-CV mean AUC (WC champion, 17 folds) | 0.931 ± 0.018 | RF: 0.810 ± 0.162 / LR: 0.843 ± 0.169 |
| 2026 WC forecast order | France, Brazil, Argentina, Germany | France, Argentina, Brazil, Germany |

Also newly implemented (previously claimed in the paper's text/tables without
corresponding code): a DeLong test for AUC comparison in both pipelines, and
bootstrap-CI + McNemar + walk-forward analysis for the NBA pipeline (the World Cup
pipeline already had these). All statistics now printed by the scripts match the
values published in the paper.

## Supplementary figures

`supplementary_figures/world_cup/*` were regenerated from the corrected, deterministic
pipeline and are up to date. `supplementary_figures/nba/*` did not need regeneration
(pipeline was already deterministic and matched the paper).
