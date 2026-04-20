# SENTINEL

**Based on Kaggle competition**: [ESA-ADB Challenge](https://www.kaggle.com/competitions/esa-adb-challenge)


## Installation

```bash
git clone <repo>
cd sentinel
pip install -r requirements.txt
pip install -e .          # makes `import sentinel` work from any notebook
```

**Data**: place competition files in `data/raw/`:
```
data/raw/
├── train.parquet
├── test.parquet
├── target_channels.csv
└── sample_submission.parquet
```

Raw data download from the [Kaggle competition page](https://www.kaggle.com/competitions/esa-adb-challenge/data).

---

## Running the notebooks

Run in order - each notebook saves artefacts that the next one loads:

| # | Notebook | Output |
|---|---|---|
| 01 | `01-eda.ipynb` | EDA findings |
| 02 | `02-preprocessing.ipynb` | `data/processed/` arrays, `models/robust_scaler.pkl` |
| 03 | `03-baseline_iforest.ipynb` | IForest counter-example (documented failure) |
| 04 | `04-baseline_pca.ipynb` | `submissions/baseline_pca.parquet`, `models/pca_baseline.pkl` |

---

## Project Structure

```
sentinel/
├── data/
│   ├── raw/              # competition data (not committed)
│   └── processed/        # scaled arrays, preprocessing config
├── docs/
│   └── results.md        # baseline comparison and ablation
├── models/               # saved model artefacts (.pkl - not committed)
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-baseline_iforest.ipynb   # IForest - documented counter-example
│   └── 04-baseline_pca.ipynb       # PCA - best submission, val F0.5 = 0.770
├── submissions/
│   ├── baseline_iforest.parquet
│   └── baseline_pca.parquet
├── src/sentinel/
│   ├── __init__.py
│   └── ml_logic/
│       ├── data.py       # data loading, find_anomaly_segments
│       ├── metrics.py    # corrected_event_f05 (official ESA-ADB formula)
│       └── viz.py        # plotting utilities
├── tests/
│   └── test_metrics.py
├── pyproject.toml
└── requirements.txt
```


---

## Dataset

The dataset is 14 years of telemetry from one anonymous ESA satellite, sampled every 30 seconds.

**76 channels** - continuous sensor readings from the spacecraft (temperatures, voltages, attitude, etc. - real names hidden by ESA). Of these, **58 are scored** (listed in `target_channels.csv`), the other 18 are auxiliary.
Channels 4–11 are special: originally monotonic counters (like an odometer), ESA differenced them before publishing, so they hover near zero.

**11 telecommands** - sparse binary flags marking when ground control sent a command to the satellite.
Near-zero activation (< 0.0005%) - operators rarely intervene. Useful as context, not as model features.

**Labels**: `is_anomaly` = 0 (normal) or 1 (anomalous). ~10.5% of samples are anomalous, clustered into
**190 contiguous events** ranging from 1 sample to 116,000+ samples. The anomalies were hand-annotated by
ESA flight control engineers at ESOC in Darmstadt.

**Train** = 14.7M rows (the first ~14 years).
**Test** = 521K rows (the next ~6 months), no labels - you predict these.

---

## Metric

**Corrected event-wise F0.5** (Kotowski et al. 2024 / Sehili et al. 2023)

Reference implementation: [`kplabs-pl/ESA-ADB`](https://github.com/kplabs-pl/ESA-ADB)

```
Pr_c  = Pr_ew × TNR

Pr_ew = TP_events / (TP_events + FP_pred_segments)   ← event-level: penalises spurious segments
TNR   = 1 − fp_samples / N_nominal                    ← sample-level: penalises FP density

F0.5  = 1.25 × Pr_c × Re_e / (0.25 × Pr_c + Re_e)   ← precision weighted 2×
```

Touch one sample per event → TP. Every falsely flagged normal sample hurts precision.
Every spurious contiguous predicted segment (not overlapping any true event) hurts precision separately.

**Optimal strategy**: short, precise detections - avoid false alarms and avoid scattered predictions.

---

## EDA Findings

Full analysis in [`notebooks/01-eda.ipynb`](notebooks/01-eda.ipynb). Key results:

### Dataset Structure
- **14.7M training rows** (≈14 years, 30 s sampling) + **521K test rows** (≈6 months)
- **76 sensor channels** + **11 sparse telecommand flags** - 58 channels scored by the metric
- **Zero missing values** - no imputation needed

### Anomaly Distribution
- **10.5% of samples** anomalous, clustered into **190 events** (1 – 116,061 samples, median 602)
- 80/20 temporal split → **38 val events** (consistent anomaly rate: 10.47% fit / 10.53% val)
- Metric is event-wise: one TP per event suffices, but spurious segments and FP density are penalised heavily

### Channel Characteristics
- **Channels 4–11**: ESA-differenced counters - hover near zero, must **not** be differenced again
- Block correlation structure across 58 scored channels → low-dimensional nominal manifold → PCA is a natural fit
- Anomalies manifest as both **level shifts** (mean separation) and **variance amplification** - both patterns must be captured

### Train vs Test Distribution Shift
- **Mean KS distance = 0.43** across 58 channels - distributions differ substantially between train and test
- Highest-shift channels: channel_15 (KS = 0.97), channel_23 (0.97), channel_38 (0.97)
- Direct cause of the `per_row/max` scoring failure: val score max 19,528 vs test max 124.7 → threshold collapses on test → Kaggle 0.277

### Telecommand Co-occurrence
- All telecommands show lift > 6× inside anomaly windows (most ≈ 9.5×)
- Absolute counts are negligible (≤ 56 firings total) - they are operator *responses* to faults, not causal triggers

---

## Preprocessing

- **Temporal split**: 80/20 by row order (no shuffling - time series)
- **Scaler**: RobustScaler fitted on nominal training rows only (no data leakage)
- **Windows**: 100-row windows; stride=100 for training arrays and inference (non-overlapping)
- **Channels 4–11**: pre-differenced by ESA - not re-differenced
- **Target**: 58 channels from `target_channels.csv` (no hard-coded ranges)

---

## Results

| Model | Val F0.5 | Kaggle Public | Val Events | Notes |
|---|---|---|---|---|
| Isolation Forest (NB 03) | 0.091 | 0.00 | 2 / 38 | Counter-example: spiky per-row scores → FP segment explosion |
| **PCA Reconstruction (NB 04)** | **0.770** | **0.522** | **21 / 38** | Best submission - window-mean MSE, stride=100 |


Metric **corrected event-wise F0.5 (Val F0.5)** is the primary signal. The 80/20 temporal split gives one held-out F0.5; Kaggle public confirms the threshold transfers to test.
5-fold temporal CV (NB 04 §6) confirms the 0.770 number is stable across training phases.

**Ablation (NB 04 §9):** per-row scoring (variant B) yields val F0.5 = 0.698 but collapses to Kaggle 0.277 - threshold drifts with train/test distribution shift (mean KS = 0.43). Window-mean MSE is the only stable strategy.
