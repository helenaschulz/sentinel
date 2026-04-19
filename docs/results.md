# Baseline Results

Side-by-side comparison of all baselines. See individual notebooks for full analysis.

---

## Summary Table

| Baseline | Notebook | Val F0.5 | Kaggle Public | Val Events | % Test Flagged | Notes |
|---|---|---|---|---|---|---|
| Isolation Forest | 03 | 0.091 | ≈ 0.00 | 2 / 38 | 0.0% | Score drift: test max < val threshold |
| **PCA Reconstruction** | 04 | **0.770** | **0.522** | **21 / 38** | **6.4%** | Best model |

---

## Metric

**Corrected event-wise F0.5**: `Pr_c = Pr_ew × TNR`, β = 0.5 (precision weighted 2×).

- `Pr_ew = TP_events / (TP_events + FP_pred_segments)` — penalises spurious segments
- `TNR = 1 − fp_samples / N_nominal` — penalises false-alarm density

---

## Notebook 03 — Isolation Forest

| Metric | Value |
|---|---|
| Val F0.5 | 0.091 |
| Kaggle public F0.5 | ≈ 0.00 |
| Val events detected | 2 / 38 |
| % test flagged | 0.0% |
| Predicted segments @ best thr | 2,203 |
| FP predicted segments | 16 |

**Why it fails**: IForest scores rows independently → spiky score curve → thousands of 1-sample
predicted segments → `FP_pred_events` explodes → `Pr_ew ≈ 0.1`. Score distribution shifts between
train and test phases → optimal val threshold (0.628) above all test scores → 0% flagged.

Do not attempt to improve IForest for this metric — the failure is architectural, not tuning.

---

## Notebook 04 — PCA Reconstruction

| Metric | Value |
|---|---|
| Val F0.5 | **0.770** |
| Kaggle public F0.5 | **0.522** |
| Val events detected | 21 / 38 |
| FP predicted segments | 0 |
| `Pr_ew` | 1.000 |
| TNR | 0.855 |
| % test flagged | 6.35% |
| PCA components (k) | 39 (95% variance) |
| Scoring | window-mean MSE, stride=100 |
| Threshold | peak |

**Why it works**: one MSE score per 100-row window is smooth → compact predicted segments →
`FP_pred_events = 0`. No score drift: test score range [0.045, 1.31] overlaps val range at the
threshold (0.110).

**Validation stability (NB 04 §6):** 5-fold temporal CV on training 80% confirms consistent F0.5
across different mission phases. The 80/20 result (0.770) is not a lucky draw.

**Ceiling — 17 / 38 events missed**: every missed event is shorter than 100 rows. Window-mean MSE
dilutes the anomaly signal with surrounding nominal samples. This is the linear model's ceiling;
a non-linear reconstruction model is needed to recover these events.

### Ablation: alternative scoring strategies (NB 04 §9)

All variants use PCA k=39. The root cause of variants B–D failing is **threshold drift** from
train/test distribution shift (mean KS distance = 0.43 across 58 channels; worst: channel_15 KS=0.97).

| Variant | Val F0.5 | Val events | Test flagged | Kaggle public | Notes |
|---|---|---|---|---|---|
| **Baseline** (window_mean, stride=100, peak) | **0.770** | 21 / 38 | 6.4% | **0.522** | recommended |
| A — stride=20, window_mean, peak | 0.770 | 21 / 38 | 6.4% | same | stride irrelevant for window_mean |
| B — per_row/max, peak | 0.698 | 12 / 38 | 0.001% | **0.277** | threshold drifts with KS shift |
| C — per_row/max, knee | 0.667 | 12 / 38 | 0.01% | 0.294 | knee selection doesn't rescue drift |
| D — C + postprocessing | 0.667 | 12 / 38 | 0.01% | same | postprocessing is a no-op here |

Variant B val F0.5 (0.698) looks reasonable but Kaggle collapses to 0.277 — a 60% drop. The per-row
val score max is 19,528 vs test max 124.7; the val-tuned threshold sits above 99.99% of test scores.
Window-mean MSE (baseline) keeps val and test distributions compatible at the threshold.
