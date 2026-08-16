# Motor Insurance Premium Prediction

Predicts motor insurance premiums from policy and vehicle attributes.
Trained on `motor_data11-14lats.csv` — 293,537 real motor insurance policy
records (vehicle attributes, policy dates, insured value, premium).

## Problem framing

- **Target:** `PREMIUM` (continuous, currency units — currency unlabeled in
  source data; filename suggests Latvian lats, pre-2014 euro adoption, but
  this is not confirmed and the model is currency-agnostic regardless).
- **Why not just accuracy:** this is a regression problem. Success is
  measured on **RMSE, MAE, and R²** on the original premium scale, plus the
  over-/under-pricing split, since a pricing model that's right on average
  but wildly wrong on individual policies is not useful for underwriting.
- **No business cost ratio was specified** for over- vs under-pricing, so
  both directions are reported symmetrically rather than optimized toward one.

## Key engineering decisions

- **Group-aware train/test split by `OBJECT_ID`.** The same vehicle
  reappears across multiple renewal years (avg ~2x, up to 9x in this
  dataset). A random split would leak the same vehicle into both train and
  test. Used `GroupShuffleSplit`/`GroupKFold` throughout.
- **Log-transformed target (`log1p(PREMIUM)`).** Raw premium is
  right-skewed (skew ≈ 3.1); log1p brings it close to symmetric.
- **Log-transformed `INSURED_VALUE` as a feature.** The raw column has an
  extreme outlier (max 250M vs. mean ~415K). Left untransformed, this
  produced an ill-conditioned design matrix — plain `LinearRegression`
  produced astronomical, unusable coefficients during model comparison.
  Log-transforming it (as `LOG_INSURED_VALUE`) fixed this; trees are
  invariant to the transform anyway since it's monotonic.
- **`CLAIM_PAID` excluded entirely.** It's known only after a claim occurs,
  which is after the policy — and premium — already exist. Using it as a
  feature would leak the future into a pricing model.
- **`EFFECTIVE_YR` excluded.** Found to be corrupted during inspection (142
  distinct values including garbage like `SR`, `/2`, `IN` instead of clean
  years). Vehicle age is derived from `INSR_BEGIN` and `PROD_YEAR` instead.
- **`INSURED_VALUE == 0` treated as missing, not a real zero** (38.2% of
  rows) — flagged via `INSURED_VALUE_MISSING` rather than silently used as
  a genuine low value.
- **`MAKE` cleaned and bucketed.** 454 distinct raw values including
  known data-entry duplicates (e.g. `BAJAJI` → `BAJAJ`). Values occurring
  ≥100 times in training keep their own category (111 kept); everything
  else buckets to `OTHER`. The vocabulary is learned once at train time and
  saved to `artifacts/feature_names.json` so inference never needs the
  training set again.

## Model comparison

Three-fold group-aware CV, metrics on the original premium scale:

| Model | CV RMSE | CV MAE | CV R² |
|---|---|---|---|
| Ridge Regression | 4760.45 | 2102.22 | 0.721 |
| **Random Forest (winner)** | **3695.80** | **1499.44** | **0.832** |
| HistGradientBoostingRegressor | 3729.63 | 1531.80 | 0.829 |

Plain `LinearRegression` (unregularized) was tried first and discarded —
it was numerically unstable given the one-hot encoded high-cardinality
categoricals, producing unusable coefficients even before the
`INSURED_VALUE` fix. Ridge (L2-regularized) replaced it as the linear
baseline.

`LightGBM`/`XGBoost` were considered but unavailable in the training
environment (no network access to install); `HistGradientBoostingRegressor`
(scikit-learn's native histogram-based GBM, same algorithm family, native
categorical support) was used instead.

**Held-out test set (Random Forest, 100 trees, max_depth=12,
min_samples_leaf=5):**

| Metric | Value |
|---|---|
| RMSE | 3704.19 |
| MAE | 1529.25 |
| R² | 0.837 |
| Mean signed error | -524.41 (slight underpricing bias) |
| % overpriced | 45.4% |
| % underpriced | 54.6% |
| Median absolute % error | 14.2% |

Full comparison and parameters: `artifacts/metrics.json`, `artifacts/params.json`.

## Documentation

`docs/` contains five PDFs documenting the project in depth:

| File | Contents |
|---|---|
| `01_Project_Log.pdf` | The actual sequence of decisions, findings, and course corrections made building this project — including the real numerical debugging story behind the Ridge/log-transform fix. |
| `02_Folder_Structure.pdf` | Annotated directory tree with a file-by-file explanation of purpose and design intent. |
| `03_Reproduction_Guide.pdf` | Step-by-step setup, training, API, testing, and Docker instructions, with expected output at each step. |
| `04_Code_Walkthrough.pdf` | Key design decisions with real code excerpts — the single-prediction-path pattern, the two-pass MAKE vocabulary, the linear model failure, the sanity-check guardrail. |
| `05_Master_Guide.pdf` | Comprehensive reference: cover page, table of contents, full code listing for every source file, an interview Q&A tied to this project's real metrics, and a glossary. |

All figures in these documents are drawn from the actual verified `artifacts/metrics.json` and live test runs — not illustrative placeholders.

## Project structure
Insurance_Premium_Prediction/
├── src/
│ ├── model_loader.py # loads/caches artifacts/model.pkl, reports health
│ ├── predict.py # THE single authoritative preprocessing + inference path
│ └── schemas.py # Pydantic request/response models
├── tests/
│ ├── test_predict.py # unit tests on predict.py, real model, no mocks
│ ├── test_api.py # tests /health, /predict, /predict/batch
│ └── test_smoke.py # hits a live deployed endpoint, skips if none configured
├── conf/
│ ├── dev.yaml / staging.yaml / prod.yaml # env, log level, sanity-check bounds
├── data/
│ ├── holdout_sample.csv # 200 raw held-out rows, safe to commit
│ └── baseline_stats.json # training distribution, for future drift checks
├── artifacts/
│ ├── model.pkl # NOT in git — see Model artifact below
│ ├── metrics.json / params.json / feature_names.json
├── model_training/
│ └── train.py # produces everything in artifacts/ + data/
├── docs/
│ ├── 01_Project_Log.pdf # decisions, findings, and course corrections, in order
│ ├── 02_Folder_Structure.pdf # annotated project layout
│ ├── 03_Reproduction_Guide.pdf # step-by-step setup, train, run, test, Docker
│ ├── 04_Code_Walkthrough.pdf # key design decisions with real code excerpts
│ └── 05_Master_Guide.pdf # cover, TOC, full code reference, interview Q&A, glossary
├── app.py # FastAPI entrypoint — routing only
├── Dockerfile
├── requirements.txt
└── .gitignore / .dockerignore

## Model artifact — GitHub Releases

`artifacts/model.pkl` (~22.7MB) is excluded from git (`.gitignore`) to
avoid permanent binary bloat in git history. It's attached to the
[`v1.0.0` GitHub Release](../../releases/tag/v1.0.0) instead, tagged to the
commit and metrics that produced it.

To get it: download from the Release, or regenerate it yourself:

```powershell
python model_training/train.py --data motor_data11-14lats.csv
```

Takes roughly 3 minutes on a typical machine.

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python model_training/train.py --data motor_data11-14lats.csv
```

Produces `artifacts/model.pkl`, `artifacts/metrics.json`,
`artifacts/params.json`, `artifacts/feature_names.json`,
`data/holdout_sample.csv`, `data/baseline_stats.json`.

## Run the API

```powershell
uvicorn app:app --reload
```

Defaults to `APP_ENV=dev` (`conf/dev.yaml`). Override with:

```powershell
$env:APP_ENV = "prod"
uvicorn app:app --reload
```

**Endpoints:**
- `GET /health` — artifact status, environment, model version
- `POST /predict` — single policy → `{predicted_premium, model_version, flagged}`
- `POST /predict/batch` — `{"records": [...]}` → list of predictions

`flagged` is `true` when a prediction falls outside the environment's
configured sanity bounds (`conf/*.yaml` → `sanity_check`) — a guardrail
for implausible predictions, not a hard rejection. Bounds are placeholders
pending real actuarial limits from underwriting.

Example request:

```json
{
  "SEX": 0,
  "INSR_BEGIN": "08-AUG-13",
  "INSR_END": "07-AUG-14",
  "INSR_TYPE": 1202,
  "INSURED_VALUE": 519755.22,
  "PROD_YEAR": 2007,
  "SEATS_NUM": 4,
  "CARRYING_CAPACITY": 6,
  "TYPE_VEHICLE": "Pick-up",
  "CCM_TON": 3153,
  "MAKE": "NISSAN",
  "USAGE": "Own Goods"
}
```

Date fields (`INSR_BEGIN`, `INSR_END`) must use the same `DD-MON-YY` format
as the training data (e.g. `08-AUG-13`) — this keeps date parsing to one
implementation in the whole system (`src/predict.py`), rather than a
separate translation layer in the API.

## Tests

```powershell
pytest tests/test_predict.py tests/test_api.py -v
```

Both require `artifacts/model.pkl` to exist (run training first).

`test_smoke.py` hits a **live** endpoint and is skipped unless
`SMOKE_TEST_URL` is set:

```powershell
$env:SMOKE_TEST_URL = "http://127.0.0.1:8000"
pytest tests/test_smoke.py -v
```

## Docker

```powershell
docker build -t insurance-premium-api .
docker run -p 8000:8000 insurance-premium-api
```

`artifacts/model.pkl` must exist locally before building — it's copied
into the image from the build context even though it's gitignored.

## Known limitations

- Currency of `PREMIUM`/`INSURED_VALUE` is unconfirmed (unlabeled in
  source data).
- Sanity-check bounds in `conf/*.yaml` are placeholders, not real
  actuarial limits.
- No business cost ratio was available for over- vs under-pricing, so the
  model is optimized on symmetric RMSE, not asymmetric business cost.
- `HistGradientBoostingRegressor` substitutes for LightGBM/XGBoost, which
  were unavailable in the training environment.