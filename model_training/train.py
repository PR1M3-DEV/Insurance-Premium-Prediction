"""
train.py — Motor Insurance Premium Prediction

Trains and compares 3 regression models on log1p(PREMIUM), selects a winner,
and exports everything src/predict.py needs to serve predictions:
    artifacts/model.pkl
    artifacts/metrics.json
    artifacts/params.json
    artifacts/feature_names.json
    data/holdout_sample.csv
    data/baseline_stats.json

Run:
    python model_training/train.py --data /path/to/motor_data11-14lats.csv
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.predict import CATEGORICAL_FEATURES, NUMERIC_FEATURES, clean_and_engineer  # noqa: E402

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Data loading + cleaning
# --------------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] raw shape: {df.shape}")
    return df


def clean_training_data(df: pd.DataFrame, min_make_count: int = 100) -> tuple[pd.DataFrame, list]:
    """Cleaning steps that only make sense at TRAIN time (dropping rows,
    learning the MAKE vocabulary). Row-preserving cleaning/feature
    engineering lives in src/predict.py (clean_and_engineer) so train and
    serve share the exact same logic — this function just calls it.

    Returns (cleaned_df, known_makes_list). known_makes_list is saved to
    artifacts/feature_names.json so predict.py can bucket MAKE identically
    at inference time without ever needing the training set again.
    """
    before = len(df)
    df = df.drop_duplicates()
    print(f"[clean] dropped {before - len(df)} exact-duplicate rows")

    before = len(df)
    df = df[df['PREMIUM'] > 0].copy()
    print(f"[clean] dropped {before - len(df)} rows with PREMIUM <= 0")

    # Pass 1: engineer features with an empty vocabulary so MAKE_BUCKET is
    # just the cleaned/normalized MAKE string (uncapped) — lets us learn
    # which makes are frequent enough to deserve their own category.
    df = clean_and_engineer(df, known_makes=set())
    make_counts = df['MAKE_BUCKET'].value_counts()
    known_makes = sorted(make_counts[make_counts >= min_make_count].index.tolist())
    print(f"[clean] MAKE: {df['MAKE_BUCKET'].nunique()} distinct values -> "
          f"{len(known_makes)} kept (>= {min_make_count} occurrences), rest bucketed as OTHER")

    # Pass 2: re-engineer with the learned vocabulary so MAKE_BUCKET now
    # matches exactly what predict.py will produce at inference time.
    df = clean_and_engineer(df, known_makes=set(known_makes))

    before = len(df)
    df = df.dropna(subset=['DURATION_DAYS'])
    df = df[df['DURATION_DAYS'] >= 0]
    print(f"[clean] dropped {before - len(df)} rows with unparseable/negative policy duration")

    print(f"[clean] final training shape: {df.shape}")
    return df, known_makes


# --------------------------------------------------------------------------
# Model pipelines
# --------------------------------------------------------------------------
def build_preprocessor(model_kind: str) -> ColumnTransformer:
    """Different models want categoricals encoded differently:
    - linear model: one-hot (avoids implying false ordering)
    - tree models: ordinal integer codes (trees split on thresholds fine,
      and it avoids an exploding one-hot width from MAKE_BUCKET)
    """
    # Numeric NaNs (CARRYING_CAPACITY, SEATS_NUM, CCM_TON, VEHICLE_AGE can all
    # be missing) are median-imputed here, fit on training folds only —
    # HistGradientBoosting could accept NaN natively, but imputing uniformly
    # keeps all three models on the exact same input matrix, which makes
    # the comparison a fair test of model choice, not input handling.
    if model_kind == 'linear':
        cat_encoder = OneHotEncoder(handle_unknown='ignore', min_frequency=20, sparse_output=False)
        num_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ])
    else:
        cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        num_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
        ])

    return ColumnTransformer(
        transformers=[
            ('num', num_pipe, NUMERIC_FEATURES),
            ('cat', cat_encoder, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict:
    return {
        'ridge_regression': Pipeline([
            ('prep', build_preprocessor('linear')),
            # Plain OLS LinearRegression was numerically unstable here
            # (RMSE exploded to astronomical values during comparison) —
            # the one-hot encoded MAKE_BUCKET/TYPE_VEHICLE/USAGE columns
            # produce a near-singular design matrix. Ridge's L2 penalty
            # fixes this while keeping the model linear/interpretable.
            ('model', Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ]),
        'random_forest': Pipeline([
            ('prep', build_preprocessor('tree')),
            ('model', RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=5,
                n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
        'hist_gradient_boosting': Pipeline([
            ('prep', build_preprocessor('tree')),
            ('model', HistGradientBoostingRegressor(
                max_iter=300, max_depth=8, learning_rate=0.05,
                l2_regularization=1.0, random_state=RANDOM_STATE,
            )),
        ]),
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(y_true_original: np.ndarray, y_pred_original: np.ndarray) -> dict:
    """Metrics on the ORIGINAL premium scale (currency units) — this is what's
    actually interpretable to the business, not the log scale."""
    residual = y_pred_original - y_true_original  # positive = overpriced
    n = len(y_true_original)
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true_original, y_pred_original))),
        'mae': float(mean_absolute_error(y_true_original, y_pred_original)),
        'r2': float(r2_score(y_true_original, y_pred_original)),
        'mean_signed_error': float(residual.mean()),
        'pct_overpriced': float((residual > 0).sum() / n * 100),
        'pct_underpriced': float((residual < 0).sum() / n * 100),
        'median_abs_pct_error': float(
            np.median(np.abs(residual) / np.maximum(y_true_original, 1.0)) * 100
        ),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(data_path: str):
    df = load_raw(data_path)
    df, known_makes = clean_training_data(df)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y_log = np.log1p(df['PREMIUM'].values)
    y_original = df['PREMIUM'].values
    groups = df['OBJECT_ID'].values

    # Group-aware split: same vehicle (OBJECT_ID) never appears in both
    # train and test, since the same vehicle recurs across renewal years.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y_log, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_log, y_test_log = y_log[train_idx], y_log[test_idx]
    y_train_orig, y_test_orig = y_original[train_idx], y_original[test_idx]

    print(f"\n[split] train: {len(X_train)} rows | test: {len(X_test)} rows")
    print(f"[split] unique vehicles train: {df['OBJECT_ID'].iloc[train_idx].nunique()} "
          f"| test: {df['OBJECT_ID'].iloc[test_idx].nunique()} (no overlap by construction)")

    # Group-aware CV on the training set for model comparison
    gkf = GroupKFold(n_splits=3)
    train_groups = df['OBJECT_ID'].iloc[train_idx].values

    models = build_models()
    cv_results = {}
    print("\n[cv] 3-fold group CV (metrics on original premium scale)")
    for name, pipe in models.items():
        fold_rmse, fold_mae, fold_r2 = [], [], []
        for fold_train_i, fold_val_i in gkf.split(X_train, y_train_log, groups=train_groups):
            pipe.fit(X_train.iloc[fold_train_i], y_train_log[fold_train_i])
            pred_log = pipe.predict(X_train.iloc[fold_val_i])
            pred_orig = np.expm1(pred_log)
            true_orig = y_train_orig[fold_val_i]
            fold_rmse.append(np.sqrt(mean_squared_error(true_orig, pred_orig)))
            fold_mae.append(mean_absolute_error(true_orig, pred_orig))
            fold_r2.append(r2_score(true_orig, pred_orig))
        cv_results[name] = {
            'cv_rmse_mean': float(np.mean(fold_rmse)), 'cv_rmse_std': float(np.std(fold_rmse)),
            'cv_mae_mean': float(np.mean(fold_mae)), 'cv_mae_std': float(np.std(fold_mae)),
            'cv_r2_mean': float(np.mean(fold_r2)), 'cv_r2_std': float(np.std(fold_r2)),
        }
        print(f"  {name:24s} RMSE={cv_results[name]['cv_rmse_mean']:>10.2f} "
              f"MAE={cv_results[name]['cv_mae_mean']:>10.2f} "
              f"R2={cv_results[name]['cv_r2_mean']:.4f}")

    winner_name = min(cv_results, key=lambda k: cv_results[k]['cv_rmse_mean'])
    print(f"\n[select] winner by CV RMSE: {winner_name}")

    # Refit winner on FULL training set, evaluate once on held-out test set
    winner = models[winner_name]
    winner.fit(X_train, y_train_log)
    test_pred_log = winner.predict(X_test)
    test_pred_orig = np.expm1(test_pred_log)
    test_metrics = evaluate(y_test_orig, test_pred_orig)
    print(f"\n[test] held-out performance ({winner_name}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Also record test metrics for every candidate, for an honest comparison table
    all_test_metrics = {}
    for name, pipe in models.items():
        if name == winner_name:
            all_test_metrics[name] = test_metrics
            continue
        pipe.fit(X_train, y_train_log)
        pred_orig = np.expm1(pipe.predict(X_test))
        all_test_metrics[name] = evaluate(y_test_orig, pred_orig)

    # --------------------------------------------------------------------
    # Export artifacts
    # --------------------------------------------------------------------
    artifacts_dir = ROOT / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    joblib.dump(winner, artifacts_dir / 'model.pkl')

    metrics_out = {
        'winner': winner_name,
        'cv_comparison': cv_results,
        'test_set_comparison': all_test_metrics,
        'selected_model_test_metrics': test_metrics,
        'n_train_rows': int(len(X_train)),
        'n_test_rows': int(len(X_test)),
        'n_train_vehicles': int(df['OBJECT_ID'].iloc[train_idx].nunique()),
        'n_test_vehicles': int(df['OBJECT_ID'].iloc[test_idx].nunique()),
    }
    with open(artifacts_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)

    params_out = {
        'winner_model': winner_name,
        'target_transform': 'log1p',
        'random_state': RANDOM_STATE,
        'test_size': 0.2,
        'cv_folds': 3,
        'split_strategy': 'GroupShuffleSplit / GroupKFold by OBJECT_ID',
        'model_params': winner.named_steps['model'].get_params(),
    }
    with open(artifacts_dir / 'params.json', 'w') as f:
        json.dump(params_out, f, indent=2, default=str)

    with open(artifacts_dir / 'feature_names.json', 'w') as f:
        json.dump({
            'numeric_features': NUMERIC_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'known_makes': known_makes,
        }, f, indent=2)

    # Holdout sample for eval / smoke tests (small, safe to commit).
    # Stored as RAW input columns (what a real API caller sends), not the
    # engineered features — this way tests exercise the full predict_premium()
    # path (clean_and_engineer + pipeline), not just the sklearn pipeline.
    data_dir = ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    raw_input_cols = [
        'SEX', 'INSR_BEGIN', 'INSR_END', 'INSR_TYPE', 'INSURED_VALUE',
        'PROD_YEAR', 'SEATS_NUM', 'CARRYING_CAPACITY', 'TYPE_VEHICLE',
        'CCM_TON', 'MAKE', 'USAGE', 'PREMIUM',
    ]
    holdout_sample = df.iloc[test_idx].sample(n=min(200, len(test_idx)), random_state=RANDOM_STATE)
    holdout_sample[raw_input_cols].to_csv(data_dir / 'holdout_sample.csv', index=False)

    # Baseline stats (training distribution) for drift checks
    baseline_stats = {'numeric': {}, 'categorical': {}}
    for col in NUMERIC_FEATURES:
        baseline_stats['numeric'][col] = {
            'mean': float(X_train[col].mean()), 'std': float(X_train[col].std()),
            'min': float(X_train[col].min()), 'max': float(X_train[col].max()),
            'p50': float(X_train[col].median()),
        }
    for col in CATEGORICAL_FEATURES:
        baseline_stats['categorical'][col] = X_train[col].value_counts(normalize=True).head(20).to_dict()
    baseline_stats['target'] = {
        'mean': float(y_train_orig.mean()), 'std': float(y_train_orig.std()),
        'p50': float(np.median(y_train_orig)),
        'p95': float(np.percentile(y_train_orig, 95)),
    }
    with open(data_dir / 'baseline_stats.json', 'w') as f:
        json.dump(baseline_stats, f, indent=2)

    print(f"\n[export] artifacts written to {artifacts_dir}")
    print(f"[export] holdout_sample.csv + baseline_stats.json written to {data_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT / 'motor_data11-14lats.csv'))
    args = parser.parse_args()
    main(args.data)