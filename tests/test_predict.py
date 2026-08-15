"""
test_predict.py — unit tests on the real prediction logic in src/predict.py.

No mocking: these load the actual trained model.pkl and run real inputs
through clean_and_engineer() + predict_premium(). Requires artifacts/model.pkl
to exist — run model_training/train.py first if these fail with a
FileNotFoundError.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model_loader import load_pipeline
from src.predict import clean_and_engineer, predict_premium


@pytest.fixture
def sample_raw_df():
    """Two realistic rows: one clean, one exercising missing/unknown values."""
    return pd.DataFrame([
        {
            'SEX': 0, 'INSR_BEGIN': '08-AUG-13', 'INSR_END': '07-AUG-14',
            'INSR_TYPE': 1202, 'INSURED_VALUE': 519755.22, 'PROD_YEAR': 2007,
            'SEATS_NUM': 4, 'CARRYING_CAPACITY': 6, 'TYPE_VEHICLE': 'Pick-up',
            'CCM_TON': 3153, 'MAKE': 'NISSAN', 'USAGE': 'Own Goods',
        },
        {
            # missing CARRYING_CAPACITY/SEATS_NUM/CCM_TON, unknown MAKE,
            # INSURED_VALUE == 0 (treated as missing)
            'SEX': 1, 'INSR_BEGIN': '01-JAN-12', 'INSR_END': '31-DEC-12',
            'INSR_TYPE': 1201, 'INSURED_VALUE': 0, 'PROD_YEAR': 2000,
            'SEATS_NUM': None, 'CARRYING_CAPACITY': None, 'TYPE_VEHICLE': 'Truck',
            'CCM_TON': None, 'MAKE': 'SomeRandomBrandXYZ', 'USAGE': 'Private',
        },
    ])


class TestCleanAndEngineer:
    def test_duration_days_computed_correctly(self, sample_raw_df):
        eng = clean_and_engineer(sample_raw_df, known_makes={'NISSAN'})
        assert eng.loc[0, 'DURATION_DAYS'] == 364

    def test_vehicle_age_computed_correctly(self, sample_raw_df):
        eng = clean_and_engineer(sample_raw_df, known_makes={'NISSAN'})
        assert eng.loc[0, 'VEHICLE_AGE'] == 6  # 2013 - 2007

    def test_insured_value_log_transformed(self, sample_raw_df):
        eng = clean_and_engineer(sample_raw_df, known_makes={'NISSAN'})
        assert eng.loc[0, 'INSURED_VALUE_MISSING'] == 0
        assert np.isclose(eng.loc[0, 'LOG_INSURED_VALUE'], np.log1p(519755.22))

    def test_zero_insured_value_flagged_as_missing(self, sample_raw_df):
        eng = clean_and_engineer(sample_raw_df, known_makes={'NISSAN'})
        assert eng.loc[1, 'INSURED_VALUE_MISSING'] == 1
        assert eng.loc[1, 'LOG_INSURED_VALUE'] == 0.0

    def test_missing_carrying_capacity_flagged_not_dropped(self, sample_raw_df):
        eng = clean_and_engineer(sample_raw_df, known_makes={'NISSAN'})
        assert eng.loc[1, 'CARRYING_CAPACITY_MISSING'] == 1
        assert pd.isna(eng.loc[1, 'CARRYING_CAPACITY'])

    def test_known_make_kept_unknown_bucketed_as_other(self, sample_raw_df):
        eng = clean_and_engineer(sample_raw_df, known_makes={'NISSAN'})
        assert eng.loc[0, 'MAKE_BUCKET'] == 'NISSAN'
        assert eng.loc[1, 'MAKE_BUCKET'] == 'OTHER'

    def test_make_alias_normalized(self):
        df = pd.DataFrame([{
            'SEX': 0, 'INSR_BEGIN': '08-AUG-13', 'INSR_END': '07-AUG-14',
            'INSR_TYPE': 1202, 'INSURED_VALUE': 100000, 'PROD_YEAR': 2010,
            'SEATS_NUM': 2, 'CARRYING_CAPACITY': 1, 'TYPE_VEHICLE': 'Motor-cycle',
            'CCM_TON': 150, 'MAKE': 'bajaji', 'USAGE': 'Private',
        }])
        eng = clean_and_engineer(df, known_makes={'BAJAJ'})
        assert eng.loc[0, 'MAKE_BUCKET'] == 'BAJAJ'

    def test_does_not_drop_rows(self, sample_raw_df):
        eng = clean_and_engineer(sample_raw_df, known_makes={'NISSAN'})
        assert len(eng) == len(sample_raw_df)


class TestPredictPremium:
    def test_returns_positive_predictions(self, sample_raw_df):
        pipeline = load_pipeline()
        preds = predict_premium(sample_raw_df, pipeline)
        assert len(preds) == len(sample_raw_df)
        assert (preds > 0).all()

    def test_deterministic(self, sample_raw_df):
        pipeline = load_pipeline()
        preds_1 = predict_premium(sample_raw_df, pipeline)
        preds_2 = predict_premium(sample_raw_df, pipeline)
        assert np.allclose(preds_1, preds_2)

    def test_holdout_sample_predictions_within_reasonable_range(self):
        """Real held-out data should produce predictions in the same order
        of magnitude as actual premiums, not wildly off."""
        pipeline = load_pipeline()
        holdout = pd.read_csv(ROOT / 'data' / 'holdout_sample.csv')
        preds = predict_premium(holdout.drop(columns=['PREMIUM']), pipeline)
        median_actual = holdout['PREMIUM'].median()
        median_pred = np.median(preds)
        # predictions should be in the right ballpark: within 3x of median actual
        assert median_pred < median_actual * 3
        assert median_pred > median_actual / 3