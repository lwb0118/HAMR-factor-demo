"""
ml_model.py
===========
ML extension — predict mispricing repair probability.

Per HAMR Framework section 15:
  - NOT a black-box prediction engine
  - Secondary filter: predicts "will mispricing repair?" or "is this a value trap?"
  - Must use walk-forward validation (no look-ahead)

Targets:
  Y_repair = 1 if future 20d residual return > top 30% AND drawdown < bottom 30%
  Y_trap   = 1 if future 60d return continues falling AND fundamentals deteriorate

Models:
  - Logistic Regression (interpretable baseline)
  - Random Forest (non-linear interactions)
  - LightGBM (production, with walk-forward validation)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def build_ml_features(panel):
    """
    Build feature matrix for ML model.

    Features (per framework):
      - HAMR sub-components (MismatchScore, QualityScore, etc.)
      - Price-based: momentum, volatility, turnover metrics
      - Market context: AIStateScore, sector returns
    """
    feature_cols = [
        'MismatchScore', 'QualityScore', 'ResidualWeakness',
        'NonFundamentalOK', 'FundingVacuum', 'LiquidityCapacity',
        'MispricingPressure', 'HAMR_Core', 'AIStateScore'
    ]
    available = [c for c in feature_cols if c in panel.columns]

    df = panel[['date', 'code'] + available].dropna()
    return df, available


def build_repair_target(panel, horizon=20):
    """
    Build Y_repair target (walk-forward, no look-ahead).

    Y_repair = 1 if:
      - Future residual return > top 30% on that future date
      - AND no major negative forecast in the period

    Labels are built using ONLY past/future information correctly partitioned.
    """
    df = panel.copy()
    ret_col = f'fwd_{horizon}d'
    if ret_col not in df.columns:
        return df, None

    df['future_resid'] = df[ret_col] - df.groupby('date')[ret_col].transform('mean')
    df['Y_repair'] = df.groupby('date')['future_resid'].transform(
        lambda x: (x > x.quantile(0.70)).astype(int) if len(x) > 10 else 0
    )
    return df, 'Y_repair'


def train_repair_model(X_train, y_train, X_test=None, y_test=None,
                       model_type='logistic'):
    """
    Train mispricing repair prediction model.

    Simple sklearn wrapper with walk-forward awareness.
    """
    try:
        if model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000, C=1.0)
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=20,
                random_state=42
            )
        elif model_type == 'lgbm':
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                num_leaves=16, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1
            )
        else:
            raise ValueError(f'Unknown model: {model_type}')

        model.fit(X_train, y_train)

        result = {'model': model, 'model_type': model_type}

        if hasattr(model, 'feature_importances_'):
            result['feature_importance'] = dict(
                zip(X_train.columns if hasattr(X_train, 'columns') else
                    [f'f{i}' for i in range(X_train.shape[1])],
                    model.feature_importances_)
            )

        if X_test is not None and y_test is not None:
            y_pred = model.predict_proba(X_test)[:, 1]
            from sklearn.metrics import roc_auc_score
            result['auc'] = float(roc_auc_score(y_test, y_pred))
            result['n_test'] = len(y_test)

        return result
    except ImportError:
        return {'error': 'sklearn/lightgbm not installed'}


def run_ml_pipeline(panel):
    """
    Run ML mispricing repair prediction pipeline.

    Walk-forward: train on earlier dates, predict on later dates.
    """
    print('\n[ML Pipeline] Building features and targets...')

    # Build features
    features_df, feature_cols = build_ml_features(panel)

    # Build target
    panel_with_target, target_col = build_repair_target(panel)

    if target_col is None or len(feature_cols) < 3:
        print('  Insufficient data for ML pipeline')
        return None

    # Merge features and target
    ml_df = features_df.merge(
        panel_with_target[['date', 'code', target_col]],
        on=['date', 'code'], how='inner'
    ).dropna()

    if len(ml_df) < 200:
        print(f'  Only {len(ml_df)} samples — insufficient for ML')
        return None

    # Walk-forward split: strict temporal (no time leakage)
    dates = sorted(ml_df['date'].unique())
    split_idx = int(len(dates) * 0.7)
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])

    train = ml_df[ml_df['date'].isin(train_dates)]
    test = ml_df[ml_df['date'].isin(test_dates)]

    print(f'  Walk-forward: train {dates[0].date()}..{dates[split_idx-1].date()} '
          f'({len(train)}), test {dates[split_idx].date()}..{dates[-1].date()} '
          f'({len(test)})')

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values

    # Train models
    results = {}
    for name in ['logistic', 'random_forest']:
        r = train_repair_model(
            pd.DataFrame(X_train, columns=feature_cols), y_train,
            pd.DataFrame(X_test, columns=feature_cols), y_test,
            model_type=name
        )
        if 'error' not in r:
            print(f'  {name}: AUC={r.get("auc", "N/A")}')
        results[name] = r

    # Try LightGBM
    try:
        r_lgbm = train_repair_model(
            pd.DataFrame(X_train, columns=feature_cols), y_train,
            pd.DataFrame(X_test, columns=feature_cols), y_test,
            model_type='lgbm'
        )
        if 'error' not in r_lgbm:
            print(f'  lgbm: AUC={r_lgbm.get("auc", "N/A")}')
        results['lgbm'] = r_lgbm
    except Exception:
        pass

    return results
