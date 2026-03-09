from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sentiment_regression.modeling import ModelConfig, build_pipeline


@dataclass(frozen=True)
class ThresholdConfig:
    """
    Convert continuous scores to {negative, neutral, positive}.

    neutral_band_ratio=0.10 means: neutral region width is 10% of (max_y - min_y),
    centered at the midpoint of the score range.
    """

    neutral_band_ratio: float = 0.10


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _target_summary(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    return {
        "count": int(y.shape[0]),
        "min": float(np.nanmin(y)),
        "p25": float(np.nanpercentile(y, 25)),
        "median": float(np.nanpercentile(y, 50)),
        "p75": float(np.nanpercentile(y, 75)),
        "max": float(np.nanmax(y)),
        "mean": float(np.nanmean(y)),
        "std": float(np.nanstd(y)),
        "frac_lt_0": float(np.mean(y < 0)) if np.isfinite(y).all() else float(np.mean(np.nan_to_num(y) < 0)),
        "frac_gt_0": float(np.mean(y > 0)) if np.isfinite(y).all() else float(np.mean(np.nan_to_num(y) > 0)),
    }


def _param_distributions(random_state: int) -> dict:
    # Parameter space for pipeline tuning.
    return {
        "tfidf__max_features": [20_000, 50_000, 80_000],
        "svd__n_components": [100, 200, 300, 500],
        "rf__n_estimators": [200, 400, 800],
        "rf__max_depth": [None, 10, 20, 40],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ["sqrt", 0.3, 0.5, 1.0],
        "rf__random_state": [random_state],
    }


def train_and_save(
    df: pd.DataFrame,
    text_col: str,
    target_col: str,
    out_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: Optional[int] = None,
    tune: bool = False,
    n_iter: int = 20,
    cv: int = 3,
    model_cfg: Optional[ModelConfig] = None,
    threshold_cfg: Optional[ThresholdConfig] = None,
) -> dict:
    """
    Train a RandomForestRegressor-based sentiment model, evaluate, and save artifacts.

    Artifacts:
      - model.joblib (sklearn pipeline)
      - metadata.json (config + metrics + thresholds + score range)
    """
    model_cfg = model_cfg or ModelConfig(random_state=random_state)
    threshold_cfg = threshold_cfg or ThresholdConfig()

    if max_rows is not None and len(df) > int(max_rows):
        df = df.sample(n=int(max_rows), random_state=random_state).reset_index(drop=True)

    X = df[text_col].astype(str).to_numpy()
    y = df[target_col].astype(float).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline(model_cfg)

    search_info: Optional[dict] = None
    if tune:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=_param_distributions(random_state),
            n_iter=n_iter,
            cv=cv,
            scoring="neg_mean_absolute_error",
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        search_info = {
            "best_params": search.best_params_,
            "best_cv_score_neg_mae": float(search.best_score_),
        }
    else:
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = _evaluate(y_test, y_pred)
    baseline_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=float)
    baseline_metrics = _evaluate(y_test, baseline_pred)

    y_min = float(np.nanmin(y_train))
    y_max = float(np.nanmax(y_train))
    y_train_summary = _target_summary(y_train)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    meta_path = out_dir / "metadata.json"

    joblib.dump(pipeline, model_path)

    metadata = {
        "created_at_utc": _utc_now_iso(),
        "text_col": text_col,
        "target_col": target_col,
        "max_rows": max_rows,
        "target_min_train": y_min,
        "target_max_train": y_max,
        "target_summary_train": y_train_summary,
        "thresholds": asdict(threshold_cfg),
        "model_config": model_cfg.to_dict(),
        "tuning": search_info,
        "metrics_holdout": metrics,
        "metrics_holdout_baseline_mean": baseline_metrics,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model_dir": str(out_dir),
        "artifacts": {"model": str(model_path), "metadata": str(meta_path)},
        "metrics_holdout": metrics,
        "metrics_holdout_baseline_mean": baseline_metrics,
        "target_summary_train": y_train_summary,
        "tuning": search_info,
    }
