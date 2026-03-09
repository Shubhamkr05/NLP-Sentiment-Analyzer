from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _load_metadata(model_dir: Path) -> dict:
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {model_dir}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_model(model_dir: Path):
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.joblib in {model_dir}")
    model = joblib.load(model_path)
    # Force serial inference to avoid multiprocessing/thread-pool permission
    # issues in restricted environments.
    try:
        model.set_params(rf__n_jobs=1)
    except Exception:
        if hasattr(model, "n_jobs"):
            try:
                model.n_jobs = 1
            except Exception:
                pass
    return model


def _score_to_label(score: float, min_y: float, max_y: float, neutral_band_ratio: float) -> str:
    mid = (min_y + max_y) / 2.0
    band = (max_y - min_y) * float(neutral_band_ratio)
    lo = mid - band / 2.0
    hi = mid + band / 2.0

    if score < lo:
        return "negative"
    if score > hi:
        return "positive"
    return "neutral"


def predict_single_text(model_dir: Path, text: str) -> dict:
    meta = _load_metadata(model_dir)
    model = _load_model(model_dir)

    score_raw = float(model.predict([text])[0])
    score = float(np.clip(score_raw, float(meta["target_min_train"]), float(meta["target_max_train"])))
    label = _score_to_label(
        score=score,
        min_y=float(meta["target_min_train"]),
        max_y=float(meta["target_max_train"]),
        neutral_band_ratio=float(meta["thresholds"]["neutral_band_ratio"]),
    )

    return {"text": text, "predicted_score": score, "predicted_score_raw": score_raw, "label": label}


def predict_from_csv(model_dir: Path, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    meta = _load_metadata(model_dir)
    model = _load_model(model_dir)

    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}'. Found: {list(df.columns)}")

    texts = df[text_col].astype(str).tolist()
    scores_raw = model.predict(texts).astype(float)

    min_y = float(meta["target_min_train"])
    max_y = float(meta["target_max_train"])
    neutral_band_ratio = float(meta["thresholds"]["neutral_band_ratio"])

    scores = np.clip(scores_raw, min_y, max_y).astype(float)
    labels = [_score_to_label(s, min_y=min_y, max_y=max_y, neutral_band_ratio=neutral_band_ratio) for s in scores]

    out = df.copy()
    out["predicted_score"] = scores
    out["predicted_score_raw"] = scores_raw
    out["predicted_label"] = labels
    return out
