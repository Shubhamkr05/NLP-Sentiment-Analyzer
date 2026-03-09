import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from sentiment_regression.data_io import load_dataset_csv
from sentiment_regression.modeling import ModelConfig
from sentiment_regression.prediction import (
    predict_from_csv,
    predict_single_text,
)
from sentiment_regression.training import ThresholdConfig, train_and_save


def _none_or_int(v: str):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"none", "null"}:
        return None
    return int(s)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentiment_regression",
        description="Sentiment score prediction (regression) for social-media text.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a RandomForestRegressor sentiment model.")
    train.add_argument("--data", required=True, help="Path to CSV with text and sentiment_score.")
    train.add_argument("--text-col", default="text", help="Text column name (default: text).")
    train.add_argument(
        "--target-col",
        default="sentiment_score",
        help="Target score column name (default: sentiment_score).",
    )
    train.add_argument("--out", required=True, help="Output directory for model artifacts.")
    train.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction (default: 0.2).")
    train.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42).")
    train.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows used for training (useful for very large CSVs).",
    )
    train.add_argument("--tfidf-max-features", type=int, default=50_000, help="TF-IDF max features (default: 50000).")
    train.add_argument("--svd-components", type=int, default=300, help="SVD components (default: 300).")
    train.add_argument("--rf-n-estimators", type=int, default=400, help="RF trees (default: 400).")
    train.add_argument(
        "--rf-max-depth",
        type=_none_or_int,
        default=None,
        help="RF max depth (int) or 'none' (default: none).",
    )
    train.add_argument("--rf-min-samples-split", type=int, default=2, help="RF min samples split (default: 2).")
    train.add_argument("--rf-min-samples-leaf", type=int, default=1, help="RF min samples leaf (default: 1).")
    train.add_argument("--neutral-band-ratio", type=float, default=0.10, help="Neutral band ratio (default: 0.10).")
    train.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV tuning (slower).")
    train.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations (default: 20).")
    train.add_argument("--cv", type=int, default=3, help="Cross-validation folds (default: 3).")

    pred = sub.add_parser("predict", help="Predict sentiment scores using a saved model.")
    pred.add_argument("--model-dir", required=True, help="Directory containing model.joblib and metadata.json.")
    pred.add_argument("--text", help="Single text input to score.")
    pred.add_argument("--input-csv", help="CSV file to score.")
    pred.add_argument("--text-col", default="text", help="Text column for --input-csv (default: text).")
    pred.add_argument("--output-csv", help="Where to write CSV predictions (optional).")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        df = load_dataset_csv(
            args.data,
            text_col=args.text_col,
            target_col=args.target_col,
            max_rows=args.max_rows,
            random_state=args.random_state,
        )
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_cfg = ModelConfig(
            tfidf_max_features=args.tfidf_max_features,
            svd_components=args.svd_components,
            random_state=args.random_state,
            rf_n_estimators=args.rf_n_estimators,
            rf_max_depth=args.rf_max_depth,
            rf_min_samples_split=args.rf_min_samples_split,
            rf_min_samples_leaf=args.rf_min_samples_leaf,
        )

        result = train_and_save(
            df=df,
            text_col=args.text_col,
            target_col=args.target_col,
            out_dir=out_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            max_rows=args.max_rows,
            tune=args.tune,
            n_iter=args.n_iter,
            cv=args.cv,
            model_cfg=model_cfg,
            threshold_cfg=ThresholdConfig(neutral_band_ratio=float(args.neutral_band_ratio)),
        )

        print(json.dumps(result, indent=2))
        return 0

    if args.command == "predict":
        model_dir = Path(args.model_dir)

        if args.text:
            pred = predict_single_text(model_dir=model_dir, text=args.text)
            print(json.dumps(pred, indent=2))
            return 0

        if args.input_csv:
            df_in = pd.read_csv(args.input_csv)
            df_out = predict_from_csv(model_dir=model_dir, df=df_in, text_col=args.text_col)

            if args.output_csv:
                Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
                df_out.to_csv(args.output_csv, index=False)
                print(f"Wrote predictions to {args.output_csv}")
            else:
                print(df_out.to_string(index=False))
            return 0

        parser.error("Provide either --text or --input-csv for prediction.")

    parser.error("Unknown command.")
    return 2
