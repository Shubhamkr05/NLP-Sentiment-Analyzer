from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def _read_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read CSV with a practical encoding fallback (common for social datasets).
    """
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="strict", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", encoding_errors="replace", **kwargs)


def _map_sentiment140_polarity_to_score(polarity: pd.Series) -> pd.Series:
    pol = pd.to_numeric(polarity, errors="coerce")
    uniq = set(pol.dropna().unique().tolist())

    # Common Sentiment140 encoding: 0=negative, 4=positive, sometimes 2=neutral.
    if uniq.issubset({0, 2, 4}):
        return pol.map({0: -1.0, 2: 0.0, 4: 1.0})

    # Some variants may use 0/1.
    if uniq.issubset({0, 1}):
        return pol.map({0: -1.0, 1: 1.0})

    # Otherwise, assume it's already a numeric sentiment score.
    return pol.astype(float)


def _rebalance_scores(
    df: pd.DataFrame,
    target_col: str,
    random_state: int,
) -> pd.DataFrame:
    """
    Rebalance class-like sentiment targets (-1/0/+1) via downsampling.
    """
    if target_col not in df.columns or df.empty:
        return df

    y = pd.to_numeric(df[target_col], errors="coerce")
    uniq = sorted(set(y.dropna().unique().tolist()))
    if not uniq or not set(uniq).issubset({-1.0, 0.0, 1.0}):
        return df

    groups = []
    for cls in uniq:
        g = df[y == cls]
        if not g.empty:
            groups.append(g)
    if len(groups) < 2:
        return df

    min_count = min(len(g) for g in groups)
    if min_count <= 0:
        return df

    balanced = pd.concat(
        [g.sample(n=min_count, random_state=random_state) if len(g) > min_count else g for g in groups],
        ignore_index=True,
    )
    return balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_dataset_csv(
    path: Union[str, Path],
    text_col: str = "text",
    target_col: str = "sentiment_score",
    max_rows: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load a CSV dataset with at least [text_col, target_col].

    Expected:
      - text_col: raw social media text (str)
      - target_col: sentiment score (float), e.g. -1..+1 or 0..1
    """
    if max_rows is not None and int(max_rows) <= 0:
        raise ValueError("--max-rows must be a positive integer.")

    # Read only a small header sample to decide which loader path to take.
    head = _read_csv(path, nrows=5)

    inferred_sentiment140 = False

    # Standard expected format: explicit columns "text" + "sentiment_score".
    if text_col in head.columns and target_col in head.columns:
        if max_rows is None:
            df = _read_csv(path, usecols=[text_col, target_col])
            df[text_col] = df[text_col].astype(str)
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df = df.dropna(subset=[target_col])
        else:
            # Stream + reservoir sample for large CSVs.
            k = int(max_rows)
            rng = np.random.default_rng(int(random_state))
            sample_texts: list[str] = []
            sample_scores: list[float] = []
            seen = 0

            for chunk in _read_csv(path, usecols=[text_col, target_col], chunksize=50_000):
                texts = chunk[text_col].astype(str).to_numpy()
                scores = pd.to_numeric(chunk[target_col], errors="coerce").to_numpy()
                for t, s in zip(texts, scores):
                    if s is None or (isinstance(s, float) and np.isnan(s)):
                        continue
                    seen += 1
                    if len(sample_texts) < k:
                        sample_texts.append(str(t))
                        sample_scores.append(float(s))
                        continue
                    j = int(rng.integers(0, seen))
                    if j < k:
                        sample_texts[j] = str(t)
                        sample_scores[j] = float(s)

            df = pd.DataFrame({text_col: sample_texts, target_col: sample_scores})
    else:
        inferred_sentiment140 = True
        # Fallback: support Sentiment140-style CSVs (often 6 columns):
        # [polarity, id, date, query, user, text]
        if head.shape[1] < 2:
            raise ValueError(
                f"Missing required column(s): {[text_col, target_col]}. "
                f"Found: {list(head.columns)} and could not infer a fallback format."
            )

        if max_rows is None:
            df_full = _read_csv(path)
            text_series = df_full.iloc[:, -1].astype(str)
            score_series = _map_sentiment140_polarity_to_score(df_full.iloc[:, 0])

            df = pd.DataFrame({text_col: text_series, target_col: score_series})
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df = df.dropna(subset=[target_col])
        else:
            # Stream + reservoir sample to avoid loading very large CSVs into memory.
            k = int(max_rows)
            rng = np.random.default_rng(int(random_state))

            sample_texts: list[str] = []
            sample_scores: list[float] = []
            seen = 0

            for chunk in _read_csv(path, chunksize=50_000):
                if chunk.shape[1] < 2:
                    continue
                texts = chunk.iloc[:, -1].astype(str).to_numpy()
                scores = _map_sentiment140_polarity_to_score(chunk.iloc[:, 0]).to_numpy()

                for t, s in zip(texts, scores):
                    if s is None or (isinstance(s, float) and np.isnan(s)):
                        continue
                    seen += 1
                    if len(sample_texts) < k:
                        sample_texts.append(str(t))
                        sample_scores.append(float(s))
                        continue
                    j = int(rng.integers(0, seen))
                    if j < k:
                        sample_texts[j] = str(t)
                        sample_scores[j] = float(s)

            df = pd.DataFrame({text_col: sample_texts, target_col: sample_scores})

    if df.empty:
        raise ValueError("Dataset is empty after cleaning target values.")

    # The bundled Sentiment140 file in this project can be class-imbalanced.
    # Rebalance inferred {-1,0,+1} labels to reduce constant-label predictions.
    if inferred_sentiment140:
        df = _rebalance_scores(df=df, target_col=target_col, random_state=random_state)

    return df
