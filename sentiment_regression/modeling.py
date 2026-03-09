from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sentiment_regression.preprocessing import PreprocessConfig, TextNormalizer


class AdaptiveTruncatedSVD(BaseEstimator, TransformerMixin):
    """
    TruncatedSVD wrapper that automatically reduces n_components to fit the data.

    This prevents errors on small datasets where TF-IDF may have fewer features than
    the configured n_components.
    """

    def __init__(self, n_components: int = 300, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):  # noqa: N803
        n_features = int(getattr(X, "shape")[1])
        requested = int(self.n_components)
        # Keep at least 1 component, and stay within sklearn's constraints.
        n_components = min(requested, max(1, n_features - 1))

        self.n_components_ = n_components
        self._svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self._svd.fit(X)
        return self

    def transform(self, X):  # noqa: N803
        return self._svd.transform(X)


@dataclass(frozen=True)
class ModelConfig:
    """
    Default modeling configuration.

    Notes:
      - TF-IDF uses unigrams + bigrams (ngram_range=(1,2)).
      - TruncatedSVD reduces sparse TF-IDF to a dense low-dimensional space.
      - RandomForestRegressor is the primary regression model.
    """

    preprocess: PreprocessConfig = PreprocessConfig()
    tfidf_max_features: int = 50_000
    svd_components: int = 300
    random_state: int = 42

    rf_n_estimators: int = 400
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_n_jobs: int = -1

    def to_dict(self) -> dict:
        return asdict(self)


def build_pipeline(cfg: ModelConfig) -> Pipeline:
    """
    Build an end-to-end sklearn pipeline:
      TextNormalizer -> TF-IDF (1-2 grams) -> SVD -> RandomForestRegressor
    """
    return Pipeline(
        steps=[
            ("prep", TextNormalizer(cfg.preprocess)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=cfg.tfidf_max_features,
                    lowercase=False,  # already done in preprocessing
                ),
            ),
            ("svd", AdaptiveTruncatedSVD(n_components=cfg.svd_components, random_state=cfg.random_state)),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=cfg.rf_n_estimators,
                    max_depth=cfg.rf_max_depth,
                    min_samples_split=cfg.rf_min_samples_split,
                    min_samples_leaf=cfg.rf_min_samples_leaf,
                    random_state=cfg.random_state,
                    n_jobs=cfg.rf_n_jobs,
                ),
            ),
        ]
    )
