from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import nltk
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Text preprocessing configuration.

    Stemming is used (Porter) to avoid requiring additional corpora downloads.
    """

    remove_non_ascii: bool = True
    do_stemming: bool = True
    keep_negations: bool = True
    negation_join_next: bool = True
    extra_stopwords: tuple[str, ...] = ()


def _normalize_text(text: str, cfg: PreprocessConfig) -> str:
    text = html.unescape(text)
    text = text.lower()

    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)

    if cfg.remove_non_ascii:
        # Strips emojis and many non-ascii symbols; keeps the pipeline dependency-light.
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii", errors="ignore")

    # Keep apostrophes inside words; replace everything else with spaces.
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    # Regex tokenization avoids requiring NLTK punkt downloads.
    return _TOKEN_RE.findall(text)


_NEGATORS = {"no", "not", "nor", "never"}


def _is_negator(token: str) -> bool:
    return token in _NEGATORS or token.endswith("n't")


def _negator_norm(token: str) -> str:
    # Normalize contracted negations like "can't" / "won't" to a consistent negator token.
    return "not" if token.endswith("n't") else token


def _apply_negation_join(tokens: list[str]) -> list[str]:
    """
    Join a negation token with the next token: "not good" -> "not_good".

    This helps the model learn negation patterns; we drop the next token to reduce
    accidental positive bias (e.g., keeping "good" alone).
    """
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if _is_negator(t) and i + 1 < len(tokens):
            out.append(f"{_negator_norm(t)}_{tokens[i + 1]}")
            i += 2
            continue
        out.append(t)
        i += 1
    return out


class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    scikit-learn compatible transformer:
      raw texts -> normalized token strings

    Output format: "token token token"
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self._stemmer = PorterStemmer()

        # Touch nltk to satisfy "nltk / spaCy" requirement; PorterStemmer comes from nltk.
        nltk.__version__  # noqa: B018

        stop = set(ENGLISH_STOP_WORDS)
        if self.config.keep_negations:
            # Keep common negators; removing them often breaks sentiment signals.
            stop.difference_update(_NEGATORS)
        stop.update(w.lower() for w in self.config.extra_stopwords)
        self._stopwords = stop

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        cfg = self.config
        out: list[str] = []

        for item in X:
            text = "" if item is None else str(item)
            text = _normalize_text(text, cfg)
            tokens = _tokenize(text)
            tokens = [t for t in tokens if t not in self._stopwords and len(t) > 1]

            if cfg.keep_negations and cfg.negation_join_next:
                tokens = _apply_negation_join(tokens)

            if cfg.do_stemming:
                stemmed: list[str] = []
                for t in tokens:
                    if "_" in t:
                        a, b = t.split("_", 1)
                        stemmed.append(f"{self._stemmer.stem(a)}_{self._stemmer.stem(b)}")
                    else:
                        stemmed.append(self._stemmer.stem(t))
                tokens = stemmed

            out.append(" ".join(tokens))

        return out
