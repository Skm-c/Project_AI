from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lower_snake_case."""
    df = df.copy()
    df.columns = [
        re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip().lower())).strip("_")
        for c in df.columns
    ]
    return df


def normalize_text(value: object) -> str:
    """Normalize text for joins by title/artist."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^]]*\]", " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"feat\.|ft\.|featuring", " ", text)
    text = re.sub(r"[^0-9a-zа-яё]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_track_id_from_url(value: object) -> str | None:
    """Extract Spotify track id from open.spotify.com/track/<id> URL."""
    if pd.isna(value):
        return None
    text = str(value)
    match = re.search(r"track/([A-Za-z0-9]{22})", text)
    return match.group(1) if match else None


def numeric_from_messy(value: object) -> float:
    """Convert values like '1,234' to float."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).replace(",", "").strip()
    if text == "":
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def safe_divide(num: pd.Series, den: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Vectorized safe division."""
    result = num / den.replace({0: np.nan})
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def linear_slope(days: Iterable[float], values: Iterable[float]) -> float:
    """Slope of y over days. Returns 0 when there are not enough points."""
    x = np.asarray(list(days), dtype=float)
    y = np.asarray(list(values), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(np.unique(x)) < 2 or len(y) < 2:
        return 0.0
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def get_one_hot_encoder():
    """Create OneHotEncoder compatible with sklearn 1.0+ and 1.2+."""
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
