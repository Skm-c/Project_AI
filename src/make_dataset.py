from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (
    ensure_dir,
    extract_track_id_from_url,
    linear_slope,
    normalize_text,
    numeric_from_messy,
    safe_divide,
    snake_case_columns,
)


AUDIO_NUMERIC_COLUMNS = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

AUDIO_CATEGORICAL_COLUMNS = [
    "explicit",
    "track_genre",
]


def _pick_first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_charts(path: str | Path, region: str = "Global", top200_only: bool = True) -> pd.DataFrame:
    """Load and normalize Spotify chart data."""
    df = pd.read_csv(path, dtype={"track_id": str})
    df = snake_case_columns(df)

    columns = set(df.columns)

    rename = {}
    title_col = _pick_first_existing(columns, ["title", "track_name", "song", "name"])
    artist_col = _pick_first_existing(columns, ["artist", "artists", "artist_name"])
    rank_col = _pick_first_existing(columns, ["rank", "position"])
    date_col = _pick_first_existing(columns, ["date", "chart_date"])
    streams_col = _pick_first_existing(columns, ["streams", "stream"])
    region_col = _pick_first_existing(columns, ["region", "country"])

    if title_col and title_col != "track_name":
        rename[title_col] = "track_name"
    if artist_col and artist_col != "artist_name":
        rename[artist_col] = "artist_name"
    if rank_col and rank_col != "rank":
        rename[rank_col] = "rank"
    if date_col and date_col != "date":
        rename[date_col] = "date"
    if streams_col and streams_col != "streams":
        rename[streams_col] = "streams"
    if region_col and region_col != "region":
        rename[region_col] = "region"

    df = df.rename(columns=rename)

    required = ["track_name", "artist_name", "rank", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Chart CSV lacks required columns after normalization: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if "region" not in df.columns:
        df["region"] = "Global"

    if "streams" not in df.columns:
        df["streams"] = np.nan

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rank"] = df["rank"].map(numeric_from_messy)
    df["streams"] = df["streams"].map(numeric_from_messy)

    if "url" in df.columns and "track_id" not in df.columns:
        df["track_id"] = df["url"].map(extract_track_id_from_url)
    elif "track_id" not in df.columns:
        df["track_id"] = None
    else:
        df["track_id"] = df["track_id"].astype("string")

    if top200_only and "chart" in df.columns:
        df = df[df["chart"].astype(str).str.lower().str.contains("top", na=False)].copy()

    if region.lower() != "all":
        df = df[df["region"].astype(str).str.lower() == region.lower()].copy()

    df = df.dropna(subset=["date", "track_name", "artist_name", "rank"])
    df["track_norm"] = df["track_name"].map(normalize_text)
    df["artist_norm"] = df["artist_name"].map(normalize_text)
    df["text_key"] = df["track_norm"] + "__" + df["artist_norm"]
    df["track_key"] = df["track_id"].fillna("")
    df.loc[df["track_key"].eq(""), "track_key"] = df.loc[df["track_key"].eq(""), "text_key"]

    return df


def load_audio_features(path: str | Path | None) -> pd.DataFrame | None:
    """Load and normalize audio features dataset."""
    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        return None

    df = pd.read_csv(path, dtype={"track_id": str})
    df = snake_case_columns(df)

    columns = set(df.columns)
    rename = {}

    track_name_col = _pick_first_existing(columns, ["track_name", "title", "song", "name"])
    artist_col = _pick_first_existing(columns, ["artists", "artist", "artist_name"])

    if track_name_col and track_name_col != "track_name":
        rename[track_name_col] = "track_name"
    if artist_col and artist_col != "artist_name":
        rename[artist_col] = "artist_name"

    df = df.rename(columns=rename)

    if "track_name" not in df.columns or "artist_name" not in df.columns:
        raise ValueError(
            "Audio CSV should contain track and artist columns. "
            f"Available columns: {list(df.columns)}"
        )

    if "track_id" not in df.columns:
        df["track_id"] = None
    else:
        df["track_id"] = df["track_id"].astype("string")

    df["track_norm"] = df["track_name"].map(normalize_text)
    df["artist_norm"] = df["artist_name"].map(normalize_text)
    df["text_key"] = df["track_norm"] + "__" + df["artist_norm"]

    for col in AUDIO_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    existing_numeric = [c for c in AUDIO_NUMERIC_COLUMNS if c in df.columns]
    existing_cat = [c for c in AUDIO_CATEGORICAL_COLUMNS if c in df.columns]

    keep_cols = ["track_id", "text_key"] + existing_numeric + existing_cat
    df = df[keep_cols].copy()

    # Aggregate duplicates. Some Kaggle datasets contain one row per track-genre pair.
    agg_spec = {c: "mean" for c in existing_numeric}
    agg_spec.update({c: "first" for c in existing_cat})

    by_text = df.groupby("text_key", as_index=False).agg(agg_spec)
    by_text = by_text.rename(columns={c: f"audio_{c}" for c in existing_numeric + existing_cat})

    by_id = None
    valid_id = df["track_id"].notna() & df["track_id"].astype(str).str.len().gt(0)
    if valid_id.any():
        by_id = df.loc[valid_id].groupby("track_id", as_index=False).agg(agg_spec)
        by_id = by_id.rename(columns={c: f"audio_{c}" for c in existing_numeric + existing_cat})

    return {"by_text": by_text, "by_id": by_id, "audio_columns": [f"audio_{c}" for c in existing_numeric + existing_cat]}


def _build_observation_features(
    charts: pd.DataFrame,
    observation_days: int,
    horizon_days: int,
    target_rank: int,
) -> pd.DataFrame:
    group_cols = ["track_key", "region"]

    first_dates = (
        charts.groupby(group_cols, as_index=False)["date"]
        .min()
        .rename(columns={"date": "first_date"})
    )
    df = charts.merge(first_dates, on=group_cols, how="left")
    df["days_since_first"] = (df["date"] - df["first_date"]).dt.days

    obs = df[(df["days_since_first"] >= 0) & (df["days_since_first"] < observation_days)].copy()
    future = df[
        (df["days_since_first"] >= observation_days)
        & (df["days_since_first"] < observation_days + horizon_days)
    ].copy()

    if obs.empty:
        raise ValueError("No observations in the selected observation window.")

    obs_sorted = obs.sort_values(group_cols + ["date"])
    first_rows = (
        obs_sorted.groupby(group_cols, as_index=False)
        .first()[
            group_cols
            + [
                "date",
                "track_name",
                "artist_name",
                "track_id",
                "text_key",
                "track_norm",
                "artist_norm",
                "rank",
                "streams",
                "first_date",
            ]
        ]
        .rename(
            columns={
                "date": "first_observed_date",
                "rank": "first_rank",
                "streams": "first_streams",
            }
        )
    )

    last_rows = (
        obs_sorted.groupby(group_cols, as_index=False)
        .last()[group_cols + ["rank", "streams"]]
        .rename(columns={"rank": "last_rank_obs", "streams": "last_streams_obs"})
    )

    agg = (
        obs.groupby(group_cols)
        .agg(
            obs_days_count=("date", "nunique"),
            best_rank_7d=("rank", "min"),
            worst_rank_7d=("rank", "max"),
            mean_rank_7d=("rank", "mean"),
            median_rank_7d=("rank", "median"),
            std_rank_7d=("rank", "std"),
            total_streams_7d=("streams", "sum"),
            mean_streams_7d=("streams", "mean"),
            max_streams_7d=("streams", "max"),
            std_streams_7d=("streams", "std"),
        )
        .reset_index()
    )

    slopes = (
        obs.groupby(group_cols)
        .apply(
            lambda g: pd.Series(
                {
                    "rank_slope_7d": linear_slope(g["days_since_first"], g["rank"]),
                    "streams_slope_7d": linear_slope(g["days_since_first"], g["streams"]),
                }
            )
        )
        .reset_index()
    )

    features = first_rows.merge(last_rows, on=group_cols, how="left")
    features = features.merge(agg, on=group_cols, how="left")
    features = features.merge(slopes, on=group_cols, how="left")

    features["rank_change_7d"] = features["first_rank"] - features["last_rank_obs"]
    features["streams_growth_abs_7d"] = features["last_streams_obs"] - features["first_streams"]
    features["streams_growth_pct_7d"] = safe_divide(
        features["streams_growth_abs_7d"], features["first_streams"]
    )
    features["log_first_streams"] = np.log1p(features["first_streams"])
    features["log_total_streams_7d"] = np.log1p(features["total_streams_7d"])
    features["already_top50_obs"] = (features["best_rank_7d"] <= target_rank).astype(int)

    # Target: future Top-N in the period after the observation window.
    future_best = (
        future.groupby(group_cols)["rank"]
        .min()
        .rename("best_future_rank")
        .reset_index()
    )
    features = features.merge(future_best, on=group_cols, how="left")
    features["target_future_top50"] = (
        features["best_future_rank"].notna()
        & (features["best_future_rank"] <= target_rank)
    ).astype(int)

    # Tracks that disappear from charts in the future window are treated as negatives.
    features["best_future_rank"] = features["best_future_rank"].fillna(9999)

    # Track-level regional spread in the observation window.
    if "region" in obs.columns:
        region_count = (
            obs.groupby("track_key")["region"]
            .nunique()
            .rename("regions_count_obs")
            .reset_index()
        )
        features = features.merge(region_count, on="track_key", how="left")
    else:
        features["regions_count_obs"] = 1

    features["first_month"] = features["first_date"].dt.month
    features["first_dayofweek"] = features["first_date"].dt.dayofweek
    features["first_quarter"] = features["first_date"].dt.quarter

    return features


def add_artist_history(features: pd.DataFrame) -> pd.DataFrame:
    """Create lagged artist-level features without using future information."""
    df = features.sort_values(["artist_norm", "first_date", "track_key", "region"]).copy()

    df["artist_previous_chart_entries"] = 0
    df["artist_prev_best_rank"] = np.nan
    df["artist_prev_mean_streams"] = np.nan

    pieces = []
    for _, g in df.groupby("artist_norm", sort=False):
        g = g.copy()
        g["artist_previous_chart_entries"] = np.arange(len(g))
        g["artist_prev_best_rank"] = g["best_rank_7d"].cummin().shift(1)
        g["artist_prev_mean_streams"] = g["mean_streams_7d"].shift(1).expanding().mean()
        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True)
    out["artist_is_new"] = (out["artist_previous_chart_entries"] == 0).astype(int)
    return out.sort_values("first_date").reset_index(drop=True)


def merge_audio(features: pd.DataFrame, audio: dict | None) -> pd.DataFrame:
    if audio is None:
        return features

    out = features.copy()
    audio_cols = audio["audio_columns"]

    # First try track_id join when possible.
    if audio.get("by_id") is not None and "track_id" in out.columns:
        by_id = audio["by_id"]
        out = out.merge(by_id, on="track_id", how="left")

    # Then fill missing audio values by normalized title + artist.
    by_text = audio["by_text"]
    out = out.merge(by_text, on="text_key", how="left", suffixes=("", "_text"))

    for col in audio_cols:
        text_col = f"{col}_text"
        if col in out.columns and text_col in out.columns:
            out[col] = out[col].combine_first(out[text_col])
            out = out.drop(columns=[text_col])
        elif text_col in out.columns:
            out = out.rename(columns={text_col: col})

    return out


def add_structure_inspired_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple audio interactions that approximate musical profile/structure."""
    out = df.copy()

    if "audio_duration_ms" in out.columns:
        out["duration_min"] = out["audio_duration_ms"] / 60000
        out["is_short_track"] = (out["audio_duration_ms"] < 150_000).astype(float)
        out["is_long_track"] = (out["audio_duration_ms"] > 300_000).astype(float)

    if {"audio_danceability", "audio_energy"}.issubset(out.columns):
        out["dance_energy"] = out["audio_danceability"] * out["audio_energy"]

    if {"audio_valence", "audio_energy"}.issubset(out.columns):
        out["positive_energy"] = out["audio_valence"] * out["audio_energy"]

    if {"audio_loudness", "audio_energy"}.issubset(out.columns):
        out["loudness_energy"] = out["audio_loudness"] * out["audio_energy"]

    if {"audio_acousticness", "audio_energy"}.issubset(out.columns):
        out["acoustic_energy_contrast"] = out["audio_acousticness"] * (1 - out["audio_energy"])

    if "audio_tempo" in out.columns:
        out["tempo_squared"] = out["audio_tempo"] ** 2

    return out


def build_dataset(
    charts_path: str | Path,
    audio_path: str | Path | None,
    out_path: str | Path,
    region: str,
    observation_days: int,
    horizon_days: int,
    target_rank: int,
    top200_only: bool,
    exclude_observed_hits: bool,
) -> pd.DataFrame:
    charts = load_charts(charts_path, region=region, top200_only=top200_only)
    audio = load_audio_features(audio_path)

    features = _build_observation_features(
        charts,
        observation_days=observation_days,
        horizon_days=horizon_days,
        target_rank=target_rank,
    )
    features = add_artist_history(features)
    features = merge_audio(features, audio)

    audio_cols = [c for c in features.columns if c.startswith("audio_")]
    if audio_cols:
        features["audio_features_non_missing"] = features[audio_cols].notna().sum(axis=1).astype(int)
        features["has_audio_features"] = (features["audio_features_non_missing"] > 0).astype(int)
    else:
        features["audio_features_non_missing"] = 0
        features["has_audio_features"] = 0

    features = add_structure_inspired_features(features)

    if exclude_observed_hits:
        features = features[features["already_top50_obs"] == 0].copy()

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    features.to_csv(out_path, index=False)

    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML table for Spotify hit prediction.")
    parser.add_argument("--charts", required=True, help="Path to Spotify charts CSV.")
    parser.add_argument("--audio", default=None, help="Path to audio features CSV.")
    parser.add_argument("--out", default="data/processed/model_table.csv", help="Output CSV path.")
    parser.add_argument("--region", default="Global", help="Region to use, e.g. Global, United States, or all.")
    parser.add_argument("--observation-days", type=int, default=7)
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--target-rank", type=int, default=50)
    parser.add_argument("--all-charts", action="store_true", help="Do not filter to Top charts.")
    parser.add_argument(
        "--exclude-observed-hits",
        action="store_true",
        help="Remove tracks that already reached target rank during observation window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_dataset(
        charts_path=args.charts,
        audio_path=args.audio,
        out_path=args.out,
        region=args.region,
        observation_days=args.observation_days,
        horizon_days=args.horizon_days,
        target_rank=args.target_rank,
        top200_only=not args.all_charts,
        exclude_observed_hits=args.exclude_observed_hits,
    )

    print(f"Saved dataset: {args.out}")
    print(f"Rows: {len(df):,}")
    print("Target distribution:")
    print(df["target_future_top50"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
