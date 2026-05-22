from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def save_target_distribution(df: pd.DataFrame, out_dir: Path, target: str) -> None:
    counts = df[target].value_counts().sort_index()
    labels = ["not future hit", "future hit"]

    plt.figure(figsize=(7, 5))
    plt.bar([labels[int(i)] if int(i) in [0, 1] else str(i) for i in counts.index], counts.values)
    plt.title("Target distribution")
    plt.ylabel("Number of tracks")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_dir / "target_distribution.png", dpi=180)
    plt.close()


def save_rank_by_target(df: pd.DataFrame, out_dir: Path, target: str) -> None:
    if "first_rank" not in df.columns:
        return

    data = [
        df.loc[df[target] == 0, "first_rank"].dropna(),
        df.loc[df[target] == 1, "first_rank"].dropna(),
    ]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=["not future hit", "future hit"])
    plt.title("First chart rank by future hit status")
    plt.ylabel("First rank; lower is better")
    plt.tight_layout()
    plt.savefig(out_dir / "first_rank_by_target.png", dpi=180)
    plt.close()


def save_streams_growth_by_target(df: pd.DataFrame, out_dir: Path, target: str) -> None:
    if "streams_growth_pct_7d" not in df.columns:
        return

    clipped = df.copy()
    clipped["streams_growth_pct_7d"] = clipped["streams_growth_pct_7d"].clip(-5, 10)

    data = [
        clipped.loc[clipped[target] == 0, "streams_growth_pct_7d"].dropna(),
        clipped.loc[clipped[target] == 1, "streams_growth_pct_7d"].dropna(),
    ]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=["not future hit", "future hit"])
    plt.title("Streams growth during observation window")
    plt.ylabel("Growth pct, clipped")
    plt.tight_layout()
    plt.savefig(out_dir / "streams_growth_by_target.png", dpi=180)
    plt.close()


def save_audio_match_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    if "has_audio_features" not in df.columns:
        return

    counts = df["has_audio_features"].value_counts().sort_index()
    labels = ["no audio match" if int(i) == 0 else "audio matched" for i in counts.index]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts.values)
    plt.title("Audio feature matching coverage")
    plt.ylabel("Number of rows")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_dir / "audio_match_distribution.png", dpi=180)
    plt.close()


def save_audio_feature_histograms(df: pd.DataFrame, out_dir: Path, target: str) -> None:
    for col in ["audio_energy", "audio_danceability", "audio_valence", "audio_tempo", "duration_min"]:
        if col not in df.columns:
            continue

        plt.figure(figsize=(7, 5))
        df.loc[df[target] == 0, col].dropna().hist(alpha=0.6, bins=30, label="not future hit")
        df.loc[df[target] == 1, col].dropna().hist(alpha=0.6, bins=30, label="future hit")
        plt.title(f"{col} distribution")
        plt.xlabel(col)
        plt.ylabel("Number of tracks")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}_hist.png", dpi=180)
        plt.close()


def save_correlation_table(df: pd.DataFrame, out_dir: Path, target: str) -> None:
    numeric = df.select_dtypes(include=["number"]).copy()
    if target not in numeric.columns:
        return

    # These columns are useful for sanity checks, but should not be presented as
    # normal model features because they are target-like, metadata, or potential
    # temporal leakage.
    excluded_from_report = {
        target,
        "best_future_rank",
        "already_top50_obs",
        "has_audio_features",
        "audio_features_non_missing",
        "audio_popularity",
    }

    corr = (
        numeric.corr(numeric_only=True)[target]
        .drop(labels=list(excluded_from_report), errors="ignore")
        .sort_values(key=lambda x: x.abs(), ascending=False)
        .head(30)
        .rename("correlation_with_target")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    corr.to_csv(out_dir.parent / "top_correlations.csv", index=False)


def save_leakage_sanity_table(df: pd.DataFrame, out_dir: Path, target: str) -> None:
    numeric = df.select_dtypes(include=["number"]).copy()
    if target not in numeric.columns:
        return

    sanity_cols = [
        "best_future_rank",
        "already_top50_obs",
        "audio_popularity",
        "has_audio_features",
        "audio_features_non_missing",
    ]
    sanity_cols = [c for c in sanity_cols if c in numeric.columns]
    if not sanity_cols:
        return

    corr = (
        numeric[sanity_cols + [target]]
        .corr(numeric_only=True)[target]
        .drop(labels=[target], errors="ignore")
        .rename("correlation_with_target")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    corr["note"] = "sanity check only; excluded from model features"
    corr.to_csv(out_dir.parent / "leakage_sanity_check.csv", index=False)


def run_eda(data_path: str | Path, out_dir: str | Path, target: str) -> None:
    df = pd.read_csv(data_path)
    out_dir = ensure_dir(out_dir)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    save_target_distribution(df, out_dir, target)
    save_audio_match_distribution(df, out_dir)
    save_rank_by_target(df, out_dir, target)
    save_streams_growth_by_target(df, out_dir, target)
    save_audio_feature_histograms(df, out_dir, target)
    save_correlation_table(df, out_dir, target)
    save_leakage_sanity_table(df, out_dir, target)

    print(f"Saved EDA figures to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple EDA figures.")
    parser.add_argument("--data", required=True, help="Processed model table CSV.")
    parser.add_argument("--out-dir", default="reports/figures")
    parser.add_argument("--target", default="target_future_top50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eda(args.data, args.out_dir, args.target)


if __name__ == "__main__":
    main()
