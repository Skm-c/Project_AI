from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .train import EMBEDDED_ESTIMATOR_CHOICES, FILTER_SCORE_FUNC_CHOICES, FEATURE_SELECTION_CHOICES, train_and_evaluate
from .utils import ensure_dir


EXPERIMENTS = {
    "fair": [
        {
            "experiment": "model_A_chart_only_matched",
            "feature_set": "chart_only",
            "require_audio_features": True,
            "description": "A: only chart and artist-history features, trained on the same audio-matched subset as B.",
        },
        {
            "experiment": "model_B_chart_plus_audio_matched",
            "feature_set": "chart_plus_audio",
            "require_audio_features": True,
            "description": "B: chart + artist-history + audio/structure-inspired features, trained on audio-matched rows.",
        },
    ],
    "practical": [
        {
            "experiment": "model_A_chart_only_all",
            "feature_set": "chart_only",
            "require_audio_features": False,
            "description": "A: only chart and artist-history features, trained on all available chart rows.",
        },
        {
            "experiment": "model_B_chart_plus_audio_matched",
            "feature_set": "chart_plus_audio",
            "require_audio_features": True,
            "description": "B: chart + artist-history + audio/structure-inspired features, trained on audio-matched rows.",
        },
    ],
}


def _read_test_metrics(reports_dir: Path, experiment: str, description: str) -> pd.DataFrame:
    path = reports_dir / "metrics_test.csv"
    row = pd.read_csv(path)
    row.insert(0, "experiment", experiment)
    row["description"] = description
    return row


def train_ab(
    data_path: str | Path,
    target: str,
    model_root: str | Path,
    reports_root: str | Path,
    selection_metric: str,
    random_state: int,
    mode: str,
    min_audio_features: int,
    feature_selection: str,
    k_best: int,
    variance_threshold: float,
    filter_score_func: str,
    rfe_fraction: float,
    embedded_estimator: str,
    embedded_threshold: str | float,
    cv_folds: int,
) -> pd.DataFrame:
    if mode not in EXPERIMENTS:
        raise ValueError(f"Unknown mode={mode}. Use one of: {list(EXPERIMENTS)}")

    model_root = ensure_dir(model_root)
    reports_root = ensure_dir(reports_root)

    summary_rows = []
    for config in EXPERIMENTS[mode]:
        experiment = config["experiment"]
        print("\n" + "=" * 80)
        print(config["description"])
        print("=" * 80)

        model_dir = model_root / experiment
        reports_dir = reports_root / experiment

        train_and_evaluate(
            data_path=data_path,
            target=target,
            model_dir=model_dir,
            reports_dir=reports_dir,
            selection_metric=selection_metric,
            random_state=random_state,
            feature_set=config["feature_set"],
            require_audio_features=config["require_audio_features"],
            min_audio_features=min_audio_features,
            feature_selection=feature_selection,
            k_best=k_best,
            variance_threshold=variance_threshold,
            filter_score_func=filter_score_func,
            rfe_fraction=rfe_fraction,
            embedded_estimator=embedded_estimator,
            embedded_threshold=embedded_threshold,
            cv_folds=cv_folds,
        )

        summary_rows.append(_read_test_metrics(reports_dir, experiment, config["description"]))

    comparison = pd.concat(summary_rows, ignore_index=True)
    comparison_path = reports_root / f"model_comparison_{mode}.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\nModel comparison:")
    display_cols = [
        "experiment",
        "model",
        "feature_set",
        "feature_selection",
        "cv_folds",
        "model_selection_strategy",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier_score",
    ]
    print(comparison[[c for c in display_cols if c in comparison.columns]].to_string(index=False))
    print(f"\nSaved comparison: {comparison_path}")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model A and Model B for Spotify hit prediction.")
    parser.add_argument("--data", required=True, help="Processed model table CSV.")
    parser.add_argument("--target", default="target_future_top50")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument(
        "--selection-metric",
        default="f1",
        choices=["f1", "roc_auc", "average_precision", "recall", "precision", "accuracy"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help=(
            "Number of expanding-window time-series CV folds used for model selection. "
            "Use 0 or 1 to keep the single chronological validation split."
        ),
    )
    parser.add_argument(
        "--mode",
        default="fair",
        choices=["fair", "practical"],
        help=(
            "fair: train A and B on the same audio-matched subset; "
            "practical: train A on all chart rows and B on audio-matched rows."
        ),
    )
    parser.add_argument(
        "--min-audio-features",
        type=int,
        default=3,
        help="Minimum number of non-missing audio feature columns for audio-matched rows.",
    )
    parser.add_argument(
        "--feature-selection",
        default="hybrid",
        choices=FEATURE_SELECTION_CHOICES,
        help=(
            "none: no formal feature selection; "
            "filter: VarianceThreshold + SelectKBest; "
            "wrapper: RFE; "
            "embedded: SelectFromModel with L1 Logistic Regression or Random Forest; "
            "hybrid: VarianceThreshold + SelectKBest + SelectFromModel + RFE."
        ),
    )
    parser.add_argument(
        "--k-best",
        type=int,
        default=50,
        help="Number of features kept by SelectKBest for filter/hybrid methods.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.0,
        help="Threshold for VarianceThreshold in filter/hybrid methods.",
    )
    parser.add_argument(
        "--filter-score-func",
        default="f_classif",
        choices=FILTER_SCORE_FUNC_CHOICES,
        help="Scoring function for SelectKBest.",
    )
    parser.add_argument(
        "--rfe-fraction",
        type=float,
        default=0.5,
        help="Fraction of features kept by RFE for wrapper/hybrid methods.",
    )
    parser.add_argument(
        "--embedded-estimator",
        default="l1_logistic",
        choices=EMBEDDED_ESTIMATOR_CHOICES,
        help="Estimator used by embedded SelectFromModel.",
    )
    parser.add_argument(
        "--embedded-threshold",
        default="median",
        help="Threshold for embedded SelectFromModel: median, mean, or a numeric threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ab(
        data_path=args.data,
        target=args.target,
        model_root=args.model_root,
        reports_root=args.reports_root,
        selection_metric=args.selection_metric,
        random_state=args.random_state,
        mode=args.mode,
        min_audio_features=args.min_audio_features,
        feature_selection=args.feature_selection,
        k_best=args.k_best,
        variance_threshold=args.variance_threshold,
        filter_score_func=args.filter_score_func,
        rfe_fraction=args.rfe_fraction,
        embedded_estimator=args.embedded_estimator,
        embedded_threshold=args.embedded_threshold,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
