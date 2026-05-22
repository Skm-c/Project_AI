from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .utils import ensure_dir, get_one_hot_encoder


METADATA_COLUMNS = {
    "track_key",
    "track_id",
    "text_key",
    "track_norm",
    "artist_norm",
    "track_name",
    "artist_name",
    "first_observed_date",
    "first_date",
    "best_future_rank",
    "already_top50_obs",
    "has_audio_features",
    "audio_features_non_missing",
    # Spotify popularity from a static audio-feature dump is not available at the
    # first 7 chart days historically, so it is excluded to avoid temporal leakage.
    "audio_popularity",
}

STRUCTURE_INSPIRED_COLUMNS = {
    "duration_min",
    "is_short_track",
    "is_long_track",
    "dance_energy",
    "positive_energy",
    "loudness_energy",
    "acoustic_energy_contrast",
    "tempo_squared",
}

FEATURE_SET_CHOICES = ["chart_only", "chart_plus_audio"]


FEATURE_SELECTION_CHOICES = ["none", "filter", "wrapper", "embedded", "hybrid"]
FILTER_SCORE_FUNC_CHOICES = ["f_classif", "mutual_info"]
EMBEDDED_ESTIMATOR_CHOICES = ["l1_logistic", "random_forest"]


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    """SelectKBest that automatically caps k at the number of available features.

    This makes the training script robust when the number of one-hot encoded
    features is smaller than the requested k.
    """

    def __init__(self, score_func=f_classif, k: int | str = 50):
        self.score_func = score_func
        self.k = k
        self.selector_: SelectKBest | None = None
        self.k_: int | str | None = None

    def fit(self, X, y=None):
        n_features = X.shape[1]
        if self.k == "all":
            self.k_ = "all"
        else:
            self.k_ = min(int(self.k), int(n_features))
        self.selector_ = SelectKBest(score_func=self.score_func, k=self.k_)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        if self.selector_ is None:
            raise RuntimeError("SafeSelectKBest must be fitted before transform.")
        return self.selector_.transform(X)

    def get_support(self) -> np.ndarray:
        if self.selector_ is None:
            raise RuntimeError("SafeSelectKBest must be fitted before get_support.")
        return self.selector_.get_support()

    @property
    def scores_(self):
        return None if self.selector_ is None else self.selector_.scores_

    @property
    def pvalues_(self):
        return None if self.selector_ is None else getattr(self.selector_, "pvalues_", None)


def get_audio_related_columns(df: pd.DataFrame) -> list[str]:
    """Columns that come from audio features or audio-based interactions."""
    return [
        c
        for c in df.columns
        if c.startswith("audio_") or c in STRUCTURE_INSPIRED_COLUMNS
    ]


def compute_audio_match_info(df: pd.DataFrame) -> pd.Series:
    """Return number of non-missing raw audio feature columns for each row."""
    raw_audio_cols = [c for c in df.columns if c.startswith("audio_")]
    # Metadata columns should not count as audio features if the function is called after enrichment.
    raw_audio_cols = [c for c in raw_audio_cols if c not in {"audio_features_non_missing"}]
    if not raw_audio_cols:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    return df[raw_audio_cols].notna().sum(axis=1).astype(int)


def filter_rows_for_audio(
    df: pd.DataFrame,
    min_audio_features: int = 3,
) -> pd.DataFrame:
    """Keep rows where the audio dataset was actually matched."""
    out = df.copy()
    if "audio_features_non_missing" in out.columns:
        counts = pd.to_numeric(out["audio_features_non_missing"], errors="coerce").fillna(0)
    else:
        counts = compute_audio_match_info(out)
    return out[counts >= min_audio_features].copy()


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "first_date",
    train_size: float = 0.70,
    valid_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe by time into train/valid/test."""
    if date_col not in df.columns:
        raise ValueError(f"{date_col} is required for chronological split.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    unique_dates = np.array(sorted(out[date_col].unique()))
    if len(unique_dates) < 3:
        raise ValueError("Not enough unique dates for chronological split.")

    train_idx = max(1, int(len(unique_dates) * train_size))
    valid_idx = max(train_idx + 1, int(len(unique_dates) * (train_size + valid_size)))
    valid_idx = min(valid_idx, len(unique_dates) - 1)

    train_end = unique_dates[train_idx - 1]
    valid_end = unique_dates[valid_idx - 1]

    train = out[out[date_col] <= train_end].copy()
    valid = out[(out[date_col] > train_end) & (out[date_col] <= valid_end)].copy()
    test = out[out[date_col] > valid_end].copy()

    if train.empty or valid.empty or test.empty:
        raise ValueError(
            "Chronological split produced an empty split. "
            "Try using more data or changing split proportions."
        )

    return train, valid, test


def infer_feature_columns(df: pd.DataFrame, target: str, feature_set: str) -> list[str]:
    """Select model feature columns, excluding metadata, target-like columns and optional audio features."""
    if feature_set not in FEATURE_SET_CHOICES:
        raise ValueError(f"Unknown feature_set={feature_set}. Use one of: {FEATURE_SET_CHOICES}")

    excluded = set(METADATA_COLUMNS)
    excluded.add(target)

    # Exclude any other target columns to avoid accidental leakage.
    excluded.update([c for c in df.columns if c.startswith("target_") and c != target])

    if feature_set == "chart_only":
        excluded.update(get_audio_related_columns(df))

    features = [c for c in df.columns if c not in excluded]
    return features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", get_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )




def _get_filter_score_func(name: str):
    if name == "f_classif":
        return f_classif
    if name == "mutual_info":
        return mutual_info_classif
    raise ValueError(f"Unknown filter score function: {name}")


def _build_embedded_estimator(name: str, random_state: int) -> Any:
    """Estimator used inside SelectFromModel for embedded feature selection."""
    if name == "l1_logistic":
        # L1 regularization can shrink weak coefficients to zero, which makes it
        # a classical embedded feature-selection method.
        return LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
    if name == "random_forest":
        # A tree ensemble can select features by impurity-based feature importance.
        return RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown embedded estimator: {name}")


def build_feature_selection_steps(
    method: str,
    k_best: int,
    variance_threshold: float,
    filter_score_func: str,
    rfe_fraction: float,
    embedded_estimator: str,
    embedded_threshold: str | float,
    random_state: int,
) -> list[tuple[str, Any]]:
    """Build optional feature-selection steps for the sklearn Pipeline.

    Implemented methods:
    - none: no formal feature selection;
    - filter: VarianceThreshold + SelectKBest;
    - wrapper: RFE with Logistic Regression;
    - embedded: SelectFromModel with L1 Logistic Regression or Random Forest;
    - hybrid: VarianceThreshold + SelectKBest + SelectFromModel + RFE.
    """
    if method not in FEATURE_SELECTION_CHOICES:
        raise ValueError(f"Unknown feature_selection={method}. Use one of: {FEATURE_SELECTION_CHOICES}")
    if method == "none":
        return []

    steps: list[tuple[str, Any]] = []

    if method in {"filter", "hybrid"}:
        steps.append(("variance_filter", VarianceThreshold(threshold=variance_threshold)))
        steps.append(
            (
                "filter_select_kbest",
                SafeSelectKBest(score_func=_get_filter_score_func(filter_score_func), k=k_best),
            )
        )

    if method in {"embedded", "hybrid"}:
        steps.append(
            (
                "embedded_select_from_model",
                SelectFromModel(
                    estimator=_build_embedded_estimator(embedded_estimator, random_state=random_state),
                    threshold=embedded_threshold,
                ),
            )
        )

    if method in {"wrapper", "hybrid"}:
        # RFE is an obërtochnyj method: it repeatedly trains a model and removes
        # the least useful features. Logistic Regression is used only as the
        # selector; the final classifier can still be Random Forest, Boosting, etc.
        rfe_estimator = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )
        steps.append(
            (
                "wrapper_rfe",
                RFE(
                    estimator=rfe_estimator,
                    n_features_to_select=float(rfe_fraction),
                    step=0.2,
                ),
            )
        )

    return steps


def get_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def predict_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            return np.zeros(len(X))
        return proba[:, 1]

    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X)
    y_prob = predict_probability(model, X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    if y.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y, y_prob)
        metrics["average_precision"] = average_precision_score(y, y_prob)
        try:
            metrics["log_loss"] = log_loss(y, y_prob, labels=[0, 1])
        except ValueError:
            metrics["log_loss"] = np.nan
        metrics["brier_score"] = brier_score_loss(y, y_prob)
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
        metrics["log_loss"] = np.nan
        metrics["brier_score"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return metrics


def get_preprocessed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def _apply_selector_support(feature_names: list[str], selector: Any) -> list[str]:
    if not feature_names or not hasattr(selector, "get_support"):
        return feature_names
    try:
        support = np.asarray(selector.get_support(), dtype=bool)
        if len(support) == len(feature_names):
            return [name for name, keep in zip(feature_names, support) if keep]
    except Exception:
        pass
    return feature_names


def get_feature_names(pipeline: Pipeline, final: bool = True) -> list[str]:
    """Return feature names after preprocessing and, optionally, after selection."""
    feature_names = get_preprocessed_feature_names(pipeline)
    if not final:
        return feature_names

    for step_name in ["variance_filter", "filter_select_kbest", "embedded_select_from_model", "wrapper_rfe"]:
        if step_name in pipeline.named_steps:
            feature_names = _apply_selector_support(feature_names, pipeline.named_steps[step_name])
    return feature_names


def save_feature_selection_report(pipeline: Pipeline, out_path: Path) -> None:
    """Save selected features and available filter/wrapper diagnostics."""
    preprocessed_names = get_preprocessed_feature_names(pipeline)
    if not preprocessed_names:
        return

    current_names = preprocessed_names
    report_frames: list[pd.DataFrame] = []

    if "variance_filter" in pipeline.named_steps:
        selector = pipeline.named_steps["variance_filter"]
        support = selector.get_support()
        report_frames.append(
            pd.DataFrame(
                {
                    "stage": "filter_variance_threshold",
                    "feature": current_names,
                    "selected": support,
                    "score": getattr(selector, "variances_", np.nan),
                    "ranking": np.nan,
                }
            )
        )
        current_names = [name for name, keep in zip(current_names, support) if keep]

    if "filter_select_kbest" in pipeline.named_steps:
        selector = pipeline.named_steps["filter_select_kbest"]
        support = selector.get_support()
        scores = selector.scores_
        if scores is None:
            scores = np.full(len(current_names), np.nan)
        report_frames.append(
            pd.DataFrame(
                {
                    "stage": "filter_select_kbest",
                    "feature": current_names,
                    "selected": support,
                    "score": scores,
                    "ranking": np.nan,
                }
            )
        )
        current_names = [name for name, keep in zip(current_names, support) if keep]

    if "embedded_select_from_model" in pipeline.named_steps:
        selector = pipeline.named_steps["embedded_select_from_model"]
        support = selector.get_support()
        estimator = getattr(selector, "estimator_", None)
        scores = np.full(len(current_names), np.nan)
        if estimator is not None:
            if hasattr(estimator, "feature_importances_"):
                scores = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                scores = np.ravel(estimator.coef_)
        report_frames.append(
            pd.DataFrame(
                {
                    "stage": "embedded_select_from_model",
                    "feature": current_names,
                    "selected": support,
                    "score": scores,
                    "ranking": np.nan,
                }
            )
        )
        current_names = [name for name, keep in zip(current_names, support) if keep]

    if "wrapper_rfe" in pipeline.named_steps:
        selector = pipeline.named_steps["wrapper_rfe"]
        support = selector.get_support()
        report_frames.append(
            pd.DataFrame(
                {
                    "stage": "wrapper_rfe",
                    "feature": current_names,
                    "selected": support,
                    "score": np.nan,
                    "ranking": getattr(selector, "ranking_", np.nan),
                }
            )
        )
        current_names = [name for name, keep in zip(current_names, support) if keep]

    if report_frames:
        pd.concat(report_frames, ignore_index=True).to_csv(out_path, index=False)
    else:
        pd.DataFrame(
            {"stage": "none", "feature": preprocessed_names, "selected": True, "score": np.nan, "ranking": np.nan}
        ).to_csv(out_path, index=False)


def save_feature_importance(pipeline: Pipeline, out_path: Path) -> None:
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline, final=True)
    if not feature_names:
        return

    importance = None
    column_name = "importance"

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.ravel(model.coef_)
        column_name = "coefficient"

    if importance is None or len(importance) != len(feature_names):
        return

    imp = pd.DataFrame({"feature": feature_names, column_name: importance})
    imp["abs_value"] = imp[column_name].abs()
    imp = imp.sort_values("abs_value", ascending=False).drop(columns=["abs_value"])
    imp.to_csv(out_path, index=False)



def time_series_cv_splits(
    df: pd.DataFrame,
    date_col: str = "first_date",
    n_splits: int = 5,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Create expanding-window cross-validation folds over unique dates.

    This is safer than ordinary KFold for this project: every validation fold is
    later in time than the corresponding training fold, so the model cannot learn
    from future chart outcomes.
    """
    if n_splits < 2:
        return []
    if date_col not in df.columns:
        raise ValueError(f"{date_col} is required for time-series cross-validation.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    unique_dates = np.array(sorted(out[date_col].unique()))

    if len(unique_dates) <= n_splits:
        raise ValueError(
            f"Not enough unique dates for {n_splits}-fold time-series CV. "
            f"Found only {len(unique_dates)} unique dates."
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for fold_id, (train_date_idx, valid_date_idx) in enumerate(splitter.split(unique_dates), start=1):
        train_dates = set(unique_dates[train_date_idx])
        valid_dates = set(unique_dates[valid_date_idx])
        fold_train = out[out[date_col].isin(train_dates)].copy()
        fold_valid = out[out[date_col].isin(valid_dates)].copy()
        if not fold_train.empty and not fold_valid.empty:
            folds.append((fold_id, fold_train, fold_valid))
    return folds


def _make_pipeline(
    X_train: pd.DataFrame,
    clf: Any,
    feature_selection: str,
    k_best: int,
    variance_threshold: float,
    filter_score_func: str,
    rfe_fraction: float,
    embedded_estimator: str,
    embedded_threshold: str | float,
    random_state: int,
) -> Pipeline:
    selection_steps = build_feature_selection_steps(
        method=feature_selection,
        k_best=k_best,
        variance_threshold=variance_threshold,
        filter_score_func=filter_score_func,
        rfe_fraction=rfe_fraction,
        embedded_estimator=embedded_estimator,
        embedded_threshold=embedded_threshold,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            *selection_steps,
            ("model", clone(clf)),
        ]
    )


def _empty_metric_row(error: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "log_loss": np.nan,
        "brier_score": np.nan,
        "tn": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "tp": np.nan,
    }
    if error is not None:
        row["error"] = error
    return row


def summarize_cv_metrics(cv_metrics: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    """Aggregate per-fold metrics into mean/std summary for model selection."""
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier_score",
    ]
    available = [c for c in metric_cols if c in cv_metrics.columns]
    summary = cv_metrics.groupby("model", as_index=False)[available].agg(["mean", "std"])
    summary.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in summary.columns.to_flat_index()
    ]
    metric_col = f"{selection_metric}_mean"
    if metric_col not in summary.columns:
        raise ValueError(f"Metric {selection_metric} is not available in CV summary.")
    summary = summary.sort_values(metric_col, ascending=False, na_position="last").reset_index(drop=True)
    return summary

def train_and_evaluate(
    data_path: str | Path,
    target: str,
    model_dir: str | Path,
    reports_dir: str | Path,
    selection_metric: str,
    random_state: int,
    feature_set: str = "chart_plus_audio",
    require_audio_features: bool = False,
    min_audio_features: int = 3,
    feature_selection: str = "none",
    k_best: int = 50,
    variance_threshold: float = 0.0,
    filter_score_func: str = "f_classif",
    rfe_fraction: float = 0.5,
    embedded_estimator: str = "l1_logistic",
    embedded_threshold: str | float = "median",
    cv_folds: int = 0,
) -> dict[str, Any]:
    df = pd.read_csv(data_path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found. Available columns: {list(df.columns)}")

    df[target] = df[target].astype(int)
    rows_before_filter = len(df)

    if require_audio_features:
        df = filter_rows_for_audio(df, min_audio_features=min_audio_features)
        if df.empty:
            raise ValueError(
                "No rows left after filtering for audio features. "
                "Try lowering --min-audio-features or check the merge between charts.csv and tracks.csv."
            )

    rows_after_filter = len(df)

    train_df, valid_df, test_df = chronological_split(df)
    feature_cols = infer_feature_columns(df, target=target, feature_set=feature_set)

    if not feature_cols:
        raise ValueError("No feature columns were selected. Check feature_set and input data.")

    X_test, y_test = test_df[feature_cols], test_df[target]
    train_valid = pd.concat([train_df, valid_df], ignore_index=True)

    models = get_models(random_state=random_state)

    model_dir = ensure_dir(model_dir)
    reports_dir = ensure_dir(reports_dir)

    if cv_folds and cv_folds >= 2:
        print(f"Using expanding-window time-series cross-validation with {cv_folds} folds for model selection.")
        cv_rows: list[dict[str, Any]] = []
        folds = time_series_cv_splits(train_valid, date_col="first_date", n_splits=cv_folds)

        for name, clf in models.items():
            fold_scores: list[float] = []
            for fold_id, fold_train_df, fold_valid_df in folds:
                X_fold_train, y_fold_train = fold_train_df[feature_cols], fold_train_df[target]
                X_fold_valid, y_fold_valid = fold_valid_df[feature_cols], fold_valid_df[target]

                row: dict[str, Any]
                try:
                    if y_fold_train.nunique() < 2:
                        raise ValueError("Training fold contains only one target class.")
                    pipeline = _make_pipeline(
                        X_train=X_fold_train,
                        clf=clf,
                        feature_selection=feature_selection,
                        k_best=k_best,
                        variance_threshold=variance_threshold,
                        filter_score_func=filter_score_func,
                        rfe_fraction=rfe_fraction,
                        embedded_estimator=embedded_estimator,
                        embedded_threshold=embedded_threshold,
                        random_state=random_state,
                    )
                    pipeline.fit(X_fold_train, y_fold_train)
                    row = evaluate_model(pipeline, X_fold_valid, y_fold_valid)
                except Exception as exc:
                    row = _empty_metric_row(error=str(exc))

                row.update(
                    {
                        "model": name,
                        "split": "cv",
                        "fold": fold_id,
                        "feature_set": feature_set,
                        "train_rows": int(len(fold_train_df)),
                        "valid_rows": int(len(fold_valid_df)),
                        "train_min_date": str(pd.to_datetime(fold_train_df["first_date"]).min().date()),
                        "train_max_date": str(pd.to_datetime(fold_train_df["first_date"]).max().date()),
                        "valid_min_date": str(pd.to_datetime(fold_valid_df["first_date"]).min().date()),
                        "valid_max_date": str(pd.to_datetime(fold_valid_df["first_date"]).max().date()),
                    }
                )
                cv_rows.append(row)
                if not pd.isna(row.get(selection_metric, np.nan)):
                    fold_scores.append(float(row[selection_metric]))

            mean_score = np.nan if not fold_scores else float(np.mean(fold_scores))
            print(f"{name}: CV mean {selection_metric} = {mean_score:.4f}")

        cv_metrics = pd.DataFrame(cv_rows)
        cv_metrics.to_csv(Path(reports_dir) / "metrics_cv_folds.csv", index=False)
        model_selection_metrics = summarize_cv_metrics(cv_metrics, selection_metric=selection_metric)
        model_selection_metrics.to_csv(Path(reports_dir) / "metrics_cv_summary.csv", index=False)
        model_selection_metrics.to_csv(Path(reports_dir) / "metrics_validation.csv", index=False)

        metric_col = f"{selection_metric}_mean"
        if model_selection_metrics[metric_col].isna().all():
            raise ValueError("All models failed during cross-validation. Check data size and target distribution.")
        best_name = model_selection_metrics.iloc[0]["model"]
        model_selection_strategy = f"time_series_cv_{cv_folds}_folds"
        print(f"Best model by CV mean {selection_metric}: {best_name}")
    else:
        X_train, y_train = train_df[feature_cols], train_df[target]
        X_valid, y_valid = valid_df[feature_cols], valid_df[target]

        validation_rows = []
        for name, clf in models.items():
            pipeline = _make_pipeline(
                X_train=X_train,
                clf=clf,
                feature_selection=feature_selection,
                k_best=k_best,
                variance_threshold=variance_threshold,
                filter_score_func=filter_score_func,
                rfe_fraction=rfe_fraction,
                embedded_estimator=embedded_estimator,
                embedded_threshold=embedded_threshold,
                random_state=random_state,
            )
            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(pipeline, X_valid, y_valid)
            metrics["model"] = name
            metrics["split"] = "validation"
            metrics["feature_set"] = feature_set
            validation_rows.append(metrics)
            print(f"{name}: validation {selection_metric} = {metrics.get(selection_metric, np.nan):.4f}")

        model_selection_metrics = pd.DataFrame(validation_rows).sort_values(selection_metric, ascending=False)
        model_selection_metrics.to_csv(Path(reports_dir) / "metrics_validation.csv", index=False)
        best_name = model_selection_metrics.iloc[0]["model"]
        model_selection_strategy = "single_chronological_validation"
        print(f"Best model by {selection_metric}: {best_name}")

    best_model_type = models[best_name]

    X_train_valid, y_train_valid = train_valid[feature_cols], train_valid[target]
    final_pipeline = _make_pipeline(
        X_train=X_train_valid,
        clf=best_model_type,
        feature_selection=feature_selection,
        k_best=k_best,
        variance_threshold=variance_threshold,
        filter_score_func=filter_score_func,
        rfe_fraction=rfe_fraction,
        embedded_estimator=embedded_estimator,
        embedded_threshold=embedded_threshold,
        random_state=random_state,
    )
    final_pipeline.fit(X_train_valid, y_train_valid)

    test_metrics = evaluate_model(final_pipeline, X_test, y_test)
    test_metrics["model"] = best_name
    test_metrics["split"] = "test"
    test_metrics["feature_set"] = feature_set
    test_metrics["feature_selection"] = feature_selection
    test_metrics["cv_folds"] = int(cv_folds)
    test_metrics["model_selection_strategy"] = model_selection_strategy
    test_metrics["k_best"] = int(k_best)
    test_metrics["rfe_fraction"] = float(rfe_fraction)
    test_metrics["require_audio_features"] = bool(require_audio_features)
    pd.DataFrame([test_metrics]).to_csv(Path(reports_dir) / "metrics_test.csv", index=False)

    y_prob = predict_probability(final_pipeline, X_test)
    y_pred = final_pipeline.predict(X_test)

    pred_cols = [c for c in ["track_name", "artist_name", "region", "first_date"] if c in test_df.columns]
    predictions = test_df[pred_cols].copy()
    predictions["y_true"] = y_test.values
    predictions["y_pred"] = y_pred
    predictions["hit_probability"] = y_prob
    predictions = predictions.sort_values("hit_probability", ascending=False)
    predictions.to_csv(Path(reports_dir) / "predictions_test.csv", index=False)

    model_path = Path(model_dir) / "best_model.joblib"
    joblib.dump(final_pipeline, model_path)

    with open(Path(model_dir) / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    selected_transformed_features = get_feature_names(final_pipeline, final=True)
    with open(Path(model_dir) / "selected_transformed_features.json", "w", encoding="utf-8") as f:
        json.dump(selected_transformed_features, f, ensure_ascii=False, indent=2)

    metadata = {
        "best_model": best_name,
        "target": target,
        "feature_set": feature_set,
        "selection_metric": selection_metric,
        "model_selection_strategy": model_selection_strategy,
        "cv_folds": int(cv_folds),
        "raw_feature_count": len(feature_cols),
        "selected_transformed_feature_count": len(selected_transformed_features),
        "feature_selection": feature_selection,
        "k_best": int(k_best),
        "variance_threshold": float(variance_threshold),
        "filter_score_func": filter_score_func,
        "rfe_fraction": float(rfe_fraction),
        "embedded_estimator": embedded_estimator,
        "embedded_threshold": embedded_threshold,
        "require_audio_features": bool(require_audio_features),
        "min_audio_features": int(min_audio_features),
        "rows_before_filter": int(rows_before_filter),
        "rows_after_filter": int(rows_after_filter),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "train_valid_rows": int(len(train_valid)),
        "test_rows": int(len(test_df)),
        "train_min_date": str(pd.to_datetime(train_df["first_date"]).min().date()),
        "train_max_date": str(pd.to_datetime(train_df["first_date"]).max().date()),
        "valid_min_date": str(pd.to_datetime(valid_df["first_date"]).min().date()),
        "valid_max_date": str(pd.to_datetime(valid_df["first_date"]).max().date()),
        "test_min_date": str(pd.to_datetime(test_df["first_date"]).min().date()),
        "test_max_date": str(pd.to_datetime(test_df["first_date"]).max().date()),
    }
    with open(Path(model_dir) / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    save_feature_selection_report(final_pipeline, Path(reports_dir) / "feature_selection_report.csv")
    save_feature_importance(final_pipeline, Path(reports_dir) / "feature_importance.csv")

    print("\nTest metrics:")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"\nSaved model: {model_path}")

    return {
        "model_dir": str(model_dir),
        "reports_dir": str(reports_dir),
        "metadata": metadata,
        "model_selection_metrics": model_selection_metrics,
        "test_metrics": test_metrics,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical ML models for Spotify hit prediction.")
    parser.add_argument("--data", required=True, help="Processed model table CSV.")
    parser.add_argument("--target", default="target_future_top50")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
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
        "--feature-set",
        default="chart_plus_audio",
        choices=FEATURE_SET_CHOICES,
        help="chart_only excludes audio/structure features; chart_plus_audio uses all available features.",
    )
    parser.add_argument(
        "--require-audio-features",
        action="store_true",
        help="Keep only tracks that were matched with the audio features dataset.",
    )
    parser.add_argument(
        "--min-audio-features",
        type=int,
        default=3,
        help="Minimum number of non-missing audio feature columns required when --require-audio-features is used.",
    )
    parser.add_argument(
        "--feature-selection",
        default="none",
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
        help="Number of features to keep in SelectKBest for filter/hybrid methods.",
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
        help=(
            "Threshold for embedded SelectFromModel. Common values: median, mean, "
            "or a numeric threshold such as 0.001."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(
        data_path=args.data,
        target=args.target,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
        selection_metric=args.selection_metric,
        random_state=args.random_state,
        feature_set=args.feature_set,
        require_audio_features=args.require_audio_features,
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
