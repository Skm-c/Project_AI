from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "model_B_chart_plus_audio_matched"


st.set_page_config(page_title="Spotify Hit Predictor", page_icon="🎧", layout="wide")

st.title("🎧 Spotify Hit Predictor")
st.write(
    "Демо для проекта: можно загрузить подготовленный CSV и выбрать модель A или B. "
    "Модель A использует только chart/artist features, модель B добавляет audio и structure-inspired features."
)


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


def load_feature_columns(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


model_dir = st.sidebar.text_input(
    "Model folder",
    str(DEFAULT_MODEL_DIR),
    help="Например: models/model_A_chart_only_matched или models/model_B_chart_plus_audio_matched",
)
model_dir = Path(model_dir)
model_path = model_dir / "best_model.joblib"
features_path = model_dir / "feature_columns.json"
metadata_path = model_dir / "model_metadata.json"

if not model_path.exists():
    st.warning(
        "Сначала обучи модели через `python -m src.train_ab ...`. "
        "Файл модели пока не найден."
    )
    st.stop()

model = load_model(model_path)
feature_columns = load_feature_columns(features_path)

if metadata_path.exists():
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    st.sidebar.write("Loaded model metadata")
    st.sidebar.json(metadata)

uploaded = st.file_uploader(
    "Загрузи CSV с подготовленными признаками. Можно использовать `data/processed/model_table.csv`.",
    type=["csv"],
)

if uploaded is None:
    st.info("После загрузки CSV приложение покажет вероятность будущего успеха для каждого трека.")
    st.stop()

df = pd.read_csv(uploaded)
input_df = df.copy()

for col in ["target_future_top50", "best_future_rank"]:
    if col in input_df.columns:
        input_df = input_df.drop(columns=[col])

if feature_columns is not None:
    missing = [c for c in feature_columns if c not in input_df.columns]
    if missing:
        st.error(f"В CSV не хватает колонок, которые ожидает модель: {missing[:20]}")
        st.stop()
    input_df = input_df[feature_columns]

proba = model.predict_proba(input_df)[:, 1]
pred = (proba >= 0.5).astype(int)

result = df.copy()
result["hit_probability"] = proba
result["predicted_future_hit"] = pred

display_cols = [
    c
    for c in [
        "track_name",
        "artist_name",
        "region",
        "first_date",
        "first_rank",
        "best_rank_7d",
        "has_audio_features",
        "audio_features_non_missing",
        "hit_probability",
        "predicted_future_hit",
    ]
    if c in result.columns
]

st.subheader("Predictions")
st.dataframe(result[display_cols].sort_values("hit_probability", ascending=False), use_container_width=True)

st.download_button(
    "Скачать predictions.csv",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)
