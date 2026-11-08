"""Streamlit interface for ASL word-level sign recognition."""

import tempfile
from pathlib import Path

import streamlit as st
import torch

from src.inference.preprocessing import preprocess_video
from src.inference.predictor import predict_all
from src.inference.utils import format_predictions
from src.data.aslcitizen_dataset import load_label_map


st.set_page_config(
    page_title="ASL Sign Language Recognition",
    page_icon="ASL",
    layout="wide",
)

st.title("ASL Word-Level Sign Language Recognition")

st.write(
    "Upload an ASL signing video to obtain predictions from "
    "the appearance, pose, and fusion models."
)

st.caption(
    "Research prototype using ASL Citizen100 trained checkpoints. "
    "Predictions are model outputs, not guaranteed translations."
)


@st.cache_data
def get_labels():
    return load_label_map(100)


def display_predictions(predictions):
    columns = st.columns(3)

    names = [
        ("appearance", "Appearance"),
        ("pose", "Pose"),
        ("fusion", "Fusion"),
    ]

    for column, (key, title) in zip(columns, names):
        with column:
            st.subheader(title)

            entries = predictions[key]

            if entries:
                st.metric(
                    "Top prediction",
                    entries[0]["gloss"],
                    f'{entries[0]["confidence_percent"]:.2f}%',
                )

            for rank, item in enumerate(entries, start=1):
                st.write(
                    f'{rank}. **{item["gloss"]}** â€” '
                    f'{item["confidence_percent"]:.2f}%'
                )


uploaded_file = st.file_uploader(
    "Upload a signing video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Recognize Sign", type="primary"):
        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix or ".mp4",
            ) as temporary_file:
                temporary_file.write(uploaded_file.getbuffer())
                temporary_path = temporary_file.name

            with st.spinner("Extracting frames and pose landmarks..."):
                frames, lti = preprocess_video(temporary_path)

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            st.info(f"Inference device: {device}")

            with st.spinner("Running all three recognition models..."):
                results = predict_all(frames, lti, device)

            predictions = format_predictions(
                results,
                get_labels(),
                top_k=5,
            )

            st.success("Recognition completed.")

            display_predictions(predictions)

        except Exception as error:
            st.error(f"Recognition failed: {error}")
            st.exception(error)

        finally:
            if temporary_path and Path(temporary_path).exists():
                Path(temporary_path).unlink()
