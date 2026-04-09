import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ppg_pipeline import build_feature_matrix, detect_beats, preprocess_ppg, segment_signal

# -----------------------------
# Custom CSS for background
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="PPG AF Classifier", layout="wide")

# -----------------------------
# Load trained model
# -----------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "ppg_af_rf.joblib"

model = None
model_load_error = None
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model_load_error = f"Random Forest model not found at {MODEL_PATH}. Export it from the notebook first."

# -----------------------------
# Sidebar for settings
# -----------------------------
st.sidebar.header("Segmentation Settings")
window_sec = st.sidebar.slider("Window length (sec)", 1, 20, 5, key="window_slider")
overlap_sec = st.sidebar.slider("Overlap length (sec)", 0.5, 10.0, 2.5, key="overlap_slider")

# -----------------------------
# Main UI
# -----------------------------
st.title("PPG Atrial Fibrillation Classifier")
st.write("Upload a PPG `.csv` file to predict AF vs Normal rhythm using the feature-based Random Forest model.")

if model_load_error:
    st.error(model_load_error)
    st.stop()

uploaded_file = st.file_uploader("Choose a PPG file", type=["csv"], key="ppg_file_uploader")

if uploaded_file is not None:
    # Load raw signal
    df = pd.read_csv(uploaded_file)
    ppg_raw = df["PPG"].values if "PPG" in df.columns else df.iloc[:, 0].values
    
    # Display raw PPG
    st.subheader("Raw PPG Signal Overview")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ppg_raw, color="purple")
    ax.set_title("Raw PPG Signal")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)

    # Preprocess
    ppg_proc = preprocess_ppg(ppg_raw)
    peak_locs, ibi, _ = detect_beats(ppg_proc)

    st.subheader("Processed PPG Signal")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ppg_proc, color="teal", linewidth=1.1)
    if len(peak_locs) > 0:
        ax.scatter(peak_locs, ppg_proc[peak_locs], color="crimson", s=15, label="Peaks")
        ax.legend()
    ax.set_title("Filtered + Normalized PPG")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)

    st.caption(f"Detected {len(peak_locs)} beats and computed {len(ibi)} inter-beat intervals.")

    # Segment and featurize
    windows = segment_signal(ppg_proc, window_sec=window_sec, overlap_sec=overlap_sec)
    if len(windows) == 0:
        st.error("PPG signal too short for segmentation.")
        st.stop()

    X_features = build_feature_matrix(windows)

    st.subheader("Extracted Window Features")
    st.dataframe(X_features.head(min(5, len(X_features))))

    # Predict with Random Forest
    predictions = model.predict_proba(X_features)[:, 1]
    pred_labels = (predictions >= 0.5).astype(int)

    # Summary
    af_percentage = 100 * np.mean(pred_labels)
    mean_probability = 100 * np.mean(predictions)
    st.markdown(f"### AF classification: {af_percentage:.2f}% of windows predicted as AF")
    st.markdown(f"### Mean AF probability: {mean_probability:.2f}%")

    # Plot example windows
    st.subheader("Example PPG Windows")
    n_windows = min(3, len(windows))
    fig, axs = plt.subplots(n_windows, 1, figsize=(10, 3 * n_windows))

    if n_windows == 1:
        axs = [axs]

    for i in range(n_windows):
        color = "red" if pred_labels[i] == 1 else "green"
        axs[i].plot(windows[i], color=color)
        axs[i].set_title(f"Window {i + 1} - AF: {'Yes' if pred_labels[i] else 'No'} (p={predictions[i]:.3f})")
        axs[i].set_xlabel("Samples")
        axs[i].set_ylabel("Amplitude")
        axs[i].grid(True)

    st.pyplot(fig)

    # Download predictions
    df_pred = pd.DataFrame(
        {
            "Window_Index": np.arange(len(pred_labels)),
            "AF_Prediction": pred_labels,
            "AF_Probability": predictions,
        }
    )
    csv_buffer = io.StringIO()
    df_pred.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()
    st.download_button(
        label="Download Predictions CSV",
        data=csv_bytes,
        file_name="ppg_af_predictions.csv",
        mime="text/csv",
        key="download_csv",
    )
    st.success("Predictions ready for download!")
