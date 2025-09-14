import streamlit as st
import scipy.io as sio
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import io

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
    unsafe_allow_html=True
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="PPG AF Classifier", layout="wide")

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = r'D:\PPG-Arrhythmia-Detection\models\ppg_af_lstm.h5'
model = load_model(MODEL_PATH)

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_ppg(ppg_raw, fs=125):
    """Bandpass filter (0.5-8 Hz) and normalize PPG"""
    ppg_raw = np.array(ppg_raw).flatten()
    ppg_raw[~np.isfinite(ppg_raw)] = np.interp(
        np.flatnonzero(~np.isfinite(ppg_raw)),
        np.flatnonzero(np.isfinite(ppg_raw)),
        ppg_raw[np.isfinite(ppg_raw)]
    )
    b, a = butter(3, np.array([0.5, 8])/(fs/2), btype='bandpass')
    ppg_filt = filtfilt(b, a, ppg_raw)
    return (ppg_filt - np.mean(ppg_filt)) / np.std(ppg_filt)

def segment_ppg(ppg, window_sec=5, overlap_sec=2.5, fs=125):
    """Segment PPG into windows"""
    window_len = int(window_sec*fs)
    overlap_len = int(overlap_sec*fs)
    segments = []
    start = 0
    while start + window_len <= len(ppg):
        segments.append(ppg[start:start+window_len])
        start += window_len - overlap_len
    return np.array(segments)

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
st.write("Upload a PPG `.mat` or `.csv` file to predict AF vs Normal rhythm.")

uploaded_file = st.file_uploader(
    "Choose a PPG file", type=["mat", "csv"], key="ppg_file_uploader"
)

if uploaded_file is not None:
    # Load raw signal
    if uploaded_file.name.endswith('.mat'):
        data = sio.loadmat(uploaded_file)
        if 'ppg' in data:
            ppg_raw = data['ppg']
        else:
            st.error("No 'ppg' variable found in the .mat file.")
            st.stop()
    elif uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        ppg_raw = df.iloc[:,0].values

    # Display raw PPG
    st.subheader("Raw PPG Signal Overview")
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(ppg_raw, color='purple')
    ax.set_title("Raw PPG Signal")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)

    # Preprocess
    ppg_proc = preprocess_ppg(ppg_raw)

    # Segment
    windows = segment_ppg(ppg_proc, window_sec=window_sec, overlap_sec=overlap_sec)
    if len(windows) == 0:
        st.error("PPG signal too short for segmentation.")
        st.stop()

    # LSTM input
    X_input = np.expand_dims(windows, axis=2)

    # Predict
    predictions = model.predict(X_input)
    pred_labels = (predictions > 0.5).astype(int)

    # Summary
    af_percentage = 100 * np.mean(pred_labels)
    st.markdown(f"### ✅ AF probability: {af_percentage:.2f}% of windows")

    # Plot example windows
    st.subheader("Example PPG Windows")
    n_windows = min(3, len(windows))
    fig, axs = plt.subplots(n_windows, 1, figsize=(10, 3*n_windows))

    if n_windows == 1:
        axs = [axs]

    for i in range(n_windows):
        color = 'red' if pred_labels[i][0] == 1 else 'green'
        axs[i].plot(windows[i], color=color)
        axs[i].set_title(f"Window {i+1} - AF: {'Yes' if pred_labels[i][0] else 'No'}")
        axs[i].set_xlabel("Samples")
        axs[i].set_ylabel("Amplitude")
        axs[i].grid(True)

    st.pyplot(fig)

    # Download predictions
    df_pred = pd.DataFrame({
        "Window_Index": np.arange(len(pred_labels)),
        "AF_Prediction": pred_labels.flatten(),
        "AF_Probability": predictions.flatten()
    })
    csv_buffer = io.StringIO()
    df_pred.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()
    st.download_button(
        label="Download Predictions CSV",
        data=csv_bytes,
        file_name="ppg_af_predictions.csv",
        mime="text/csv",
        key="download_csv"
    )
    st.success("Predictions ready for download!")