# PPG-Based Arrhythmia Detection with Python

This repository now uses a Python-only workflow to detect **atrial fibrillation (AF)** from PPG signals. Preprocessing, visualization, windowing, feature extraction, model training, and inference all live in Python, with a shared pipeline used by both the Jupyter notebook and the Streamlit app.

---

## Background

Photoplethysmography (PPG) measures pulsatile blood volume changes using light, offering a low-cost, wearable-friendly window into cardiac rhythm. While ECG is the gold standard for arrhythmia diagnosis, PPG captures the mechanical consequence of each heartbeat at the periphery and is well suited for continuous screening.

AF is a strong target for PPG-based detection because it produces irregular pulse timing and morphology variability that can be learned from beat-to-beat dynamics and signal statistics.

---

## Project Overview

This project aims to:

1. Load raw PPG recordings from the dataset in Python.
2. Preprocess each signal with interpolation, band-pass filtering, and z-score normalization.
3. Visualize raw signals, processed signals, detected peaks, and segmented windows.
4. Train baseline models to classify **AF vs Normal Rhythm**.
5. Provide a Streamlit interface for prediction and visualization.

---

## Dataset

- **MIMIC PERform AF Dataset**
- Contains about 20-minute PPG + ECG recordings from 35 subjects sampled at 125 Hz.
- This repo uses the extracted AF and non-AF CSV folders already present under `data/`.

---

## Workflow

### 1. Shared Python Signal Pipeline

Implemented in `code/ppg_pipeline.py`:

- Load records from the AF and non-AF CSV folders.
- Handle NaN/Inf values by linear interpolation.
- Apply a 0.5 to 8 Hz band-pass filter.
- Normalize each signal with z-score scaling.
- Detect peaks and compute inter-beat intervals.
- Segment each processed signal into overlapping windows.
- Provide reusable plotting helpers for notebook and app visualizations.

### 2. Training Notebook

Implemented in `code/af_rnn.ipynb`:

- Load raw signals directly from the CSV dataset folders in `data/`.
- Visualize raw vs processed PPG and detected peaks.
- Visualize example windows before training.
- Build the training window dataset in Python.
- Train, validate, evaluate, and save both an LSTM baseline and a stronger Random Forest baseline built on engineered features.

### 3. Streamlit Interface

Implemented in `code/ppg_app.py`:

- Accept `.csv` PPG uploads.
- Show raw and processed signal visualizations.
- Detect peaks and summarize beat counts.
- Segment windows, extract engineered features, and run Random Forest AF inference.
- Export window-wise predictions as CSV.

---

## Repository Structure

```text
.
|-- code/
|   |-- af_rnn.ipynb
|   |-- ppg_app.py
|   |-- ppg_pipeline.py
|   `-- example_generate.py
|-- models/
|   |-- ppg_af_lstm.keras
|   `-- ppg_af_rf.joblib
`-- README.md
```

---

## Usage

### 1. Training and Visualization

1. Open `code/af_rnn.ipynb`.
2. Run the preprocessing and visualization cells on the raw dataset.
3. Confirm `data/record_labels.csv` is present. It is already pre-filled for the AF/non-AF CSV split in this repo.
4. Train the notebook baselines.
5. Save the Random Forest model to `models/ppg_af_rf.joblib` for the app.

### 2. Streamlit App

```bash
cd code
streamlit run ppg_app.py
```

The app will:

- plot the uploaded raw signal
- plot the processed signal with detected peaks
- segment the signal into windows
- extract window features
- run Random Forest AF predictions
- export the predictions as CSV

---

## Requirements

- Python 3.9+
- TensorFlow / Keras
- NumPy
- Pandas
- SciPy
- Matplotlib
- Streamlit
- scikit-learn
- joblib

```bash
pip install -r requirements.txt
```
