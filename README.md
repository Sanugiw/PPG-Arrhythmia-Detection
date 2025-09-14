# PPG-Based Arrhythmia Detection with Deep Learning

This repository combines **signal processing** (MATLAB) + **deep learning** (Python RNN/LSTM) to detect **atrial fibrillation (AF)** from PPG signals. It also includes a **Streamlit web interface** for easy prediction and visualization.

---

## Background: PPG for Arrhythmia Detection

Photoplethysmography (PPG) measures pulsatile blood volume changes using light, offering a low-cost, wearable-friendly window into cardiac rhythm. While ECG is the gold standard for arrhythmia diagnosis, PPG captures the **mechanical consequence** of each heartbeat at the periphery. This makes it attractive for **continuous, unobtrusive screening**, especially on the wrist or finger.

Among arrhythmias, **atrial fibrillation (AF)** is the most suitable target for PPG: it produces **irregularly irregular pulse intervals** and subtle morphology changes (damped or inconsistent dicrotic notch) that PPG can express as **pulse rate variability (PRV)** and beat-to-beat shape variability. Other rhythm disorders (e.g., PVCs, tachy-/brady-arrhythmias) can also perturb the peripheral pulse, but are harder to separate from motion artifacts and vasomotor effects.

**Key advantages:**

* **Scalability & comfort:** works with commodity LEDs/photodiodes; ideal for wearables.
* **Rich signal content:** waveform morphology encodes vascular tone and timing (systolic upstroke, dicrotic notch, reflection).
* **ML-friendly:** irregularity and morphology variability are learnable patterns in both time series and spectro-temporal views.

**Key challenges:**

* Motion artifacts & perfusion changes (especially at the wrist).
* Sensor/skin variability (skin tone, contact pressure, temperature).
* Labeling: definitive rhythm labels usually require synchronized ECG.

---

## What “arrhythmia” looks like in PPG

* **AF:** highly variable **inter-pulse intervals (IPI)**; morphology consistency drops.
* **PVCs/ectopy:** alternating strong/weak pulses; may mimic motion artifacts.
* **Tachy-/Brady-arrhythmia:** sustained rate shifts with altered upstroke time.

---

## Project Overview

This project aims to:

1. Preprocess PPG signals in MATLAB to remove noise and detect beats.
2. Segment signals into fixed-length windows.
3. Train a deep learning RNN/LSTM in Python to classify **AF vs Normal Rhythm**.
4. Provide a Streamlit web interface for prediction and visualization.

---

## Dataset

* **MIMIC PERform AF Dataset**

  * Contains \~20-minute PPG + ECG recordings from 35 subjects (19 AF, 16 Normal) sampled at 125 Hz.
  * Labels: AF vs Non-AF.

---

## Workflow

### 1. MATLAB Signal Processing

* Band-pass filter (0.5–8 Hz) to isolate cardiac components.
* Remove baseline drift and normalize amplitude.
* Detect systolic peaks → compute inter-beat intervals (IBI).
* Segment into overlapping windows (e.g., 5 sec with 50% overlap).
* Export windows and labels for deep learning.

### 2. Deep Learning (Python)

* Load preprocessed windows (`.mat` or `.csv`).
* Build an LSTM/RNN to classify AF vs Normal.
* Apply optional augmentation (noise, scaling, time warping).
* Train, validate, and evaluate model.
* Save trained model as `.h5` for later use.

### 3. Streamlit Web Interface

* Accepts **MATLAB `.mat` files** or **CSV files**.
* Preprocess, segment, and classify uploaded PPG signals.
* Shows AF probability per window and plots example windows.
* Allows **downloading window-wise AF predictions as CSV**.

---

## Repository Structure

```text
.
├── matlab/
│   ├── preprocess_ppg.m        # filter, normalize, beat detection
│   ├── detect_beats.m          # systolic peak detection
│   └── export_windows.m        # segment and save windows
├── code/
│   ├── af_rnn.ipynb            # full Jupyter notebook: LSTM training & evaluation
│   └── ppg_app.py              # Streamlit web interface for predictions
├── data/
│   └── ppg_windows.mat         # preprocessed windows (example)
├── models/
│   └── ppg_af_lstm.h5          # saved trained model
└── README.md
```

---

## Usage

### 1. Training & Evaluation (Jupyter Notebook)

1. Open `af_rnn.ipynb`.
2. Load preprocessed windows from `data/ppg_windows.mat`.
3. Train LSTM/RNN and evaluate performance.
4. Save trained model to `models/ppg_af_lstm.h5`.

### 2. Streamlit Interface

```bash
cd code
streamlit run app.py
```

* Upload `.mat` or `.csv` PPG signals.
* View AF probability, example window plot.
* Download predictions as `ppg_af_predictions.csv`.

---

## Requirements

* Python 3.9+
* TensorFlow / Keras
* NumPy, Pandas, SciPy, Matplotlib
* Streamlit

```bash
pip install tensorflow numpy pandas scipy matplotlib streamlit
```

---

