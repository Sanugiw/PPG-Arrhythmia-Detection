# PPG-Based Arrhythmia Detection with Deep Learning

This repository presents an end-to-end system for detecting **atrial fibrillation (AF)** from PPG signals, combining **signal processing (MATLAB)** and **deep learning (Python LSTM/RNN)**. It also includes a **Streamlit web application** for real-time prediction and visualization.

---

## Background: PPG for Arrhythmia Detection

Photoplethysmography (PPG) measures pulsatile blood volume changes using light, offering a low-cost and wearable-friendly alternative to ECG for cardiac monitoring. While ECG remains the clinical gold standard, PPG captures the **peripheral manifestation of cardiac activity**, enabling continuous and unobtrusive rhythm analysis.

Among arrhythmias, **atrial fibrillation (AF)** is particularly suitable for detection using PPG due to its:

* **Irregularly irregular pulse intervals**
* Increased **pulse rate variability (PRV)**
* Reduced waveform consistency

---

## Key Advantages

* **Scalable & wearable-friendly:** Uses simple optical sensors (LED + photodiode)
* **Rich physiological information:** Encodes vascular and timing characteristics
* **Machine learning compatibility:** Temporal and morphological variations are learnable

---

## Challenges

* Motion artifacts and noise (especially wrist-based signals)
* Variability due to skin tone, contact pressure, and perfusion
* Ground truth labeling requires synchronized ECG

---

## What Arrhythmia Looks Like in PPG

* **Atrial Fibrillation (AF):** Highly irregular inter-pulse intervals, inconsistent waveform morphology
* **Ectopic beats (PVCs):** Alternating pulse amplitudes
* **Tachycardia/Bradycardia:** Sustained heart rate deviations

---

## Project Overview

This project implements a complete pipeline:

1. **Signal preprocessing (MATLAB):** Filtering, normalization, beat detection
2. **Segmentation:** Fixed-length overlapping windows
3. **Deep learning model (Python):** LSTM/RNN for AF classification
4. **Deployment:** Streamlit interface for real-time analysis

---

## Dataset

* **MIMIC PERform AF Dataset**

  * ~20-minute PPG + ECG recordings
  * 35 subjects (19 AF, 16 Normal)
  * Sampling rate: 125 Hz
  * Labels: AF vs Non-AF

---

## Workflow

### 1. MATLAB Signal Processing

* Band-pass filtering (0.5–8 Hz)
* Baseline removal and normalization
* Peak detection (systolic peaks)
* Inter-beat interval (IBI) computation
* Window segmentation (e.g., 5s windows, 50% overlap)
* Export processed data for training

---

### 2. Deep Learning (Python)

* Load `.mat` or `.csv` windowed data
* Train LSTM/RNN model for AF classification
* Optional data augmentation (noise, scaling, time warping)
* Evaluate performance
* Save trained model (`.h5`)

---

### 3. Streamlit Web Application

* Upload `.mat` or `.csv` PPG signals
* Automatic preprocessing and segmentation
* Window-wise AF prediction
* Visualization of signal segments
* Export predictions as CSV

---

## Repository Structure

```
.
├── matlab/
│   ├── preprocess_ppg.m  
│   ├── detect_beats.m
│   └── export_windows.m
├── code/
│   ├── af_rnn.ipynb
│   ├── example_generate.py 
│   └── ppg_app.py
└── README.md
```

---

## Usage

### Training (Jupyter Notebook)

1. Open `af_rnn.ipynb`
2. Load dataset from `data/ppg_windows.mat`
3. Train and evaluate model
4. Save model to `models/ppg_af_lstm.h5`

---

### Run Web App

```bash
cd code
streamlit run ppg_app.py
```

* Upload PPG signal files
* View AF predictions and plots
* Download results as CSV

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

