# PPG-Based Arrhythmia Detection with Deep Learning

This repository contains a project combining **signal processing** (MATLAB) + **deep learning** (Python-RNN/LSTM) to detect atrial fibrillation from PPG signals.

---

## Project Overview

Photoplethysmography (PPG) is a non-invasive way to record blood volume changes using light. This project aims to:

- Preprocess PPG signals in MATLAB to remove noise, detect beats, etc.
- Train a deep learning (RNN/LSTM) model in Python to classify windows of PPG as **AF (atrial fibrillation)** vs **Normal Rhythm**.
- Use an open dataset: **MIMIC PERform AF Dataset** (public access). No special permission needed (beyond downloading).  

---

## Dataset

- **MIMIC PERform AF Dataset**:  
  - Contains ~20-minute PPG + ECG recordings from 35 subjects (19 in AF, 16 normal) sampled at 125 Hz.   
  - Labels: AF vs non-AF.  

---

## Workflow

### 1. MATLAB Signal Processing

- Filter the PPG (band-pass, e.g. 0.5–8 Hz) to isolate cardiac components.  
- Remove baseline wander / drift.  
- Normalize amplitude per segment.  
- Detect systolic peaks (beat detection) → compute inter-beat intervals (IBI).  
- Optionally compute secondary signals (derivative, etc.).  
- Segment into fixed-length windows (30 seconds or shorter) with overlap.

### 2. Deep Learning (Python)

- Load preprocessed windows.  
- Build RNN (LSTM/BiLSTM) model to classify “AF vs Normal.”  
- Apply data augmentation (noise, scaling, time warping) during training.  
- Evaluate with subject-wise split (train/test on different patients).  

---

## Structure

```text
.
├── matlab/
│   ├── preprocess_ppg.m
│   ├── detect_beats.m
│   └── export_windows.m
├── code/
│   ├── af_rnn.ipynb
├── data/
│   └── downloaded and preprocessed data 
├── results/
│   └── models/              # saved trained models
└── README.md
