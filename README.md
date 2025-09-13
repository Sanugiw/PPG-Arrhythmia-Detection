# 🩺 PPG-Based Arrhythmia Detection Using Signal Processing + Deep Learning  

This repository contains a hybrid project that combines **MATLAB signal processing** with **Python deep learning** (RNN/LSTM) to detect arrhythmias from **Photoplethysmography (PPG) signals**.  

---

## Project Overview  
Photoplethysmography (PPG) is a non-invasive optical technique used to monitor cardiovascular health. In this project:  
- MATLAB is used to **preprocess and segment PPG signals**.  
- Deep learning models (RNN/LSTM) are trained in Python to classify **normal vs arrhythmic patterns**.  
- Dataset: [PulseWatch / UMMC PPG Dataset](https://www.synapse.org/Synapse:syn23565056) (controlled access).  

---

## Workflow  

### **1. Signal Processing (MATLAB)**  
- Bandpass filtering (0.5–8 Hz) to remove noise.  
- Normalization (z-score).  
- Peak detection for cycle segmentation.  
- Feature enhancement: Velocity PPG (VPG), Acceleration PPG (APG), HRV metrics.  
- Export preprocessed windows to `.mat` or `.csv`.  

### **2. Deep Learning (Python/Keras)**  
- Load preprocessed PPG segments.  
- Build an **RNN/LSTM/GRU model** for sequence classification.  
- Train/test split and evaluation (accuracy, F1-score, confusion matrix).  
- Compare RNN with CNN and hybrid CNN+RNN models.  

---

## Repository Structure  

```plaintext
├── matlab/                 # MATLAB scripts for preprocessing
│   ├── filter_ppg.m
│   ├── segment_ppg.m
│   └── export_features.m
│
├── python/                 # Deep learning code
│   ├── train_rnn.py
│   ├── evaluate_model.py
│   └── utils.py
│
├── data/                   # (Placeholder) preprocessed PPG segments
│   └── README.md           # Instructions on how to access dataset
│
├── results/                # Saved models, confusion matrices, plots
│
└── README.md               # Project documentation
