# PPG Atrial Fibrillation Detection with Explainable Machine Learning

An end-to-end **PPG-based atrial fibrillation (AF) detection system** that combines **signal preprocessing**, **window-based classification**, **SHAP explainability**, and an interactive **Streamlit dashboard**.

This project detects **AF vs Non-AF rhythm** from photoplethysmography (PPG) recordings and supports interpretation through **feature-level explanations** in the notebook workflow.

---

## Key Features

- Binary rhythm classification for **AF vs Non-AF**
- Shared Python preprocessing pipeline for both the notebook and app
- PPG cleaning with **interpolation, band-pass filtering, and z-score normalization**
- Window-based analysis using **5-second overlapping segments**
- **LSTM baseline** for sequence learning
- **Random Forest deployment model** on engineered rhythm features
- **SHAP explainability** for both sequence and feature-based analysis in the notebook
- Interactive **Streamlit dashboard** for upload, preprocessing, prediction, and CSV export

---

## Project Workflow

```text
Raw PPG -> Interpolation -> Band-pass Filter -> Normalization -> Beat Detection -> Windowing -> Feature Extraction -> Random Forest -> AF Prediction
```

Notebook workflow also includes:

```text
Windowed PPG -> LSTM Baseline -> Evaluation -> SHAP Explainability
```

---

## Dataset

This project uses the **MIMIC PERform AF dataset** prepared as CSV recordings inside the repository `data/` folder.

### Dataset Notes

- PPG recordings are sampled at approximately **125 Hz**
- The project loads recordings from `data/mimic_perform_af_csv/mimic_perform_af_csv/`
- The project loads recordings from `data/mimic_perform_non_af_csv/mimic_perform_non_af_csv/`
- Labels use `1` for **AF**
- Labels use `0` for **Non-AF**

> Large raw source data is not documented for redistribution here; this repository expects the prepared CSV structure already present under `data/`.

---

## Data Preprocessing

Implemented in `code/ppg_pipeline.py`.

- Invalid values (`NaN`, `Inf`) are repaired using **linear interpolation**
- Band-pass filtering uses **0.5 to 8.0 Hz**
- Signals are normalized with **z-score normalization**
- Systolic peaks are detected from the processed PPG
- Inter-beat intervals (IBI) are computed from detected peaks
- Signals are segmented into **5.0 second windows**
- Adjacent windows use **2.5 second overlap**

At **125 Hz**, each default window contains **625 samples**.

---

## Engineered Features

The deployed Random Forest model uses 9 window-level features:

- `signal_mean`
- `signal_std`
- `signal_range`
- `signal_energy`
- `peak_count`
- `ibi_mean`
- `ibi_std`
- `ibi_rmssd`
- `ibi_cv`

These features are designed to capture **pulse morphology** and **rhythm irregularity**, which are both relevant for AF detection.

---

## Model Architecture

### 1. LSTM Baseline

The notebook includes a sequence model baseline on windowed PPG:

```text
Input -> LSTM -> Dense -> Output
```

This model is kept as a reference baseline, but it is not the deployment model used by the app.

### 2. Random Forest Deployment Model

The main practical model for this repository is a **Random Forest classifier** trained on engineered window features.

This model is exported to:

```text
models/ppg_af_rf.joblib
```

and is the model used by the Streamlit app.

---

## Explainability

This project includes **SHAP-based explainability** inside `code/af_rnn.ipynb`.

### Included Explanations

- **GradientExplainer / sequence-level SHAP** for the LSTM baseline
- **TreeExplainer SHAP** for the Random Forest deployment model
- Global feature importance visualization
- Per-sample positive and negative feature contribution analysis

### Interpretable Signals

In the Random Forest pipeline, features related to rhythm irregularity such as:

- `ibi_cv`
- `ibi_std`
- `ibi_mean`
- `ibi_rmssd`

play a strong role in distinguishing AF from Non-AF windows.

---

## Results and Performance

The notebook shows that the **Random Forest clearly outperforms the LSTM baseline** on this dataset split.

### LSTM Baseline Test Performance

| Metric | Value |
| --- | --- |
| Test Accuracy | **0.429** |

### LSTM Classification Report

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| Non-AF | 0.43 | 1.00 | 0.60 | 1434 |
| AF | 0.60 | 0.00 | 0.00 | 1912 |

### Random Forest Test Performance

| Metric | Value |
| --- | --- |
| Accuracy | **0.90** |
| Macro F1-score | **0.90** |
| Weighted F1-score | **0.90** |

### Random Forest Classification Report

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| Non-AF | 0.93 | 0.84 | 0.88 | 1434 |
| AF | 0.89 | 0.95 | 0.92 | 1912 |

---

## Streamlit Dashboard

The app is implemented in `code/ppg_app.py`.

### App Capabilities

- Upload a PPG `.csv` file
- Plot the **raw PPG signal**
- Plot the **processed PPG signal** with detected peaks
- Segment the signal into windows
- Extract window-level features
- Predict **AF probability** per window
- Summarize overall AF percentage across windows
- Download predictions as CSV

The app currently uses the **Random Forest model** and does **not** render SHAP explanations directly in the dashboard.

---

## Repository Structure

```text
PPG-Arrhythmia-Detection/
|-- code/
|   |-- af_rnn.ipynb
|   |-- ppg_app.py
|   |-- ppg_pipeline.py
|   `-- example_generate.py
|-- data/
|-- models/
|   |-- ppg_af_lstm.keras
|   `-- ppg_af_rf.joblib
|-- ppg_af_predictions.csv
|-- requirements.txt
`-- README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/PPG-Arrhythmia-Detection.git
cd PPG-Arrhythmia-Detection

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## How to Run

### 1. Run the Notebook

Open:

```text
code/af_rnn.ipynb
```

Run all cells to:

- load the dataset
- preprocess PPG signals
- build window datasets
- train the LSTM baseline
- train and evaluate the Random Forest
- generate SHAP explanations
- save the trained Random Forest model

### 2. Run the Streamlit App

```bash
cd code
streamlit run ppg_app.py
```

---

## CSV Input Format

The app accepts a CSV containing a PPG column.

Example:

```csv
PPG
0.51
0.52
0.49
...
```

If a `PPG` column is not present, the app falls back to using the **first column**.

---

## Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn
- SHAP
- NumPy
- Pandas
- SciPy
- Matplotlib
- Streamlit
- joblib

---

## Reproducibility

- A shared preprocessing pipeline is used across training and inference
- Fixed windowing logic is reused between notebook and app
- The Random Forest model is serialized to `models/ppg_af_rf.joblib`
- Evaluation results in the notebook reflect the saved train/test split used there

---

## Summary

This project combines:

- PPG signal processing
- AF rhythm detection
- engineered-feature machine learning
- notebook-based SHAP explainability
- interactive Streamlit deployment

to create a workflow that is both **practical** and **interpretable** for wearable-style AF screening experiments.
