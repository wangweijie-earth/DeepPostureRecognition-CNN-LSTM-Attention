# DeepPostureRecognition-CNN-LSTM-Attention
Implementation of the CNN + LSTM + Attention deep learning model for human posture classification using eight-channel pressure sensor signals.
# DeepPostureRecognition: CNN + LSTM + Attention Model for Human Posture Classification

This repository contains the implementation of the **CNN + LSTM + Attention** deep learning model used to classify human postures based on **eight-channel pressure sensor time-series signals**.  
The model is designed for high-precision recognition of actions such as walking, running, stair ascent, stair descent, standing, sitting, and other dynamic postures.

---

## 📌 Overview

This project proposes a hybrid deep learning architecture that integrates:

- **CNN** for spatial feature extraction  
- **LSTM** for modeling temporal dependencies  
- **Attention mechanism** for emphasizing key temporal frames  

The combined model achieves:

- **97.7% classification accuracy**  
- **Micro P/R/F1: 0.9778 / 0.9770 / 0.9772**  
- **Micro-AUC: 1.000**  
- Excellent class separability as demonstrated by the confusion matrix and t-SNE visualizations

This repository provides all necessary scripts to reproduce the results used in the corresponding research publication.

---

## 📂 Repository Structure
project/
│── README.md
│── requirements.txt
│── train.py # Main training script
│── data_preprocess.py # Data preprocessing script
│
├── data/ # Preprocessed dataset files (NPZ format)
│ ├── all_splits.npz
│ ├── all_windows.npz
│ ├── class0_windows.npz
│ ├── class1_windows.npz
│ ├── class2_windows.npz
│ ├── class3_windows.npz
│ ├── class4_windows.npz
│ ├── class5_windows.npz
│ ├── class6_windows.npz
│ ├── class7_windows.npz
│ ├── class8_windows.npz
│ └── class9_windows.npz
│
└── .idea/ # IDE configuration 
